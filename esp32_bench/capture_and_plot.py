"""Capture raw ESP32 logs and make reproducible device timing evidence."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import serial

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "esp32_audit"
OUT.mkdir(parents=True, exist_ok=True)


def parse_record(line: str) -> dict[str, str] | None:
    if not line.startswith("SB_"):
        return None
    parts = line.strip().split(",")
    result = {"record": parts[0]}
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    lines = []
    armed = False
    deadline = time.monotonic() + args.seconds
    with serial.Serial(args.port, 115200, timeout=0.5) as port:
        # The board restarts after flashing. Catch its next log sequence.
        while time.monotonic() < deadline:
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            print(line, flush=True)
            if line.startswith("SB_META,"):
                lines = [line]
                armed = True
            elif armed:
                lines.append(line)
            if armed and line == "SB_DONE":
                break

    (OUT / "board_serial_raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = [r for line in lines if (r := parse_record(line))]
    errors = [r for r in records if r["record"] == "SB_ERROR"]
    if errors:
        raise RuntimeError(f"Board reported: {errors}")
    if not any(r["record"] == "SB_DONE" for r in records):
        raise RuntimeError("No complete benchmark in captured serial log")
    meta = next(r for r in records if r["record"] == "SB_META")
    allocation = next(r for r in records if r["record"] == "SB_ALLOC")
    simulated = next(r for r in records if r["record"] == "SB_SIM")
    golden = [r for r in records if r["record"] == "SB_GOLDEN"]
    runs = [r for r in records if r["record"] == "SB_RUN"]
    if len(golden) != 3 or not all(r["pass"] == "1" for r in golden):
        raise RuntimeError("One or more on-device golden comparisons failed")
    if len(runs) != int(meta["runs"]):
        raise RuntimeError("Incomplete timed run series")
    times_ms = [int(r["invoke_us"]) / 1000 for r in runs]

    with (OUT / "board_inference_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run", "invoke_ms", "internal_free_bytes", "psram_free_bytes"])
        for run in runs:
            writer.writerow([run["index"], int(run["invoke_us"]) / 1000,
                             run["internal_free"], run["psram_free"]])

    audit = json.loads((OUT / "quantization_audit.json").read_text(encoding="utf-8"))
    if meta["model_sha256"] != audit["int8"]["sha256"]:
        raise RuntimeError("Device model hash does not match audited TFLite model")
    heaps = {r["stage"]: r for r in records if r["record"] == "SB_HEAP"}
    result = {
        "status": "measured_on_connected_XIAO_ESP32S3",
        "scope": "INT8 inference on precomputed MFCC; no microphone, MFCC, motor or display measured",
        "model_sha256": meta["model_sha256"],
        "model_bytes": int(meta["model_bytes"]),
        "cpu_mhz": int(meta["cpu_mhz"]),
        "arena_location": allocation["arena_location"],
        "arena_capacity_bytes": int(allocation["arena_capacity"]),
        "arena_used_bytes": int(allocation["arena_used"]),
        "allocate_us": int(allocation["allocate_us"]),
        "input_bytes": int(allocation["input_bytes"]),
        "output_bytes": int(allocation["output_bytes"]),
        "golden_passes": len(golden),
        "runs": len(runs),
        "invoke_ms_min": min(times_ms),
        "invoke_ms_median": statistics.median(times_ms),
        "invoke_ms_mean": statistics.mean(times_ms),
        "invoke_ms_p95": sorted(times_ms)[int(0.95 * (len(times_ms) - 1))],
        "invoke_ms_max": max(times_ms),
        "internal_free_before_bytes": int(heaps["before_model"]["internal_free"]),
        "internal_free_after_allocation_bytes": int(heaps["after_allocate_tensors"]["internal_free"]),
        "internal_min_free_after_runs_bytes": int(heaps["after_runs"]["internal_min_free"]),
        "psram_free_before_bytes": int(heaps["before_model"]["psram_free"]),
        "psram_free_after_allocation_bytes": int(heaps["after_allocate_tensors"]["psram_free"]),
        "psram_free_with_pcm_bytes": int(heaps["with_sim_pcm"]["psram_free"]),
        "simulated_pcm_bytes": int(simulated["pcm_bytes"]),
        "simulated_capture_wait_ms": int(simulated["capture_wait_us"]) / 1000,
        "simulated_feature_load_ms": int(simulated["feature_load_us"]) / 1000,
        "simulated_cycle_invoke_ms": int(simulated["invoke_us"]) / 1000,
        "simulated_postprocess_ms": int(simulated["postprocess_us"]) / 1000,
        "simulated_ui_wait_ms": int(simulated["ui_wait_us"]) / 1000,
        "mfcc_on_device_measured": False,
    }
    (OUT / "board_benchmark_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    axes[0].plot(range(len(times_ms)), times_ms, marker=".", markersize=3, linewidth=1, color="#218f88")
    axes[0].axhline(result["invoke_ms_median"], color="#ad4e37", linestyle="--",
                    label=f"Median {result['invoke_ms_median']:.2f} ms")
    axes[0].set(xlabel="Run after 5 warm-ups", ylabel="Invoke time (ms)", title="Measured ESP32-S3 INT8 inference")
    axes[0].legend()
    axes[1].bar(["Tensor arena used", "Arena reserved", "Claimed RAM"],
                [result["arena_used_bytes"] / 1024, result["arena_capacity_bytes"] / 1024, 150000 / 1024],
                color=["#218f88", "#6e99c2", "#bbbbbb"])
    axes[1].set(ylabel="KiB", title=f"Memory on board ({result['arena_location']})")
    fig.tight_layout()
    fig.savefig(OUT / "board_benchmark_evidence.png", bbox_inches="tight")
    fig.savefig(OUT / "board_benchmark_evidence.svg", bbox_inches="tight")

    fig2, axis = plt.subplots(figsize=(9, 3.8), dpi=180)
    names = ["PCM wait", "Feature load", "INT8 invoke", "Postprocess", "UI wait"]
    values = [result["simulated_capture_wait_ms"], result["simulated_feature_load_ms"],
              result["simulated_cycle_invoke_ms"], result["simulated_postprocess_ms"],
              result["simulated_ui_wait_ms"]]
    axis.barh(names, values, color=["#bbbbbb", "#bbbbbb", "#218f88", "#218f88", "#bbbbbb"])
    axis.set(xlabel="Elapsed milliseconds (log scale)",
             title="Synthetic cycle: grey stages are substitutes, MFCC is absent")
    axis.set_xscale("log")
    for i, value in enumerate(values):
        axis.text(max(value, 0.001) * 1.2, i, f"{value:.3f}", va="center", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(OUT / "synthetic_cycle_evidence.png", bbox_inches="tight")
    fig2.savefig(OUT / "synthetic_cycle_evidence.svg", bbox_inches="tight")

    report = f"""# Проверка заявленных характеристик SezimBand

## Что измерено

Выбранная модель TinyCNN занимает {audit['keras_checkpoint_bytes']:,} байт в формате Keras, {audit['float32']['size_bytes']:,} байт в FP32 TFLite и {audit['int8']['size_bytes']:,} байт после полной INT8-квантизации. На компьютере тест по всем {audit['test_examples']} примерам папки 10 дал accuracy {audit['float32_scores']['accuracy']:.4f} для FP32 и {audit['int8_scores']['accuracy']:.4f} для INT8; полнота `car_horn` изменилась с {audit['float32_scores']['car_horn_recall']:.4f} до {audit['int8_scores']['car_horn_recall']:.4f}, `siren` — с {audit['float32_scores']['siren_recall']:.4f} до {audit['int8_scores']['siren_recall']:.4f}.

На XIAO ESP32-S3 при частоте CPU {result['cpu_mhz']} МГц запущена та же INT8-модель (SHA-256 `{result['model_sha256']}`). Все {result['golden_passes']} эталонных сравнения выходов прошли. После пяти прогревочных вызовов измерены {result['runs']} вызовов `Invoke()`: медиана {result['invoke_ms_median']:.3f} мс, 95-й процентиль {result['invoke_ms_p95']:.3f} мс, диапазон {result['invoke_ms_min']:.3f}–{result['invoke_ms_max']:.3f} мс. Тензорная арена размещена в {result['arena_location']}; выделено {result['arena_capacity_bytes']:,} байт, использовано по отчёту TFLite Micro {result['arena_used_bytes']:,} байт.

Синтетический цикл дополнительно выделил {result['simulated_pcm_bytes']:,} байт для 4 секунд PCM. Его ожидание записи длилось {result['simulated_capture_wait_ms']:.3f} мс и задано прошивкой. Загрузка готового MFCC заняла {result['simulated_feature_load_ms']:.3f} мс, отдельный `Invoke()` — {result['simulated_cycle_invoke_ms']:.3f} мс, обработка выхода — {result['simulated_postprocess_ms']:.3f} мс. Пауза {result['simulated_ui_wait_ms']:.3f} мс замещала отсутствующие дисплей и мотор.

## Что осталось неподтверждённым

Число 0,8 МБ относится примерно к другой архитектуре — FP32 TFLite BaselineCNN имеет 886 560 байт. Для выбранной TinyCNN заявленные 0,8 МБ и 80 кБ не соответствуют сохранённым файлам. Общая RAM изделия не измерена: значение арены не включает PCM, стек, DSP, драйверы и периферию. Захват I²S, вычисление MFCC на плате и физическое включение дисплея/мотора не выполнялись. Поэтому 230 мс для MFCC и 4,29 с полного рабочего цикла остаются расчётными предположениями. Арифметическая сумма исходных предположений равна 4,289 с; вычислительная часть — 0,289 с.

Доказательства: `quantization_audit.json`, `test_predictions_comparison.csv`, `board_serial_raw.log`, `board_inference_runs.csv`, `quantization_evidence.png`, `board_benchmark_evidence.png`, `synthetic_cycle_evidence.png`.
"""
    (OUT / "verified_report_ru.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
