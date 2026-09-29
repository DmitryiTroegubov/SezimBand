"""Reproducible full-INT8 conversion and held-out accuracy audit.

Run with .venv_tf/Scripts/python.exe quantize_and_verify.py. Calibration uses
training fold audio only. Final comparison uses the reserved fold 10.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sezimband_2_2"
OUT = ROOT / "output" / "esp32_audit"
OUT.mkdir(parents=True, exist_ok=True)
CFG = json.loads((SOURCE / "feature_config.json").read_text(encoding="utf-8"))
CLASSES = json.loads((SOURCE / "class_mapping.json").read_text(encoding="utf-8"))["classes"]


def features(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=CFG["sr"], mono=True,
                             res_type=CFG["res_type"], dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=CFG["n_fft"], hop_length=CFG["hop_length"],
        win_length=CFG["win_length"], window=CFG["window"],
        center=CFG["center"], pad_mode=CFG["pad_mode"],
        power=CFG["power"], n_mels=CFG["n_mels"], fmin=CFG["fmin"],
        fmax=CFG["fmax"], htk=CFG["htk"], norm=CFG["mel_norm"])
    logmel = librosa.power_to_db(mel, ref=CFG["db_ref"],
                                 amin=CFG["db_amin"], top_db=CFG["db_top"])
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=CFG["n_mfcc"],
                                dct_type=CFG["dct_type"],
                                norm=CFG["dct_norm"], lifter=CFG["lifter"])
    resized = cv2.resize(mfcc, (CFG["resized_frames"], CFG["n_mfcc"]),
                         interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32)[None, :, :, None]


def load_manifest() -> list[dict[str, str]]:
    with (SOURCE / "audio_manifest.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sample_training(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    # Stable selection: no test or validation audio can enter calibration.
    train = [r for r in rows if r["split"] == "train"]
    by_class = {c: sorted((r for r in train if r["class"] == c),
                          key=lambda r: r["slice_file_name"]) for c in CLASSES}
    rng = np.random.default_rng(42)
    sampled = []
    per_class = count // len(CLASSES)
    for label in CLASSES:
        group = by_class[label]
        chosen = rng.choice(len(group), size=min(per_class, len(group)), replace=False)
        sampled.extend(group[int(i)] for i in chosen)
    return sampled


def int8_input(float_input: np.ndarray, details: dict) -> np.ndarray:
    scale, zero = details["quantization"]
    assert details["dtype"] == np.int8 and scale > 0
    return np.clip(np.rint(float_input / scale + zero), -128, 127).astype(np.int8)


def run_interpreter(interpreter: tf.lite.Interpreter, data: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], int8_input(data, inp) if inp["dtype"] == np.int8 else data)
    interpreter.invoke()
    values = interpreter.get_tensor(out["index"])[0]
    if out["dtype"] == np.int8:
        scale, zero = out["quantization"]
        return (values.astype(np.float32) - zero) * scale
    return values.astype(np.float32)


def describe_model(path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    operations = [op["op_name"] for op in interpreter._get_ops_details()]
    return {"interpreter": interpreter, "details": {
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "input_type": str(input_detail["dtype"]),
        "output_type": str(output_detail["dtype"]),
        "input_shape": input_detail["shape"].tolist(),
        "output_shape": output_detail["shape"].tolist(),
        "input_quantization": list(input_detail["quantization"]),
        "output_quantization": list(output_detail["quantization"]),
        "operations": operations,
    }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-count", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=0,
                        help="For debugging only; 0 means the complete reserved fold")
    args = parser.parse_args()
    rows = load_manifest()
    calibration = sample_training(rows, args.calibration_count)
    if len(calibration) != args.calibration_count:
        raise ValueError("Calibration count must divide equally into 10 available classes")
    with (OUT / "calibration_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["slice_file_name", "class", "fold", "split"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in calibration)

    print("Extracting calibration MFCCs...", flush=True)
    calibration_data = []
    for index, row in enumerate(calibration, 1):
        calibration_data.append(features(row["path"]))
        if index % 50 == 0:
            print(f"  {index}/{len(calibration)}", flush=True)
    model = tf.keras.models.load_model(SOURCE / "TinyCNN_best.keras", compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([item] for item in calibration_data)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int8_bytes = converter.convert()
    int8_path = OUT / "TinyCNN_full_int8.tflite"
    int8_path.write_bytes(int8_bytes)
    print(f"INT8 model: {len(int8_bytes)} bytes", flush=True)

    int8 = describe_model(int8_path)
    fp32_path = SOURCE / "TinyCNN_float32.tflite"
    fp32 = describe_model(fp32_path)
    assert int8["details"]["input_type"] == "<class 'numpy.int8'>"
    assert int8["details"]["output_type"] == "<class 'numpy.int8'>"
    assert not any(op == "DEQUANTIZE" for op in int8["details"]["operations"])

    test = [r for r in rows if r["split"] == "test"]
    if args.max_test:
        test = test[:args.max_test]
    print(f"Evaluating {len(test)} reserved test files...", flush=True)
    label_index = {label: i for i, label in enumerate(CLASSES)}
    actual, fp_pred, int8_pred = [], [], []
    comparison = []
    golden = []
    for index, row in enumerate(test, 1):
        x = features(row["path"])
        p32 = run_interpreter(fp32["interpreter"], x)
        p8 = run_interpreter(int8["interpreter"], x)
        y = label_index[row["class"]]
        a = int(np.argmax(p32))
        b = int(np.argmax(p8))
        actual.append(y)
        fp_pred.append(a)
        int8_pred.append(b)
        comparison.append([row["slice_file_name"], row["class"], CLASSES[a], CLASSES[b],
                           int(a == b), float(np.max(np.abs(p32 - p8)))])
        if index % 100 == 0 or index == len(test):
            print(f"  {index}/{len(test)}", flush=True)

    # Golden vectors are calibration examples, never held-out examples.
    for row, x in list(zip(calibration, calibration_data))[:3]:
        q = int8_input(x, int8["interpreter"].get_input_details()[0])[0]
        logits = run_interpreter(int8["interpreter"], x)
        golden.append({"file": row["slice_file_name"], "label": row["class"],
                       "input_int8": q.reshape(-1).tolist(),
                       "output_dequantized": logits.tolist(),
                       "predicted_index": int(np.argmax(logits))})
    (OUT / "golden_vectors.json").write_text(json.dumps(golden), encoding="utf-8")

    with (OUT / "test_predictions_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["slice_file_name", "true_class", "float32_predicted", "int8_predicted",
                         "predictions_agree", "max_abs_probability_difference"])
        writer.writerows(comparison)

    def scores(predictions: list[int]) -> dict:
        return {"accuracy": float(accuracy_score(actual, predictions)),
                "macro_f1": float(f1_score(actual, predictions, average="macro", labels=range(10))),
                "car_horn_recall": float(recall_score(actual, predictions, labels=[1], average=None)[0]),
                "siren_recall": float(recall_score(actual, predictions, labels=[8], average=None)[0])}

    result = {"scope": "computer test, UrbanSound8K fold 10; device timings not included",
              "tensorflow_version": tf.__version__,
              "calibration_examples": len(calibration),
              "calibration_split": "train only, folds 1-8",
              "test_examples": len(test),
              "full_test": args.max_test == 0,
              "keras_checkpoint_bytes": (SOURCE / "TinyCNN_best.keras").stat().st_size,
              "float32": fp32["details"], "int8": int8["details"],
              "float32_scores": scores(fp_pred), "int8_scores": scores(int8_pred),
              "prediction_agreement": float(np.mean(np.array(fp_pred) == np.array(int8_pred)))}
    (OUT / "quantization_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=180)
    sizes = [result["keras_checkpoint_bytes"], fp32["details"]["size_bytes"], int8["details"]["size_bytes"]]
    axes[0].bar(["Keras", "TFLite FP32", "TFLite INT8"], np.array(sizes) / 1024, color=["#44607c", "#6e99c2", "#218f88"])
    axes[0].set(ylabel="File size (KiB)", title="Measured model file sizes")
    for i, value in enumerate(sizes):
        axes[0].text(i, value / 1024 + 1, f"{value:,} B", ha="center", fontsize=8)
    names = ["Accuracy", "Macro F1", "Horn recall", "Siren recall"]
    keys = ["accuracy", "macro_f1", "car_horn_recall", "siren_recall"]
    x = np.arange(len(keys))
    axes[1].bar(x - .17, [result["float32_scores"][k] for k in keys], .34, label="FP32", color="#6e99c2")
    axes[1].bar(x + .17, [result["int8_scores"][k] for k in keys], .34, label="INT8", color="#218f88")
    axes[1].set_xticks(x, names, rotation=25, ha="right")
    axes[1].set(ylim=(0, 1), ylabel="Score", title="Same reserved test fold")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "quantization_evidence.png", bbox_inches="tight")
    fig.savefig(OUT / "quantization_evidence.svg", bbox_inches="tight")
    print(json.dumps({k: result[k] for k in ["float32_scores", "int8_scores", "prediction_agreement"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
