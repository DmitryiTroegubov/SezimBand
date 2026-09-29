"""Export report-ready metrics and a row-normalised confusion matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sezimband_2_2"
OUT = ROOT / "output" / "final_test"
OUT.mkdir(parents=True, exist_ok=True)

with (SOURCE / "test_metrics.json").open(encoding="utf-8") as handle:
    summary = json.load(handle)

with (SOURCE / "test_classification_report.csv").open(newline="", encoding="utf-8") as handle:
    report = {row["class_or_metric"]: row for row in csv.DictReader(handle)}

with (SOURCE / "test_confusion_matrix.csv").open(newline="", encoding="utf-8") as handle:
    reader = csv.reader(handle)
    labels = next(reader)[1:]
    rows = list(reader)
    matrix = np.array([[int(value) for value in row[1:]] for row in rows], dtype=int)

assert [row[0] for row in rows] == labels
assert matrix.sum() == summary["test_examples"] == 837
assert np.isclose(np.trace(matrix) / matrix.sum(), summary["accuracy"])

with (SOURCE / "holdout_predictions.csv").open(newline="", encoding="utf-8") as handle:
    prediction_rows = list(csv.DictReader(handle))
check = np.zeros_like(matrix)
index = {label: number for number, label in enumerate(labels)}
for row in prediction_rows:
    check[index[row["class"]], index[row["predicted_class"]]] += 1
assert np.array_equal(check, matrix)

macro = report["macro avg"]
horn = report["car_horn"]
siren = report["siren"]
table_rows = [
    ["Precision / точность", macro["precision"], horn["precision"], siren["precision"]],
    ["Recall / полнота", macro["recall"], horn["recall"], siren["recall"]],
    ["F1", macro["f1-score"], horn["f1-score"], siren["f1-score"]],
    ["Support / число записей", "837", "33", "83"],
]
with (OUT / "table_2_4_final_test_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["Metric", "All classes (macro average)", "car_horn", "siren"])
    writer.writerows(table_rows)

norm = matrix / matrix.sum(axis=1, keepdims=True)
with (OUT / "normalised_confusion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["True class / predicted class", *labels])
    for label, row in zip(labels, norm):
        writer.writerow([label, *[f"{value:.6f}" for value in row]])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(12.5, 10.2), dpi=180)
image = ax.imshow(norm * 100, cmap="Blues", vmin=0, vmax=100, aspect="equal")
ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)), labels)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Normalised confusion matrix - TinyCNN, UrbanSound8K fold 10 (n = 837)", pad=18)
for i in range(len(labels)):
    for j in range(len(labels)):
        value = norm[i, j] * 100
        ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                color="white" if value > 55 else "#183153", fontsize=8)
ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)
fig.colorbar(image, ax=ax, label="Share of true class (%)", fraction=0.046, pad=0.04)
fig.text(0.5, 0.02, "Each row sums to 100%; percentages are calculated from recorded test predictions.",
         ha="center", color="#425466", fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "normalised_confusion_matrix.png", bbox_inches="tight")
fig.savefig(OUT / "normalised_confusion_matrix.svg", bbox_inches="tight")
plt.close(fig)

tp = summary["alert_confusion"]["tp"]
fp = summary["alert_confusion"]["fp"]
fn = summary["alert_confusion"]["fn"]
alert_f1 = 2 * tp / (2 * tp + fp + fn)

section = f"""# Раздел 2.3. Оценка модели и подготовка к интеграции

Итоговая модель TinyCNN была выбрана по валидационной выборке (папка 9) и затем один раз оценена на отложенной папке 10 UrbanSound8K. Это 837 аудиофрагментов из десяти классов. Результаты относятся к компьютерному тесту; тестирование собранного устройства с микрофонами INMP441 здесь не представлено.

**Таблица 2.4. Метрики итогового теста TinyCNN**

| Метрика | Все классы (макросреднее) | car_horn | siren |
|---|---:|---:|---:|
| Точность (precision) | {float(macro['precision']):.3f} | {float(horn['precision']):.3f} | {float(siren['precision']):.3f} |
| Полнота (recall) | {float(macro['recall']):.3f} | {float(horn['recall']):.3f} | {float(siren['recall']):.3f} |
| F1-показатель | {float(macro['f1-score']):.3f} | {float(horn['f1-score']):.3f} | {float(siren['f1-score']):.3f} |
| Число фрагментов | 837 | 33 | 83 |

Общая доля верных ответов (accuracy) составляет **{summary['accuracy']:.4f}**: модель правильно классифицировала {np.trace(matrix)} из 837 фрагментов. Макросреднее F1 составляет **{summary['macro_f1']:.4f}**. Полнота `car_horn` равна **24/33 = {float(horn['recall']):.4f}**, а полнота `siren` — **38/83 = {float(siren['recall']):.4f}**. В частности, 33 фрагмента сирены были ошибочно отнесены к `children_playing`.

Если объединить `car_horn` и `siren` в событие оповещения после десятиклассового предсказания, получаются 65 истинных срабатываний, 40 ложных срабатываний, 51 пропуск и 681 верное отклонение. Для такого бинарного решения precision = **{summary['alert_precision']:.4f}**, recall = **{summary['alert_recall']:.4f}**, F1 = **{alert_f1:.4f}**. Это отдельные показатели, их не следует смешивать с метриками отдельных десяти классов.

![Нормализованная матрица ошибок](normalised_confusion_matrix.png)

**Рисунок 2.5. Нормализованная матрица ошибок итогового теста.** Строки — истинные классы, столбцы — предсказанные. Каждая строка нормирована на число фрагментов данного класса. В текущем PDF номер 2.4 уже занят изображением MFCC, поэтому матрица получает номер 2.5; в ином макете подпись можно изменить на 2.4.

Для подготовленной модели сохранены файл Keras размером **114 533 байта (111,85 КиБ)** и файл TensorFlow Lite float32 размером **28 700 байт (28,03 КиБ)**. Эти числа описывают размер файлов и не показывают потребление оперативной памяти во время работы. Порог уверенности для вибросигнала в данном эксперименте не выбирался и не проверялся; итоговая классификация выполнялась по классу с максимальной вероятностью (`argmax`). Время инференса и пиковая память на XIAO ESP32-S3 не измерены, поэтому численные значения для этих полей пока отсутствуют.

**Исправления относительно исходного PDF.** В нём приведены accuracy 0,65 и macro-F1 0,66; сохранённые результаты этого запуска дают 0,6249 и 0,6354. На странице с описанием обучения число параметров TinyCNN указано как 66 128, тогда как сохранённая модель и таблица сравнения показывают **6 218 обучаемых параметров**. Тест выполнялся на папке 10, а не одновременно на всех десяти папках. Утверждение о тестах на собственных записях INMP441 не подтверждается приложенными результатами и должно сопровождаться отдельным протоколом либо быть удалено.
"""
(OUT / "section_2_3_ready_to_insert.md").write_text(section, encoding="utf-8")

print(json.dumps({
    "accuracy": summary["accuracy"],
    "macro_f1": summary["macro_f1"],
    "car_horn_recall": float(horn["recall"]),
    "siren_recall": float(siren["recall"]),
    "alert_f1": alert_f1,
    "output": str(OUT),
}, ensure_ascii=False, indent=2))
