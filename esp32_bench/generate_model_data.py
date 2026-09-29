"""Embed the audited INT8 model and training-only golden vectors in flash."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output" / "esp32_audit"
MODEL = AUDIT / "TinyCNN_full_int8.tflite"
VECTORS = AUDIT / "golden_vectors.json"
OUT = Path(__file__).resolve().parent / "main" / "model_data.cc"

model = MODEL.read_bytes()
audit = json.loads((AUDIT / "quantization_audit.json").read_text(encoding="utf-8"))
vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
scale, zero = audit["int8"]["output_quantization"]
assert len(vectors) == 3

def items(values, width=24):
    return "\n".join("  " + ", ".join(str(v) for v in values[i:i + width]) + ","
                     for i in range(0, len(values), width))

parts = ['#include "model_data.h"', '',
         'alignas(16) const unsigned char kModelData[] = {',
         items(list(model)), '};',
         f'const size_t kModelDataLength = {len(model)};',
         f'const char kModelSha256[] = "{audit["int8"]["sha256"]}";',
         'const char* const kGoldenNames[3] = {' + ", ".join(json.dumps(v["file"]) for v in vectors) + '};',
         'const int8_t kGoldenInputs[3][40 * 173] = {']
for vector in vectors:
    values = vector["input_int8"]
    assert len(values) == 40 * 173
    parts.extend(['{', items(values), '},'])
parts.append('};')
parts.append('const int8_t kGoldenOutputs[3][10] = {')
for vector in vectors:
    values = np.clip(np.rint(np.array(vector["output_dequantized"]) / scale + zero), -128, 127).astype(np.int8).tolist()
    parts.append('{' + ', '.join(str(v) for v in values) + '},')
parts.append('};')
OUT.write_text("\n".join(parts) + "\n", encoding="ascii")
print(f"Generated {OUT} ({len(model)} model bytes, 3 golden vectors)")

