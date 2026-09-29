"""Optional independent ESC-50 cross-dataset test; never train on ESC-50 here."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiment import CLASSES, clip_predictions, csv_write, features, metrics, read_audio, windows


class Exported:
    def __init__(self, config):
        self.weights = [np.asarray(m) for m in config['weights_input_by_output']]
        self.biases = [np.asarray(b) for b in config['biases']]
        self.architecture = config['architecture']
    def predict_proba(self, X):
        h = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b
            if i < len(self.weights) - 1: h = np.maximum(h, 0)
        h -= np.max(h, axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True, help='ESC-50 extracted directory')
    p.add_argument('--model', type=Path, required=True, help='Real UrbanSound8K-trained model.json')
    p.add_argument('--out', type=Path, default=Path('results/esc50'))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model_data = json.loads(args.model.read_text())
    if model_data['classes_in_order'] != CLASSES:
        raise ValueError('Unexpected class ordering')
    if model_data['source'] != 'UrbanSound8K holdout folds 1-8/9/10':
        raise ValueError('Model has an unexpected training provenance')
    path = args.data / 'meta' / 'esc50.csv'
    with path.open(newline='') as f: rows = list(csv.DictReader(f))
    if len(rows) != 2000 or not {'filename', 'category', 'src_file', 'fold'} <= set(rows[0]):
        raise ValueError('Expected the complete, original 2000-row ESC-50 metadata')
    # ESC folds remain untouched; this is entirely outside UrbanSound8K fitting.
    X, y, ids = [], [], []
    for i, r in enumerate(rows):
        if r['category'] not in ('siren', 'car_horn'):
            label = 2
        else:
            label = CLASSES.index(r['category'])
        source = args.data / 'audio' / r['filename']
        if not source.is_file(): raise FileNotFoundError(source)
        for segment in windows(read_audio(source)):
            X.append(features(segment)); y.append(label); ids.append(i)
        r['_label'] = label
        r['slice_file_name'] = r['filename']
        r['fsID'] = r['src_file']
    X, y, ids = np.asarray(X), np.asarray(y), np.asarray(ids)
    mean = np.asarray(model_data['standard_scaler_mean'])
    scale = np.asarray(model_data['standard_scaler_scale'])
    class Scaler:
        def transform(self, x): return (x - mean) / scale
    actual, predicted, detail = clip_predictions(Exported(model_data), Scaler(), X, y, ids,
                                                 np.ones(len(rows), dtype=bool), rows)
    csv_write(args.out / 'external_predictions.csv', detail)
    result = {'status': 'ESC50_cross_dataset_computer_test',
              'warning': 'No prototype microphone recordings or deployed firmware were evaluated',
              'scores': metrics(actual, predicted)}
    (args.out / 'external_metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['scores'], indent=2))


if __name__ == '__main__': main()
