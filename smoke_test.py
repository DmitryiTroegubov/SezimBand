"""Synthetic plumbing check. NOT an audio classifier performance evaluation."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from experiment import SAMPLE_RATE, WINDOW_SAMPLES, features, log_mel, partition


def main():
    assert features(np.zeros(WINDOW_SAMPLES, dtype=np.float32)).shape == (256,)
    assert log_mel(np.zeros(WINDOW_SAMPLES, dtype=np.float32)).shape == (198, 32)
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) / 'data'
        meta = root / 'metadata'
        meta.mkdir(parents=True)
        rows = []
        classes = [(1, 'car_horn'), (8, 'siren'), (0, 'air_conditioner')]
        time = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
        for fold in range(1, 11):
            directory = root / 'audio' / f'fold{fold}'
            directory.mkdir(parents=True)
            for rep in range(3):
                for cid, name in classes:
                    fsid = fold * 1000 + rep * 10 + cid
                    file = f'{fsid}-{cid}-0-0.wav'
                    if cid == 1:
                        sig = np.sin(2 * np.pi * (350 + rep * 10) * time) * .7
                    elif cid == 8:
                        sig = np.sin(2 * np.pi * (550 * time + 150 * time * time)) * .7
                    else:
                        sig = rng.normal(size=len(time)) * .15
                    wavfile.write(directory / file, SAMPLE_RATE, (sig * 25000).astype(np.int16))
                    rows.append({'slice_file_name': file, 'fold': fold, 'fsID': fsid,
                                 'class': name, 'classID': cid, '_fold': fold})
        with (meta / 'UrbanSound8K.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['slice_file_name', 'fold', 'fsID', 'class', 'classID'])
            writer.writeheader()
            writer.writerows([{k: r[k] for k in writer.fieldnames} for r in rows])
        train, val, test = partition(rows, 10, 9)
        assert (sum(train), sum(val), sum(test)) == (72, 9, 9)
        out = Path(scratch) / 'results'
        cmd = [sys.executable, str(Path(__file__).parent / 'experiment.py'),
               '--data', str(root), '--out', str(out)]
        run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode:
            raise RuntimeError(run.stderr)
        metrics = json.loads((out / 'metrics.json').read_text())
        assert metrics['status'] == 'trained_on_real_UrbanSound8K'  # only the *test fixture* imitates that layout
        assert metrics['models'] and (out / 'model.json').is_file()
        model = json.loads((out / 'model.json').read_text())
        assert len(model['standard_scaler_mean']) == 256
        assert len(model['weights_input_by_output'][0]) == 256
        print('PASS: synthetic audio fixture, shape, fold grouping, training, evaluation and weight export')
        print('WARNING: these generated tones are NOT UrbanSound8K and no project accuracy was measured')


if __name__ == '__main__':
    main()
