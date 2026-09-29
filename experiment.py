"""Grouped UrbanSound8K log-mel + compact MLP/linear baseline experiment."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CLASSES = ['car_horn', 'siren', 'non_target']
TARGETS = {'car_horn': 0, 'siren': 1}
SAMPLE_RATE = 16000
WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = 32000
WINDOW_HOP = 16000
FFT_SIZE = 400
FRAME_HOP = 160
MEL_BANDS = 32
TIME_BINS = 8
SEED = 20260915
EXPECTED = {'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
            'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'}


def csv_write(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(root):
    path = root / 'metadata' / 'UrbanSound8K.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}; extract the COMPLETE official dataset first')
    with path.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'slice_file_name', 'fold', 'fsID', 'class', 'classID'} <= set(rows[0]):
        raise ValueError('Unexpected metadata schema')
    groups = defaultdict(set)
    for row in rows:
        c = row['class'].strip()
        if c not in EXPECTED or int(row['fold']) not in range(1, 11):
            raise ValueError(f'Bad class/fold: {row}')
        if c != {1: 'car_horn', 8: 'siren'}.get(int(row['classID']), c):
            raise ValueError(f'classID mismatch: {row}')
        p = root / 'audio' / f"fold{row['fold']}" / row['slice_file_name']
        if not p.is_file():
            raise FileNotFoundError(f'Missing audio: {p}')
        row['_path'] = str(p)
        row['_fold'] = int(row['fold'])
        row['_label'] = TARGETS.get(c, 2)
        groups[row['fsID']].add(row['_fold'])
    multi = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    if multi:
        print(f'WARNING: {len(multi)} fsID(s) span multiple folds (known UrbanSound8K annotation overlap):')
        for k, v in multi.items():
            print(f'  fsID={k} folds={v}')
        print('These are different sound-class annotations from the same Freesound source, not data leakage.')
    return rows


def read_audio(path):
    try:
        import soundfile as sf
        x, sr = sf.read(path, dtype='float32')
    except Exception:
        sr, raw = wavfile.read(path)
        if np.issubdtype(raw.dtype, np.integer):
            info = np.iinfo(raw.dtype)
            if np.issubdtype(raw.dtype, np.unsignedinteger):
                x = (raw.astype(np.float32) - (info.max + 1) / 2) / ((info.max + 1) / 2)
            else:
                x = raw.astype(np.float32) / max(abs(info.min), info.max)
        else:
            x = raw.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not np.isfinite(x).all() or sr <= 0:
        raise ValueError(f'Invalid waveform: {path}')
    if sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return np.clip(x, -1, 1)


def windows(audio):
    if len(audio) <= WINDOW_SAMPLES:
        return [np.pad(audio, (0, max(0, WINDOW_SAMPLES - len(audio))))]
    starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, WINDOW_HOP))
    if starts[-1] != len(audio) - WINDOW_SAMPLES:
        starts.append(len(audio) - WINDOW_SAMPLES)
    return [audio[s:s + WINDOW_SAMPLES] for s in starts]


def mel_filters():
    # HTK mel scale; unnormalized triangular filters, 80 Hz to 7600 Hz.
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    h = mel_to_hz(np.linspace(hz_to_mel(80), hz_to_mel(7600), MEL_BANDS + 2))
    frequencies = np.fft.rfftfreq(FFT_SIZE, 1 / SAMPLE_RATE)
    out = []
    for low, mid, high in zip(h[:-2], h[1:-1], h[2:]):
        left = (frequencies - low) / (mid - low)
        right = (high - frequencies) / (high - mid)
        out.append(np.maximum(0, np.minimum(left, right)))
    return np.asarray(out, dtype=np.float32)


FILTERS = mel_filters()
HANN = np.hanning(FFT_SIZE + 1)[:-1].astype(np.float32)  # periodic Hann


def log_mel(x):
    frames = np.lib.stride_tricks.sliding_window_view(x, FFT_SIZE)[::FRAME_HOP]
    power = np.abs(np.fft.rfft(frames * HANN, axis=-1)) ** 2
    energy = power @ FILTERS.T
    return np.log(np.maximum(energy, 1e-6)).astype(np.float32)  # shape 198 x 32


def features(x):
    spec = log_mel(x)
    temporal = np.stack([np.mean(part, axis=0) for part in np.array_split(spec, TIME_BINS)])
    return temporal.T.flatten().astype(np.float32)  # frequency-major, 32 x 8


def extract(rows, out):
    xs, ys, clip_ids, specs = [], [], [], []
    first = None
    for i, row in enumerate(rows):
        audio = read_audio(row['_path'])
        if first is None and row['_label'] in (0, 1):
            first = (audio, row['class'], row['slice_file_name'])
        for w in windows(audio):
            xs.append(features(w))
            ys.append(row['_label'])
            clip_ids.append(i)
    if first:
        audio, name, file = first
        wave = windows(audio)[0]
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
        axes[0].plot(np.arange(len(wave)) / SAMPLE_RATE, wave, linewidth=.5)
        axes[0].set(ylabel='Normalized amplitude', title=f'{name}: {file}')
        axes[1].imshow(log_mel(wave).T, aspect='auto', origin='lower', extent=(0, 2, 0, MEL_BANDS))
        axes[1].set(xlabel='Time (s)', ylabel='Mel band', title='32-band log-mel, 25 ms frame, 10 ms hop')
        fig.savefig(out / 'feature_example.png', dpi=160)
        plt.close(fig)
    return np.asarray(xs), np.asarray(ys), np.asarray(clip_ids)


def clip_predictions(model, scaler, X, y, ids, mask, rows):
    indices = np.flatnonzero(mask[ids])
    probs = model.predict_proba(scaler.transform(X[indices]))
    aggregated = defaultdict(list)
    for ix, p in zip(indices, probs):
        aggregated[int(ids[ix])].append(p)
    truth, pred, details = [], [], []
    for clip_ix in sorted(aggregated):
        mean = np.mean(aggregated[clip_ix], axis=0)
        expected = int(rows[clip_ix]['_label'])
        estimate = int(np.argmax(mean))
        truth.append(expected)
        pred.append(estimate)
        details.append({'filename': rows[clip_ix]['slice_file_name'],
                        'original_source_fsID': rows[clip_ix]['fsID'],
                        'actual': CLASSES[expected], 'predicted': CLASSES[estimate],
                        **{f'prob_{c}': round(float(mean[j]), 6) for j, c in enumerate(CLASSES)}})
    return np.asarray(truth), np.asarray(pred), details


def metrics(y, p):
    scores = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    binary_y = y != 2
    binary_p = p != 2
    bp, br, bf, _ = precision_recall_fscore_support(binary_y, binary_p, average='binary', zero_division=0)
    return {'n_clips': len(y), 'accuracy': float(accuracy_score(y, p)),
            'per_class': {c: {'precision': float(scores[0][i]), 'recall': float(scores[1][i]),
                              'f1': float(scores[2][i]), 'support': int(scores[3][i])}
                          for i, c in enumerate(CLASSES)},
            'alert_precision': float(bp), 'alert_recall': float(br), 'alert_f1': float(bf),
            'confusion_matrix_rows_true_columns_predicted': confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}


def partition(rows, test_fold, validation_fold):
    train = np.asarray([r['_fold'] not in (test_fold, validation_fold) for r in rows])
    val = np.asarray([r['_fold'] == validation_fold for r in rows])
    test = np.asarray([r['_fold'] == test_fold for r in rows])
    assert np.all(train.astype(int) + val.astype(int) + test.astype(int) == 1)
    for a, b, name in ((train, val, 'train/val'), (train, test, 'train/test'), (val, test, 'val/test')):
        overlap = {rows[i]['fsID'] for i in np.flatnonzero(a)} & {rows[i]['fsID'] for i in np.flatnonzero(b)}
        if overlap:
            print(f'WARNING: {len(overlap)} fsID(s) shared between {name}: {sorted(overlap)}')
    return train, val, test


def fit_models(X, y, ids, train, val, rows, out=None):
    ix = np.flatnonzero(train[ids])
    scaler = StandardScaler().fit(X[ix])
    xx = scaler.transform(X[ix]); yy = y[ix]
    linear = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced', random_state=SEED).fit(xx, yy)
    candidates = {'logistic_regression': (linear, [])}
    mlp = MLPClassifier(hidden_layer_sizes=(24, 12), activation='relu', solver='adam',
                        learning_rate_init=.001, batch_size=128, max_iter=1, warm_start=False,
                        random_state=SEED, shuffle=True)
    curve = []
    # Partial fitting reuses network state; only TRAIN windows influence parameter updates.
    weights = np.asarray([len(yy) / (3 * max(1, np.sum(yy == c))) for c in yy], dtype=float)
    best_f1 = (-1.0, -1.0); best_state = None; best_epoch = 0
    for epoch in range(1, 36):
        mlp.partial_fit(xx, yy, sample_weight=weights, classes=np.array([0, 1, 2]))
        ty, tp, _ = clip_predictions(mlp, scaler, X, y, ids, train, rows)
        vy, vp, _ = clip_predictions(mlp, scaler, X, y, ids, val, rows)
        vm = metrics(vy, vp)
        validation_ix = np.flatnonzero(val[ids])
        validation_loss = log_loss(y[validation_ix], mlp.predict_proba(scaler.transform(X[validation_ix])), labels=[0, 1, 2])
        curve.append({'epoch': epoch, 'train_loss': float(mlp.loss_), 'validation_window_loss': float(validation_loss),
                      'train_alert_f1': metrics(ty, tp)['alert_f1'],
                      'val_alert_f1': vm['alert_f1']})
        if (vm['alert_f1'], vm['accuracy']) > best_f1:
            best_f1 = (vm['alert_f1'], vm['accuracy'])
            best_epoch = epoch
            best_state = ([v.copy() for v in mlp.coefs_], [v.copy() for v in mlp.intercepts_])
    mlp.coefs_, mlp.intercepts_ = best_state
    candidates['mlp_24_12'] = (mlp, curve[:best_epoch])
    validation = {}
    for name, (model, history) in candidates.items():
        vy, vp, _ = clip_predictions(model, scaler, X, y, ids, val, rows)
        validation[name] = metrics(vy, vp)
    # Select on VALIDATION only. Tie: smaller model, then lexical name.
    name = sorted(candidates, key=lambda n: (-validation[n]['alert_f1'],
                                             -validation[n]['accuracy'],
                                             parameter_count(candidates[n][0]), n))[0]
    if out is not None:
        csv_write(out / 'training_curve.csv', curve)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([r['epoch'] for r in curve], [r['train_loss'] for r in curve], label='Training loss')
        ax.plot([r['epoch'] for r in curve], [r['validation_window_loss'] for r in curve], color='tab:orange', label='Validation window log loss')
        ax.set(xlabel='Epoch', ylabel='Log loss')
        ax.axvline(best_epoch, color='gray', linestyle=':', label='Best MLP epoch')
        fig.legend(loc='lower center', bbox_to_anchor=(.5, -.02), ncol=3)
        fig.tight_layout(); fig.savefig(out / 'training_curve.png', dpi=160); plt.close(fig)
    return scaler, candidates, validation, name, best_epoch


def parameter_count(model):
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    return int(sum(a.size for a in model.coefs_ + model.intercepts_))


def export_model(path, model, scaler, name):
    if isinstance(model, LogisticRegression):
        matrices = [model.coef_.T]
        biases = [model.intercept_]
    else:
        matrices = model.coefs_
        biases = model.intercepts_
    payload = {'source': 'UrbanSound8K holdout folds 1-8/9/10', 'architecture': name,
               'classes_in_order': CLASSES, 'class_index_check': model.classes_.tolist(),
               'preprocessing': {'sample_rate_hz': SAMPLE_RATE, 'channel': 'mono_average',
                                 'window_seconds': WINDOW_SECONDS, 'window_hop_seconds': 1,
                                 'fft_size': FFT_SIZE, 'frame_hop': FRAME_HOP, 'mel_bands': MEL_BANDS,
                                 'mel_low_hz': 80, 'mel_high_hz': 7600, 'htk_mel': True,
                                 'periodic_hann': True, 'power_spectrum': True,
                                 'log_energy_floor': 1e-6, 'time_bins': TIME_BINS,
                                 'flatten_order': 'frequency_major'},
               'standard_scaler_mean': scaler.mean_.tolist(), 'standard_scaler_scale': scaler.scale_.tolist(),
               'weights_input_by_output': [a.tolist() for a in matrices],
               'biases': [a.tolist() for a in biases],
               'float32_parameter_bytes': 4 * parameter_count(model),
               'note': 'Model is computer-tested only; board inference, feature parity, latency and power unverified.'}
    with path.open('w', encoding='utf8') as f: json.dump(payload, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--cv', action='store_true', help='Official ten-fold evaluation (extra training time)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_metadata(args.data)
    counts = Counter(r['class'] for r in rows)
    csv_write(args.out / 'dataset_composition.csv',
              [{'class': c, 'role': 'alert' if c in TARGETS else 'non_target', 'original_clips': counts[c]}
               for c in sorted(EXPECTED)])
    X, y, ids = extract(rows, args.out)
    train, val, test = partition(rows, 10, 9)
    csv_write(args.out / 'split_counts.csv', [{'split': n, 'folds': folds, 'class': c,
                                                'original_clips': sum(rows[i]['_label'] == j for i in np.flatnonzero(mask)),
                                                'derived_windows': sum(y[mask[ids]] == j)}
                                               for n, folds, mask in [('train', '1-8', train), ('validation', '9', val), ('test', '10', test)]
                                               for j, c in enumerate(CLASSES)])
    scaler, candidates, validation, selected, epoch = fit_models(X, y, ids, train, val, rows, args.out)
    test_results = {}
    for name, (model, _) in candidates.items():
        ty, tp, details = clip_predictions(model, scaler, X, y, ids, test, rows)
        test_results[name] = metrics(ty, tp)
        if name == selected: csv_write(args.out / 'holdout_predictions.csv', details)
    export_model(args.out / 'model.json', candidates[selected][0], scaler, selected)
    result = {'status': 'trained_on_real_UrbanSound8K', 'protocol': 'train folds 1-8; validate fold 9; test fold 10',
              'classes': CLASSES, 'model_selected_by_validation': selected, 'mlp_best_epoch': epoch,
              'models': {name: {'parameters': parameter_count(model),
                                'float32_parameter_bytes': 4 * parameter_count(model),
                                'validation': validation[name], 'reserved_test': test_results[name]}
                         for name, (model, _) in candidates.items()}}
    if args.cv:
        fold_rows = []
        for k in range(1, 11):
            tr, va, te = partition(rows, k, (k % 10) + 1)
            s, models, v, chosen, _ = fit_models(X, y, ids, tr, va, rows)
            for name, (model, _) in models.items():
                yt, yp, _ = clip_predictions(model, s, X, y, ids, te, rows)
                m = metrics(yt, yp)
                fold_rows.append({'test_fold': k, 'validation_fold': (k % 10) + 1,
                                  'model': name, 'chosen_by_validation': name == chosen,
                                  'n_clips': m['n_clips'], 'accuracy': m['accuracy'],
                                  'alert_precision': m['alert_precision'], 'alert_recall': m['alert_recall'],
                                  'alert_f1': m['alert_f1'], 'siren_recall': m['per_class']['siren']['recall'],
                                  'car_horn_recall': m['per_class']['car_horn']['recall']})
            print(f'Completed official test fold {k}/10', flush=True)
        csv_write(args.out / 'fold_metrics.csv', fold_rows)
        result['official_10fold_summary'] = {name: {'mean_accuracy': float(np.mean([r['accuracy'] for r in fold_rows if r['model'] == name])),
                                                   'std_accuracy': float(np.std([r['accuracy'] for r in fold_rows if r['model'] == name], ddof=1)),
                                                   'mean_alert_f1': float(np.mean([r['alert_f1'] for r in fold_rows if r['model'] == name]))}
                                             for name in candidates}
    with (args.out / 'metrics.json').open('w', encoding='utf8') as f: json.dump(result, f, indent=2)
    print(json.dumps({'selected': selected, 'holdout_alert_f1': test_results[selected]['alert_f1'],
                      'holdout_accuracy': test_results[selected]['accuracy'],
                      'official_cv': result.get('official_10fold_summary')}, indent=2))


if __name__ == '__main__':
    main()
