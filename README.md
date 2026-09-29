# SezimBand audio experiment

This is an **untrained, reproducible experiment**, not a pretrained detector. Supply the original UrbanSound8K archive unmodified. Never use results from `smoke_test.py` as project accuracy.

## Data

UrbanSound8K: https://urbansounddataset.weebly.com/urbansound8k.html (follow the official download instructions). Extract so that `DATA/metadata/UrbanSound8K.csv` and `DATA/audio/fold1/*.wav` exist. No audio or metadata are included in this package. ESC-50: https://github.com/karolpiczak/ESC-50 is an optional **external** test, not training data, and includes only 40 recordings per class for siren/car horn.

The original archive is several gigabytes; downloading it and running ten training splits can take substantial time. The official [soundata loader](https://soundata.readthedocs.io/en/stable/_modules/soundata/datasets/urbansound8k.html) identifies the archive and published checksum, if the browser form is inconvenient. Do not copy the archive into this project or publish it with model results.

On Windows, from a terminal in this directory:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python experiment.py --data "C:\\path\\to\\UrbanSound8K" --out results --cv
python smoke_test.py
```

On Linux/macOS, use `python3 -m venv .venv`, `source .venv/bin/activate`, then the same Python commands. A faster initial run is `python experiment.py --data DATA --out results`; this trains on folds 1–8, validates on fold 9, tests once on fold 10. The official comparable dataset evaluation is `--cv`: ten tests, one for every predefined test fold, with the following fold used for validation and the other eight for training. Run `--cv` before reporting dataset-wide quality. The CV models are independently selected via their validation folds; the final model exported by the script is the selected model from the *single holdout* split and is not a model trained on all ten folds. The CV and holdout reports are labelled separately.

Outputs: `dataset_composition.csv`, `split_counts.csv`, `metrics.json`, `training_curve.png`, `feature_example.png`, `holdout_predictions.csv`, `fold_metrics.csv` (with `--cv`), and `model.json`. The latter contains the exact log-mel configuration, standardization arrays, model weights and class order, suitable for implementing the same computations in firmware. Its existence does not imply successful device deployment or matched DSP outputs. Check parity with board features and timing/memory in Section 2.3.

An optional independent-corpus test uses ESC-50 without refitting any weights:

```bash
python external_esc50.py --data "C:\\path\\to\\ESC-50-master" --model results/model.json --out results/esc50
```

It assesses all 2,000 ESC-50 clips; its two alert classes contain only 40 clips each, so report their individual recall and the alert confusion matrix. The ESC-50 clips also originate in Freesound and could overlap source recordings with UrbanSound8K; confirm source IDs before calling the external test *source-independent*. They are not hardware recordings.

The intended input is **one mono channel** produced by averaging stereo channels. Dual INMP441 operation must be converted to the same mono input or a new model must be trained. Test the assembled microphones on independently captured siren, horn and non-target recordings; do not mix these into public-dataset test scores. The third class is composed of *all eight* remaining UrbanSound8K classes. This dataset does not supply a representative 'silence' class; validate silence separately and do not claim silence performance from the dataset test.
