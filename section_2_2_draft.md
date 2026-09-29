# 2.2 Audio dataset preparation and model development

**Editorial status:** This is an English draft of an implemented *computer-side experimental protocol*. Replace every bracketed result with the values in `results/metrics.json` after training on the original audio. Do not describe this package's synthetic smoke test as an empirical result. The proposed configurations are implementation choices, not measured performance of SezimBand.

The audio classification stage was defined around two acoustic events relevant to the device's warning function: vehicle horns and sirens. UrbanSound8K provides 8,732 labelled excerpts of at most four seconds across ten urban-sound categories. The experimental output has three classes: `car_horn`, `siren`, and `non_target`. The last category pools all eight other classes, rather than treating a single quiet environment as a complete negative dataset. Class predictions are mapped to the proposed warning decision only for the first two classes. This mapping specifies what the model *would* signal; the actual warning logic and false-alarm behaviour must be tested in the following implementation stage. [UrbanSound8K authors' dataset description](https://urbansounddataset.weebly.com/urbansound8k.html).

**Table 2.2. Dataset composition (populate counts from `dataset_composition.csv` generated from the original metadata; do not count derived overlapping windows as independent recordings).**

| Original label | Model label | Original excerpts |
| --- | --- | ---: |
| `car_horn` | `car_horn` | [from metadata] |
| `siren` | `siren` | [from metadata] |
| `air_conditioner`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `street_music` | `non_target` | [sum of eight rows] |
| **Total** | | **8,732** |

The recordings are decoded as numerical PCM samples, stereo channels are averaged into one mono channel, and a polyphase filter resamples each clip to 16,000 Hz where necessary. Each clip is divided into two-second windows with a one-second overlap; short clips are right-padded with zeros. Window overlap is created **after** assignment to a dataset fold. The spectrogram uses a 400-sample FFT (25 ms), a 160-sample hop (10 ms), a periodic Hann analysis window, and 32 triangular filters spanning 80–7,600 Hz on the HTK mel scale. The squared FFT magnitude is filtered, floored at `1e-6`, and log-transformed. Its 198 frames are averaged into eight sequential temporal intervals for each mel band, giving 256 input values. A standard scaler is fitted only on the training windows. Its mean and scale are exported with the weights; using a different feature implementation on the controller would invalidate direct reuse of those weights.

**Figure 2.3.** Insert `results/feature_example.png`, which plots the waveform and 32-band log-mel representation of one genuine `car_horn` or `siren` sample after dataset processing. The figure is generated automatically by the experiment script. It is not a visualization of captured INMP441 input.

UrbanSound8K assigns related excerpts from the same original Freesound recording to a predefined fold and warns that random reassignment creates leakage and inflated accuracy. For a first reserved test, folds 1–8 train the models, fold 9 selects the model and MLP epoch, and fold 10 is tested once. For a dataset-wide evaluation, the script also runs ten experiments: each fold is the reserved test in turn, the next fold is validation, and the remaining eight folds are training. Fold-wise scores and their average are reported separately from the one-fold holdout result. All windows from a clip remain in its original fold; the metadata field `fsID` is checked for separation. [UrbanSound8K authors' fold guidance](https://urbansounddataset.weebly.com/urbansound8k.html).

**Table 2.3. Allocation in the single-holdout experiment.**

| Purpose | Predefined folds | Original clips | Generated windows |
| --- | --- | ---: | ---: |
| Parameter training | 1–8 | [from `split_counts.csv`] | [from `split_counts.csv`] |
| Architecture and epoch selection | 9 | [from `split_counts.csv`] | [from `split_counts.csv`] |
| Final reserved evaluation | 10 | [from `split_counts.csv`] | [from `split_counts.csv`] |

Two small classifiers share the same input, folds and standardized feature data. The baseline is multinomial logistic regression. The neural alternative is a fully connected network with two ReLU hidden layers of 24 and 12 neurons, followed by a three-way softmax. Both assign a probability to each of the three output classes; overlapping-window probabilities are averaged to obtain a **clip-level** prediction. The logistic baseline balances its classes during fitting. The MLP uses class-balanced sample weights calculated from its training windows. These policies matter because the pooled negative class contains substantially more examples than each alert class. There is no audio augmentation in this version; recording gain, external noise and microphone position remain domain-shift concerns, rather than experimentally established robustness.

**Table 2.4. Architecture and computer-side measurements.**

| Model | Input and architecture | Trainable parameters | FP32 weight bytes | Validation alert F1 | Reserved-test alert F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Logistic regression | 256 → 3 | 771 | 3,084 | [result] | [result] |
| MLP | 256 → 24 → 12 → 3 | 6,507 | 26,028 | [result] | [result] |

*The parameter and byte columns follow from the fixed architecture and exclude input buffering, mel-filter coefficients, scaler arrays, temporary activations and firmware. They are calculated design characteristics, not measured flash usage, RAM allocation, inference duration or energy on the XIAO ESP32-S3.*

The implementation uses Python, SciPy and scikit-learn. Logistic regression minimizes multinomial log loss with L2 regularization (`C=1`, up to 500 optimization iterations). The MLP minimizes weighted log loss with L2 regularization (`alpha=0.0001`), Adam with an initial learning rate of 0.001, a batch size of 128, and 35 successive training epochs. The epoch checkpoint maximizes **validation clip-level alert F1**, then validation accuracy; ties between architectures favour fewer parameters. The reserved test fold is evaluated only after this choice. Report overall clip accuracy, alert precision/recall/F1, per-class precision/recall/F1 and the three-class confusion matrix; the warning system especially needs individual siren and horn recall and false positives from non-target audio.

**Figure 2.4.** Insert `results/training_curve.png` showing weighted training loss and unweighted validation-window log loss by epoch. Selection uses validation **clip-level alert F1**; it does not select by the lowest loss. Both values are saved to `training_curve.csv`.

After the ten-fold evaluation, report `[mean ± standard deviation of fold accuracies]` and the mean alert F1 from `fold_metrics.csv`, plus the chosen holdout model and the reserved fold-10 confusion matrix from `metrics.json`. Avoid presenting the mean as a result achieved by the physical prototype. Recordings made with SezimBand's paired INMP441 microphones should be set aside by recording session and tested separately, including quiet scenes, ordinary street traffic, music, speech, distant horns, region-specific sirens and wind/contact noise. Such a test has not yet been performed. The next stage must reproduce the mono audio and log-mel implementation on the board, quantify memory, runtime and power, and examine warning activation on continuous audio before claiming local inference or operational accuracy.
