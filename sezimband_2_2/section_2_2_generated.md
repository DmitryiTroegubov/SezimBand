# 2.2 Audio dataset preparation and model development

The classifier was developed using UrbanSound8K. The original metadata contained 8732 excerpts; 79 training excerpts whose source recording also appeared in validation were excluded to prevent leakage. The dataset used in this run contained 8653 excerpts from 1297 unique source-recording identifiers (fsID). Table 2.2 distinguishes original source recordings, distributed excerpt files, and model examples. Summed excerpt duration may include overlapping fragments; unique annotated coverage is reported separately. No prototype-microphone recordings were evaluated in this experiment.

[Insert Table 2.2: table_2_2_dataset_composition.csv]

Audio was decoded as mono floating-point samples and resampled to 22050 Hz using kaiser_fast. Each complete excerpt produced one model example; no fixed-duration audio-window segmentation was performed. A 2048-point STFT with a 2048-sample Hann window and a 512-sample hop (23.2200 ms) was used, with centred frames and constant padding. Power spectra were projected onto 128 Slaney-normalised mel bands spanning 0–11025 Hz. Log-power values used reference 1.0, amin 1e-10 and an 80 dB range. A type-II orthonormal DCT produced 40 MFCC coefficients with no liftering. The MFCC time axis was linearly interpolated to 173 columns using OpenCV. Thus the model input was 40 × 173 × 1; the resized columns do not represent a fixed physical frame step across different excerpt durations. No feature standardisation was applied.

[Insert Figure 2.3: figure_2_3_waveform_mfcc.png]

The ten output classes were air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music. An alert label was associated with car_horn or siren under an argmax decision rule. The remaining classes (air_conditioner, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, street_music) were non-alert outputs and were not merged into one training class. These classes do not exhaust possible background sounds encountered by a wrist-worn device.

Training used predefined folds [1, 2, 3, 4, 5, 6, 7, 8]; validation used fold [9]; the final test used fold [10]. An explicit check confirmed that no fsID crossed these split boundaries. Validation determined checkpoint and architecture selection. This experiment is a single predefined-fold holdout, not ten-fold cross-validation.

[Insert Table 2.3: table_2_3_data_allocation.csv]

BaselineCNN reproduced the supplied architecture: three valid 3 × 3 convolutions with 32, 128 and 128 filters, three 2 × 2 max-pooling stages with same padding, two dropout layers (0.3), global average pooling, a 256-unit ReLU layer and a ten-unit softmax output. TinyCNN used valid 3 × 3 convolutions with 8, 16 and 32 filters, two same-padded 2 × 2 max-pooling stages, global average pooling, dropout (0.3) and a ten-unit softmax output. All convolutional activations were ReLU. Both models received the same features and data split. Table 2.4 reports parameter counts, measured file sizes, and analytical Conv2D/Dense MAC estimates per example. These estimates exclude feature extraction and other operations and do not measure device latency or peak RAM.

[Insert Table 2.4: table_2_4_model_comparison.csv]

Training used TensorFlow unknown / Keras, categorical cross-entropy and Adam with learning rate 0.001. Batch size was 8, with a maximum of 15 epochs and early-stopping patience 8, monitoring validation loss. No data augmentation was used. The minimum-validation-loss checkpoint was reloaded for each model. TinyCNN was selected by the lower validation loss (0.839660), with 6,218 parameters. Its best checkpoint occurred at epoch 11 of 15 executed epochs. This selection identifies a candidate for the subsequent embedded feasibility assessment; it does not establish microcontroller compatibility.

[Insert Figure 2.4: figure_2_4_training_validation_loss.png]

After selection, the reserved test fold gave accuracy 0.6249 and macro-F1 0.6354. Class-specific and alert-level results are exported separately for the evaluation section. These are computer-based results on UrbanSound8K and must not be described as achieved by the assembled wristband. Feature settings, software versions, data allocation and model artefacts accompany the export. On-device inference, RAM, latency, microphone transfer and feedback behaviour remain to be tested.
