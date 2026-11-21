# Pulsar Candidate Classification, Robustness, and Throughput Studies

This project contains two Google Colab notebooks for machine-learning experiments on pulsar-candidate classification. They share the broader goal of identifying pulsars while reducing the number of candidates that require human inspection, but they are **separate experimental workflows** with different evaluation protocols.

## Notebooks

| Notebook | Main purpose | Main methods |
|---|---|---|
| `Germany_bonnipynb(1).ipynb` | Reproduce the published SGAN inference path on sample candidates, inspect the public SGAN data release, and run HTRU1 CNN experiments | Pretrained SGAN feature classifiers, logistic-regression score combination, three-channel CNN, single-/multi-view ablation, threshold-based candidate triage, three-seed repeatability test |
| `HTRU1_Experiment_B_Robustness_Throughput(1).ipynb` | Measure the robustness and inference throughput of image classifiers under controlled image-space shifts | ResNet18, MobileNetV3-Small, YOLO11 classification, and a matched ResNet18/YOLO26 classification scale sweep |

The notebooks are not intended to be run as one continuous notebook. Each contains its own data setup, model workflow, and evaluation choices.

---

## 1. `Germany_bonnipynb(1).ipynb`

### 1.1 SGAN reproduction and sample inference

The notebook clones the [SGAN repository](https://github.com/vishnubk/sgan) and loads the published pretrained models for four pulsar-candidate representations:

- Frequency–phase image
- Time–phase image
- Dispersion-measure (DM) curve
- Pulse profile

The four neural networks produce class predictions. Their outputs are combined by the pretrained logistic-regression model to produce an SGAN score. The notebook checks the expected model input shapes, loads the pretrained weights, and runs inference on five sample `.pfd` / `.ar2` candidates supplied with the repository.

The sample inference cell writes:

```text
/content/sgan/our_sgan_sample_results.csv
```

The recorded sample run reports a mean inference time of approximately 0.675 seconds across five candidates. This is a small smoke test, not a throughput benchmark.

### 1.2 Published SGAN validation check

The notebook also applies the published model to the repository's small validation set. The recorded validation set contains **five candidates** and the displayed accuracy, precision, recall, F1, and ROC-AUC are all 1.0. Because the sample contains only five examples, these values should be treated as a functional check that the inference path runs—not as a reliable estimate of generalisation performance.

### 1.3 SGAN public-data availability check

The notebook examines the public data listed by the SGAN project and attempts to download:

- `downsampled_normalised_data.tar.gz`
- `labelled_candidates_sgan_paper_jan_2021.csv`
- `test_set_relabelled_jan_2021.csv`

The two CSV files download successfully in the recorded run. The archive download returns 417,236,700 bytes (about 397.91 MiB), but the gzip integrity check reports `unexpected end of file`. A second attempt through the original MPIfR FTP endpoint fails the same integrity check. **Do not assume that the full SGAN archive is usable just because the download command completed.** The notebook therefore stops short of treating the full archive as a valid dataset.

The repository inventory finds 12 unique sample/validation candidates with all four feature representations available. This is a small inventory from the checked-out repository, not the full intended SGAN candidate corpus.

### 1.4 HTRU1 dataset and CNN experiments

Because the large SGAN archive could not be verified, the notebook moves to the public [HTRU1 dataset loader](https://github.com/as595/HTRU1). The recorded dataset contains:

| Split | Total examples | Pulsars | Non-pulsars |
|---|---:|---:|---:|
| Training | 50,000 | 995 | 49,005 |
| Test | 10,000 | 199 | 9,801 |

The positive class is pulsar. The strong class imbalance is handled in the CNN experiments using a positive-class weight of approximately 49.251.

The notebook includes these HTRU1 analyses:

1. **Three-channel CNN baseline.** A lightweight CNN is trained on the 32 × 32, three-channel HTRU1 images. In the recorded run, the test-set metrics are precision 0.5433, recall 0.9146, F1 0.6816, ROC-AUC 0.9930, and PR-AUC 0.8331. The recorded inference time for 10,000 test examples is 0.418 seconds on the notebook's GPU runtime.
2. **Single-view versus multi-view ablation.** Separate single-channel models are compared with a model that receives all three channels. The recorded PR-AUC values are 0.7789, 0.7655, and 0.7043 for the three individual channels, compared with 0.8366 for the three-channel model. The multi-channel model also has higher recall in this run, with a corresponding precision trade-off.
3. **Candidate-triage threshold analysis.** Rather than relying only on a default 0.5 threshold, the notebook selects thresholds for target recall levels. For the three-channel model's approximately 95% recall operating point, the recorded values are precision 0.4578, recall 0.9548, false-positive rate 0.0230, and 415 candidates retained out of 10,000 (95.85% candidate reduction).
4. **Three-seed repeatability test.** The three-channel model is trained with seeds 42, 123, and 2026. The recorded mean ± standard deviation are ROC-AUC 0.99333 ± 0.00104 and PR-AUC 0.84015 ± 0.01612. At the selected approximately 95% recall operating point, mean precision is 0.41345 ± 0.03124 and mean candidate reduction is 95.387% ± 0.356 percentage points.

The HTRU1 results and outputs in this notebook are written mainly to the temporary Colab filesystem under `/content/`. Download or copy the result files before the runtime is deleted.

---

## 2. `HTRU1_Experiment_B_Robustness_Throughput(1).ipynb`

This notebook evaluates how image classifiers behave when pulsar-candidate images are subjected to controlled changes in contrast, noise, and stripe-like artifacts. It also benchmarks inference throughput on the current Colab hardware.

### 2.1 Dataset preparation and evaluation split

The notebook downloads the HTRU1 Python-batch archive from [Zenodo](https://zenodo.org/records/3205409) and checks it against the MD5 value embedded in the notebook:

```text
e7b063301ada3eb50f212afeea185a36
```

It identifies the pulsar class as the positive class. The recorded setup has 50,000 training examples (995 pulsars and 49,005 non-pulsars) and confirms the 10,000-example official test batch is readable. **The official test batch is not used for the Experiment B model evaluation.**

A stratified 85/15 split is made from the 50,000-example training pool, using seed 42:

- Training subset: 42,500 examples, including 846 pulsars
- Validation subset: 7,500 examples, including 149 pulsars

The notebook attempts to reuse a previously saved split when available; otherwise, it creates and saves the stratified split. Reported Experiment B performance is therefore validation performance on this split, not official HTRU1 test-set performance.

### 2.2 Controlled image-space conditions

The corruption pipeline evaluates the following eight conditions, including the unmodified baseline:

| Condition | Description |
|---|---|
| `clean` | Original image |
| `contrast_low` | Reduced contrast |
| `contrast_high` | Increased contrast |
| `gaussian_noise_mild` | Mild additive Gaussian noise |
| `gaussian_noise_strong` | Strong additive Gaussian noise |
| `rfi_stripes_mild` | Mild artificial stripe-like interference |
| `rfi_stripes_strong` | Strong artificial stripe-like interference |
| `combined_stress` | Combination of image-space perturbations |

These perturbations are **engineering proxies** for image-domain shifts. They are not calibrated simulations of real telescope radio-frequency interference (RFI), and the notebook does not establish performance under real observing conditions.

### 2.3 Experiment B: ResNet18, MobileNetV3, and YOLO11

The first training/evaluation workflow compares four model configurations:

- `resnet18_clean`
- `resnet18_shiftaug`
- `mobilenet_v3_small_clean`
- `yolo11n_cls_clean`

The backbones are initialized from pretrained weights. The notebook uses class-weighted training, a fixed seed of 42, a maximum of eight epochs, early-stopping patience of three epochs, and a batch size of 128. It evaluates average precision (AP), ROC-AUC, precision/recall at the 0.5 threshold, and precision/recall among the top 100 ranked candidates. Inference is benchmarked using batch sizes 1 and 64 on the available CPU/GPU paths.

The recorded best validation AP during training is 0.9717 for `resnet18_shiftaug`, 0.9704 for `resnet18_clean`, 0.9483 for `mobilenet_v3_small_clean`, and 0.8304 for `yolo11n_cls_clean`. These are validation results from the fixed split and single seed used in this experiment.

### 2.4 YOLO26 final scale sweep

A second preflight/training pair in the same notebook runs a matched comparison between ResNet18 and YOLO26 classification models. It trains 12 configurations:

- ResNet18: clean-trained and shift-augmented
- YOLO26 classification scales `n`, `s`, `m`, `l`, and `x`: clean-trained and shift-augmented variants for each scale

The recorded configuration uses seed 42, batch size 64, up to 20 epochs, early-stopping patience 5, class-weighted cross-entropy, AdamW with learning rate `1e-4` and weight decay `5e-4`, and AP-based checkpoint selection. The same saved training/validation split is used across the matched model runs.

The notebook evaluates all models on the clean validation images and the seven synthetic stress conditions. It also records inference throughput at batch sizes 1 and 64. The communication-plot cell reads the saved robustness and throughput CSV files and creates summary figures; it does not train models.

One useful result from the recorded YOLO26 sweep is the contrast between the two ResNet18 training regimes:

| Model | Clean validation AP | Mean AP over seven stress conditions |
|---|---:|---:|
| `resnet18_clean` | 0.9752 | 0.7794 |
| `resnet18_shiftaug` | 0.9645 | 0.9471 |

In this run, shift augmentation reduces clean AP slightly but substantially improves mean AP under the seven synthetic stress conditions. This result is specific to the fixed validation split and the tested image-space perturbations; it should not be interpreted as proof of robustness to real telescope RFI.

### 2.5 Communication plots

The final plotting cell creates the following figures when the corresponding report CSV files are available:

- `01_AP_robustness_heatmap.png`
- `02_clean_vs_stress_AP.png`
- `03_top100_candidate_review.png` (when top-100 metrics are present)
- `04_stress_AP_vs_throughput.png` (when throughput data are present)

It also writes a derived model summary, copies the source metric tables, and creates `communication_graphs_bundle.zip`. **These plots and the ZIP bundle are written to temporary Colab storage at `/content/yolo26_communication_plots/`, not to Google Drive.** Download the bundle before ending the runtime.

---

## 3. Setup and execution

### Recommended environment

Both notebooks were authored for Google Colab and use notebook-specific absolute paths under `/content/` and `/content/drive/MyDrive/`. A Colab GPU runtime is required for the training workflows. The recorded runs used an NVIDIA Tesla T4 GPU; exact speed measurements will vary with the selected GPU, driver, CUDA, library versions, and runtime load.

The notebooks use a combination of:

- Python 3
- NumPy and pandas
- scikit-learn
- Matplotlib
- PyTorch and TorchVision
- TensorFlow / Keras for the pretrained SGAN models
- Ultralytics for YOLO classification models
- OpenCV (`cv2`) and standard Python archive / file utilities
- Google Colab and Google Drive integration

The second notebook attempts to install Ultralytics if it is not already installed. The notebooks do not provide one unified, pinned `requirements.txt`; package compatibility should be checked in the actual Colab runtime before a full rerun. In particular, the SGAN logistic-regression pickle was saved with scikit-learn 0.22.1, while the recorded runtime used scikit-learn 1.6.1 and displayed an `InconsistentVersionWarning`.

### Run order

**For `Germany_bonnipynb(1).ipynb`:**

1. Open the notebook in Colab and select a GPU runtime.
2. Run the SGAN repository setup from a clean `/content` working directory, then confirm that the `REPO` path in later cells points to the actual repository root. Avoid rerunning the clone/`%cd` cell from inside the repository directory, as this can create a nested `sgan/sgan` path or cause a clone-directory conflict.
3. Run the SGAN pretrained-model load and sample-inference checks.
4. Run the SGAN dataset checks. If the archive fails gzip integrity verification, do not extract or use it as though it were valid; the recorded attempts through both HTTPS and FTP were incomplete.
5. Continue with the HTRU1 setup and CNN experiments. Run the HTRU1 setup/download cell before the cells that load the `HTRU1` Python package and its data.
6. Run the ablation, triage, and repeatability cells after the baseline tensors and labels have been created in the runtime.

**For `HTRU1_Experiment_B_Robustness_Throughput(1).ipynb`:**

1. Open in Colab, select a GPU runtime, and allow Google Drive access.
2. Run the first preflight cell. It downloads/verifies HTRU1, prepares the fixed split, previews perturbations, and smoke-tests model architectures; it does not train models.
3. Run the following training cell in the **same runtime**, because it uses arrays and functions created by the preflight cell.
4. For the YOLO26 sweep, run its second preflight cell, followed by its training/evaluation cell. This pair establishes the YOLO26 project directory and its model set.
5. Run the artifact-check cell to inspect saved checkpoints and report files.
6. Run the final communication-plot cell after the metric CSV files exist. Download its ZIP bundle from the Colab runtime if you need to keep the figures.

If a Colab session disconnects, rerun the relevant preflight cell first. The training code saves checkpoints and progress files to Drive and is designed to resume some interrupted model runs, but the in-memory arrays and functions must be recreated in the new runtime.

---

## 4. Output locations

### Notebook 1: SGAN and HTRU1 CNN experiments

Important files produced in the recorded workflow include:

```text
/content/sgan/our_sgan_sample_results.csv
/content/htru1_multiview_ablation_results.csv
/content/htru1_model_predictions.npz
/content/htru1_candidate_triage_operating_points.csv
/content/htru1_3channel_robustness.csv
/content/htru1_3channel_robustness_summary.csv
/content/htru1_3channel_seed_predictions.npz
```

These are in temporary Colab storage unless copied elsewhere.

### Notebook 2: Experiment B and YOLO26 sweep

The first Experiment B workflow stores persistent files under:

```text
/content/drive/MyDrive/Pulsar_Experiment_B_Robustness/
    raw/
    splits/
    models/
    reports/
    robustness_cache/
    torch_home/
```

The YOLO26 scale sweep stores persistent files under:

```text
/content/drive/MyDrive/Pulsar_Experiment_B_YOLO26_FinalSweep/
    raw/
    splits/
    models/
    reports/
    robustness_cache/
    torch_home/
```

Key YOLO26 report files include:

```text
reports/model_training_summary.csv
reports/robustness_metrics.csv
reports/inference_throughput.csv
reports/yolo26_final_model_summary.csv
reports/YOLO26_FINAL_SWEEP_REPORT.md
reports/yolo26_final_robustness_ap_by_condition.png
reports/yolo26_final_single_image_throughput.png
```

Individual model folders contain checkpoints such as `best.pt`, `last.pt`, training logs, and `completed.json` status files.

---

## 5. Interpretation and reproducibility notes

- **Do not compare the headline scores across the two notebooks as if they came from the same test protocol.** The first notebook evaluates its HTRU1 CNN experiments on the official 10,000-example test batch. Experiment B and the YOLO26 sweep use a validation split drawn from the 50,000-example training set and explicitly do not evaluate on the official test batch.
- **Class imbalance matters.** Pulsars form about 1.99% of the HTRU1 training set in the recorded data. Accuracy alone would hide the cost of false negatives and false positives; AP/PR-AUC, recall, precision, FPR, and candidate-retention metrics are more informative for candidate triage.
- **Synthetic corruption is not real RFI.** Contrast, Gaussian-noise, and stripe perturbations are controlled image-space proxies only. A claim about real telescope robustness requires testing on representative observations and instrument-specific artifacts.
- **Throughput is a partial pipeline measurement.** The Experiment B notebook times in-memory preprocessing, device transfer, and model forward passes. It does not include `.pfd` parsing, candidate generation, file I/O, network transfer, telescope data ingest, or deployment optimisations such as TensorRT.
- **YOLO here means image classification.** The YOLO11/YOLO26 models in these notebooks are classification models (`*-cls`), not YOLO object-detection or oriented-bounding-box models.
- **Seed replication differs by workflow.** The initial Experiment B and YOLO26 sweeps use one fixed seed (42). The earlier HTRU1 three-channel repeatability cell runs seeds 42, 123, and 2026.
- **The five-row SGAN validation set is only a smoke test.** Its perfect metrics should not be presented as a robust validation result.
- **The full SGAN data archive was not validated.** The recorded public archive download failed gzip integrity checks on both attempted transports, even though the download command returned a file with the expected byte count.

## 6. Data and code references

- SGAN implementation: [github.com/vishnubk/sgan](https://github.com/vishnubk/sgan)
- SGAN public data directory: [MPIfR Bonn SGAN data](https://ftp.mpifr-bonn.mpg.de/outgoing/vishnu/sgan_data/)
- HTRU1 Python loader: [github.com/as595/HTRU1](https://github.com/as595/HTRU1)
- HTRU1 batch archive used by Experiment B: [Zenodo record 3205409](https://zenodo.org/records/3205409)

---

## License and data use

The notebooks use external repositories, pretrained checkpoints, and public datasets. Review the license and citation requirements of each source before redistributing its code, weights, or data. The SGAN repository includes its own license; HTRU1 and the Zenodo-hosted data should be cited and used according to their respective source terms.
