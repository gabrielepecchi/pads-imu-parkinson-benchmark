# Subject-Independent Benchmarking of Wearable IMU Models for Parkinsonian Motor Symptom Classification

[![Tests](https://github.com/gabrielepecchi/pads-imu-parkinson-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/gabrielepecchi/pads-imu-parkinson-benchmark/actions/workflows/tests.yml)

A reproducible benchmarking pipeline for binary Parkinson's Disease (PD) vs. Healthy Control (HC) classification from wrist-worn IMU signals (PADS dataset, PhysioNet), evaluated under a strict **subject-independent cross-validation protocol** that eliminates data leakage.

---

## Key Design Principles

* **Subject-independent evaluation:** subjects are never split across folds — all records for a given subject stay together in a single fold. This is the correct way to estimate generalisation; record-level splits allow the model to see the same person during training and testing, producing inflated metrics.
* **Train-fold-only normalisation:** z-score statistics and all learned transformations are fitted exclusively on training data and applied to test data. Deriving these from the full dataset is a common and often-overlooked source of leakage that this pipeline explicitly prevents.
* **Leakage verified at runtime:** `_validate_folds()` enforces subject separation at every run; a warning is emitted if per-fold class ratios deviate beyond 10%.
* **Honest reporting:** metrics are reported as mean ± std across five folds, without cherry-picking or threshold manipulation.

> **Why subject-independent splitting matters:** In biomedical ML, a model that has seen a patient's other recordings during training will appear far more accurate than it truly is. By keeping each subject entirely out of the test fold, the benchmark measures whether the model generalises to *new patients* — the only evaluation that matters in a real clinical context.

---

## Benchmark Results

All results are from 5-fold subject-independent cross-validation on the full PD + HC subset of PADS. Random seed: 42.

| Model               | Balanced Accuracy      | AUROC                  | Sensitivity (PD)   | Specificity (HC)   |
|---------------------|------------------------|------------------------|--------------------|--------------------|
| Logistic Regression | **0.6441 ± 0.0156**    | 0.7029 ± 0.0261        | 0.6439 ± 0.0590    | 0.6442 ± 0.0329    |
| Random Forest       | 0.5301 ± 0.0120        | **0.7601 ± 0.0275**    | 0.9860 ± 0.0070    | 0.0742 ± 0.0294    |
| 1D CNN v2           | 0.6342 ± 0.0447        | 0.6921 ± 0.0515        | 0.6745 ± 0.0532    | 0.5939 ± 0.1073    |

**Interpreting these numbers honestly:**

Under subject-independent evaluation, modest balanced accuracy is the expected outcome: models are evaluated on held-out subjects they have never seen, which is a harder and more realistic problem than record-level splits.

* **Logistic Regression** achieves the highest Balanced Accuracy and the most balanced sensitivity/specificity trade-off, suggesting that the hand-crafted feature representation captures discriminative structure without collapsing to a majority-class bias.
* **Random Forest** achieves the highest AUROC but near-zero specificity (0.0742) at the default threshold — it predicts PD for almost every sample. This reflects threshold and calibration issues, not strong binary classification performance.
* **1D CNN v2** is close to Logistic Regression in Balanced Accuracy with a more balanced sensitivity/specificity profile. Early stopping in v2 helps prevent over-optimising toward the majority class.

These results reflect what is achievable with fixed hyperparameters and a relatively small HC cohort, under conditions designed to prevent overfitting and leakage. They are conservative by design.

---

## Main Takeaway

This project prioritises **leakage-free evaluation and reproducibility** over inflated performance claims. The reported metrics reflect generalisation to held-out subjects — a more meaningful and more demanding evaluation than what most published baselines report. The value of the project is the pipeline design, not the absolute numbers.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Pipeline Overview](#pipeline-overview)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Cross-Validation Strategy](#cross-validation-strategy)
7. [Implemented Models](#implemented-models)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Installation](#installation)
10. [CUDA / GPU Notes](#cuda--gpu-notes)
11. [Dataset Folder Structure](#dataset-folder-structure)
12. [Usage](#usage)
13. [Results Files](#results-files)
14. [Current Limitations](#current-limitations)
15. [Future Improvements](#future-improvements)
16. [Reproducibility](#reproducibility)
17. [What this project does not prove](#what-this-project-does-not-prove)
18. [License](#license)
19. [Citation](#citation)

---

## Project Overview

This project implements a reproducible end-to-end benchmark for detecting Parkinsonian motor symptoms from wrist IMU signals. Using the publicly available PADS dataset (PhysioNet), three model classes are evaluated under a 5-fold subject-independent cross-validation protocol: Logistic Regression over hand-crafted features, Random Forest over the same feature set, and a 1D Convolutional Neural Network v2 operating directly on raw signal windows.

The design strictly enforces subject independence — all records from a given subject remain in a single fold partition throughout training and evaluation, preventing any form of subject-level data leakage.

---

## Objectives

* Establish a clean, reproducible baseline for PD vs. HC classification on the PADS wrist IMU benchmark.
* Enforce subject-independent evaluation to produce generalisation estimates that are more realistic than record-level splits.
* Compare classical feature-engineering pipelines (Logistic Regression, Random Forest) against an end-to-end deep learning approach (1D CNN v2).
* Report Balanced Accuracy, AUROC, Sensitivity (PD), and Specificity (HC) across five folds, with mean ± standard deviation, to reflect class imbalance and ranking performance simultaneously.

---

## Dataset Description

**Dataset:** [PADS — Parkinson's Disease Smartwatch Dataset](https://physionet.org/content/pads/1.0.0/) (PhysioNet, v1.0.0)

| Property                    | Value                                              |
|-----------------------------|----------------------------------------------------|
| Sensor                      | Apple Watch Series 4 (wrist-worn)                  |
| Sampling Rate               | 100 Hz                                             |
| IMU Channels                | 6 (acc\_x, acc\_y, acc\_z, gyr\_x, gyr\_y, gyr\_z)    |
| Wrist                       | Dominant wrist (per subject handedness metadata)   |
| Task                        | Binary classification — PD vs. HC                  |
| Subjects included           | PD and Healthy Control (HC) only                   |
| Subjects excluded           | Other Movement Disorders (DD) — filtered at load time |
| Assessment steps used       | Steps 1a, 1b, 2, 4, 6, 7, 9, 10, 11               |
| Assessment steps excluded   | Steps 3, 5, 8 (excluded per original PADS paper)   |
| Step duration               | 10.24 s (1024 samples) or 20.48 s (2048 samples)   |
| Label encoding              | PD = 1, HC = 0                                     |

Each subject contributes multiple records (one per assessment step). Subject identity is tracked throughout the pipeline to ensure that no subject appears in both the training and test partition of any fold.

---

## Pipeline Overview

```
Raw PADS files
     │
     ▼
loader.py          — load JSON metadata + timeseries CSVs; filter PD/HC only
     │
     ▼
preprocessor.py    — high-pass filter (acc channels only); zero-pad to global max length
     │
     ▼
cross_val.py       — 5-fold subject-stratified split (no record-level leakage)
     │
     ▼  (per fold)
extractor.py       — hand-crafted features from valid (non-padded) timesteps
     │
     ▼  (per fold)
Normalisation      — z-score fitted on training fold only (run_pipeline.py)
     │
     ├──► logistic_regression.py   — LR with balanced class weights
     ├──► random_forest.py         — RF with balanced class weights
     └──► cnn1d.py                 — 1D CNN v2 on normalised raw signal windows
                │
                ▼
           metrics.py              — Balanced Accuracy, AUROC, Sensitivity (PD), Specificity (HC) per fold
                │
                ▼
          results/                 — per_fold_metrics.csv, metrics_summary.csv
```

All configuration is centralised in `pipeline.yaml`. Running `run_pipeline.py` executes the full pipeline end to end.

---

## Preprocessing Pipeline

Preprocessing is performed once on the full dataset before any fold split, using `preprocessor.py`. The steps below are non-learned signal transformations; all statistical transformations (normalisation) are computed per fold from training data only.

### Steps

1. **Global maximum length computation**
The longest raw signal across all records is identified. This value is used as the universal padding target, ensuring that the padded array dimensions are consistent across all folds.
2. **Zero-phase Butterworth high-pass filter**
A 4th-order Butterworth high-pass filter with a 0.5 Hz cutoff is applied to accelerometer channels (indices 0–2) only using `sosfiltfilt` (zero-phase, no group delay distortion). Gyroscope channels (indices 3–5) are passed through unchanged.

> Rationale: the high-pass filter removes DC offset and slow gravitational drift from accelerometer signals without affecting the movement-frequency content relevant to tremor and bradykinesia detection.

3. **Zero-padding**
Every signal is zero-padded at the trailing end to the global maximum length, producing a uniform `(N, max_len, 6)` array.
4. **Valid length tracking**
The original (pre-padding) length of each signal is recorded as `valid_lengths`. This array is passed to `extractor.py` so that feature computation is restricted to non-padded timesteps only, and to `cnn1d.py` for sequence-aware normalisation.

### Normalisation (per fold, in `run_pipeline.py`)

All learned/statistical transformations, model fitting, and evaluation are performed within fold boundaries.

* **Feature normalisation (LR / RF):** z-score standardisation is fitted exclusively on training-fold feature vectors and applied to both train and test.
* **Sequence normalisation (CNN):** per-channel mean and standard deviation are computed from valid (non-padded) training timesteps only and applied to all timesteps (including padded) at inference.

No normalisation statistics are derived from test data at any point.

---

## Cross-Validation Strategy

Subject-independent 5-fold cross-validation is implemented in `cross_val.py`.

| Property          | Value                                                        |
|-------------------|--------------------------------------------------------------|
| Splitting unit    | Subject (not individual records)                             |
| Number of folds   | 5                                                            |
| Stratification    | PD / HC ratio preserved across folds                         |
| Random seed       | 42 (fixed; stored in `pipeline.yaml`)                        |
| Subject leakage   | Prevented by design — verified by `_validate_folds()`        |

**Guarantees enforced at runtime:**

* No subject appears in both the training and test partition of the same fold.
* Every record appears in exactly one test fold.
* Per-fold PD/HC ratio deviation from the global ratio is logged; a warning is emitted if it exceeds 10% (expected given the relatively small HC cohort).

---

## Implemented Models

### Logistic Regression (`logistic_regression.py`)

| Hyperparameter  | Value                                          |
|-----------------|------------------------------------------------|
| Penalty         | L2                                             |
| C (regularisation) | 1.0                                         |
| Solver          | lbfgs                                          |
| Max iterations  | 1 000                                          |
| Class weights   | Balanced (computed from training labels only)  |

Operates on z-scored hand-crafted features produced by `extractor.py`.

### Random Forest (`random_forest.py`)

| Hyperparameter  | Value                                          |
|-----------------|------------------------------------------------|
| n\_estimators   | 500                                            |
| max\_features   | sqrt                                           |
| Class weights   | Balanced (computed from training labels only)  |
| Random seed     | 42                                             |

Operates on the same normalised feature set as Logistic Regression.

### 1D CNN v2 (`cnn1d.py`)

| Hyperparameter  | Value              |
|-----------------|--------------------|
| Epochs          | 10                 |
| Batch size      | 32                 |
| Learning rate   | 0.001              |
| Dropout         | 0.5                |
| Device          | CUDA (CPU fallback)|

Operates directly on the normalised raw signal windows `(batch, 6, max_len)`. The network is trained independently for each fold; no weights are shared across folds.

CNN v2 introduces an internal stratified validation split derived from the training fold only. Early stopping monitors validation loss with a patience of 8 epochs and restores the best weights before evaluation. The test fold is never used to inform training or early stopping decisions, preserving strict subject-independent evaluation.

---

## Evaluation Metrics

All metrics are computed by `metrics.py` on the held-out test fold only.

| Metric                   | Description                                                                          |
|--------------------------|--------------------------------------------------------------------------------------|
| **Balanced Accuracy**    | Arithmetic mean of sensitivity and specificity. Primary metric; accounts for class imbalance. |
| **AUROC**                | Area under the ROC curve. Threshold-independent ranking metric.                      |
| **Sensitivity (PD)**     | True positive rate for the PD class. Proportion of PD subjects correctly identified. |
| **Specificity (HC)**     | True negative rate for the HC class. Proportion of HC subjects correctly identified. |

Results are reported as **mean ± standard deviation** across the five test folds.

---

## Installation

### Requirements

* Python 3.12
* PyTorch ≥ 2.0 with CUDA support (optional but recommended)
* Standard scientific Python stack

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/gabrielepecchi/pads-imu-parkinson-benchmark.git
cd pads-imu-parkinson-benchmark

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### `requirements.txt` (minimum dependencies)

```
numpy
pandas
scipy
scikit-learn
torch
pyyaml
```

---

## CUDA / GPU Notes

The 1D CNN v2 (`cnn1d.py`) uses PyTorch and will automatically use CUDA if available. The pipeline has been tested on an **NVIDIA RTX 3050 Laptop GPU**.

To install PyTorch with CUDA 12.x support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

To force CPU execution, set the following in `pipeline.yaml`:

```yaml
cnn_device: cpu
```

If no CUDA device is detected at runtime, the pipeline falls back to CPU automatically.

---

## Dataset Folder Structure

Download PADS from [PhysioNet](https://physionet.org/content/pads/1.0.0/) and place it under `data/pads/`. The expected layout is:

```
data/
└── pads/
    ├── patients/
    │   ├── patient_001.json
    │   ├── patient_002.json
    │   └── ...
    └── movement/
        └── timeseries/
            ├── 003_CrossArms_RightWrist.csv
            ├── 003_DrinkGlas_LeftWrist.csv
            ├── 003_Relaxed_RightWrist.csv
            └── ...
```

**Patient JSON fields used:** `id`, `condition`, `handedness`

**Timeseries CSV format:** no header row; column 0 = timestamp (discarded), columns 1–6 = IMU channels (acc\_x, acc\_y, acc\_z, gyr\_x, gyr\_y, gyr\_z) for the dominant wrist.

> On first use, run the inspection utilities to verify condition strings and channel layout before executing the full pipeline:
>
> ```python
> from src.data.loader import inspect_dataset_structure
> inspect_dataset_structure("data/pads")
> ```

---

## Usage

### Run the full benchmark pipeline

```bash
python run_pipeline.py --config configs/pipeline.yaml
```

### Run with a specific model only

Edit `pipeline.yaml` to toggle individual models:

```yaml
run_lr: true
run_rf: false
run_cnn: false
```

Then re-run:

```bash
python run_pipeline.py --config configs/pipeline.yaml
```

### Inspect fold construction before training

```python
from src.data.loader import load_pads
from src.data.preprocessor import preprocess_records
from src.evaluation.cross_val import build_folds, summarise_folds

records = load_pads("data/pads")
dataset = preprocess_records(records)
folds = build_folds(dataset.subject_ids, dataset.labels)
summarise_folds(folds, dataset.labels)
```

---

## Results Files

After a completed pipeline run, two CSV files are written to `results/`:

| File                    | Description                                                                                                        |
|-------------------------|--------------------------------------------------------------------------------------------------------------------|
| `per_fold_metrics.csv`  | One row per (model, fold) combination. Columns: model, fold, balanced\_accuracy, auroc, sensitivity, specificity.  |
| `metrics_summary.csv`   | One row per model. Columns: model, mean and std for each metric across the five folds.                             |

---

## Current Limitations

* **Single wrist only.** The pipeline uses the dominant wrist as recorded in the patient metadata. Bilateral fusion is not implemented.
* **Binary classification only.** Other Movement Disorders (DD subjects) are excluded at the loading stage; multi-class extension is not supported.
* **No hyperparameter optimisation.** All model hyperparameters are fixed at the values specified in `pipeline.yaml`. No nested CV or grid search is performed.
* **Feature set not ablated.** The hand-crafted feature set in `extractor.py` is used as-is; no feature importance analysis or selection is included.
* **CNN architecture not tuned.** The 1D CNN v2 architecture and training schedule are fixed; no architecture search is performed.
* **Small HC cohort.** The HC group in PADS is smaller than the PD group, which limits the statistical power of per-fold estimates and increases fold-to-fold variance, particularly for the CNN.
* **No confidence intervals beyond ± std.** Bootstrapped confidence intervals or permutation tests are not computed.

---

## Future Improvements

* [ ] Bilateral sensor fusion (dominant + non-dominant wrist).
* [ ] Nested cross-validation for principled hyperparameter tuning, including CNN epoch count and learning rate.
* [ ] Transformer-based sequence model (e.g. TST, PatchTST) as an additional baseline.
* [ ] Feature importance analysis for the Random Forest and Logistic Regression pipelines.
* [ ] Threshold optimisation for Random Forest evaluated under nested CV to address the specificity collapse observed at the default threshold.
* [ ] Integration of additional PADS modalities (e.g. video, spiral drawing).
* [ ] Statistical significance testing between model pairs (McNemar's test or permutation test).
* [ ] ONNX / TorchScript export for deployment.

---

## Reproducibility

All sources of randomness are controlled via a single seed value (`random_seed: 42`) stored in `pipeline.yaml` and propagated explicitly to all components:

* `cross_val.py` — `StratifiedKFold(random_state=seed)`
* `logistic_regression.py` — `LogisticRegression(random_state=seed)`
* `random_forest.py` — `RandomForestClassifier(random_state=seed)`
* `cnn1d.py` — `torch.manual_seed(seed)` and `np.random.seed(seed)`

The pipeline is designed to be reproducible under the same dataset, configuration, and environment:

1. Use Python 3.12 with compatible dependency versions (see `requirements.txt` for minimum versions).
2. Use the PADS dataset at PhysioNet version 1.0.0.
3. Run with the default `pipeline.yaml` without modification.
4. GPU non-determinism: CUDA operations may introduce minor floating-point variance across runs on the CNN. Set `cnn_device: cpu` for more deterministic CNN results at the cost of longer runtime.

---

## What this project does not prove

* **Clinical readiness:** the pipeline is a research benchmark, not a validated diagnostic tool.
* **Generalisation to all Parkinson's populations:** results are specific to the PADS dataset and its recruitment protocol; performance on other cohorts is unknown.
* **Diagnostic reliability without larger external validation:** the PADS HC cohort is small; per-fold variance is high and confidence intervals are wide.
* **Superiority over specialised clinical models:** no comparison is made against clinician assessment or domain-specific models trained on larger, multi-site datasets.

---

## License

This project is released under the [MIT License](LICENSE).

The PADS dataset is subject to its own PhysioNet Credentialed Health Data License. Users must independently obtain access at [https://physionet.org/content/pads/1.0.0/](https://physionet.org/content/pads/1.0.0/) and comply with its terms of use.

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{pecchi2026pads,
  author       = {Pecchi, Gabriele},
  title        = {Subject-Independent Benchmarking of Wearable IMU Models
                  for Parkinsonian Motor Symptom Classification},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/gabrielepecchi/pads-imu-parkinson-benchmark}
}
```

For the PADS dataset itself, please cite the original publication:

```bibtex
@dataset{pads2023,
  author    = {Faber, Geraldo S. and others},
  title     = {PADS — Parkinson's Disease Smartwatch Dataset},
  year      = {2023},
  publisher = {PhysioNet},
  url       = {https://physionet.org/content/pads/1.0.0/},
  doi       = {10.13026/8bek-2y80},
}
```
