# Network Intrusion Anomaly Detection with Autoencoders (KDD Cup 1999)

SEP 740 Deep Learning — Final Project.

This repository implements and compares three autoencoder architectures for
unsupervised network-intrusion / anomaly detection on the **KDD Cup 1999**
dataset. All models are trained **only on normal traffic** and use the
**reconstruction error** as an anomaly score: samples whose reconstruction
error exceeds a calibrated threshold are flagged as anomalies.

Three models are provided:

| Model | Framework | Code folder |
|-------|-----------|-------------|
| Basic Autoencoder (baseline) | PyTorch | `src/` |
| Deep Autoencoder | TensorFlow / Keras | `deep_autoencoder/` |
| Sparse Autoencoder (KL sparsity) | PyTorch | `sparse_autoencoder/` |

All three share the same preprocessed data, the same evaluation protocol
(percentile thresholds calibrated on normal traffic; primary threshold = **p95**),
and the same evaluation metrics (precision, recall, F1, confusion matrix, and
per-attack-category recall for DoS / Probe / R2L / U2R).

---

## 1. Repository Structure

```
SEP740-DeepLearning/
├── README.md
├── requirements.txt
├── data_preprocessing.ipynb          # Raw KDD data -> preprocessed .npz (shared by all models)
│
├── dataset/                          # Raw KDD Cup 1999 files (see Section 3)
│   ├── kddcup.data_10_percent_corrected
│   ├── corrected
│   ├── kddcup.names.txt
│   └── training_attack_types.txt
│
├── src/                              # Basic Autoencoder
│   ├── run_basic_autoencoder_pipeline.py
│   ├── select_basic_autoencoder_config.py
│   ├── train_final_basic_autoencoder.py
│   ├── calibrate_basic_threshold.py
│   ├── evaluate_basic_autoencoder.py
│   └── visualize_basic_results.py
│
├── deep_autoencoder/                 # Deep Autoencoder
│   ├── train_final_deep_autoencoder.py
│   ├── calibrate_deep_threshold.py
│   ├── evaluate_deep_autoencoder.py
│   └── visualize_deep_results.py
│
├── sparse_autoencoder/               # Sparse Autoencoder
│   ├── train_sparse_autoencoder.py
│   ├── calibrate_sparse_threshold.py
│   ├── evaluate_sparse_autoencoder.py
│   ├── visualize_sparse_results.py
│   ├── experiment_sparse_beta.py     # beta (sparsity weight) sweep
│   └── experiment_sparse_rho.py      # rho (sparsity target) sweep
│
└── artifacts/                        # Generated data, models, thresholds, metrics, figures
    ├── kdd99_preprocessed_data.npz          # Preprocessed data used by all models
    ├── kdd99_preprocessing_metadata.json
    ├── kdd99_preprocessor.joblib
    ├── models/                              # Trained model weights
    ├── thresholds/                          # Calibrated thresholds per model
    ├── training_history/                    # Per-epoch training logs
    ├── evaluation/                          # Metrics (.json/.csv) + reconstruction errors
    ├── figures/                             # Result figures per model
    ├── hyperparameter_search/               # Basic model config search results
    └── experiments/                         # Sparse beta/rho sweep outputs
```

Each script resolves paths relative to the repository root (via
`Path(__file__).resolve().parents[1]`), so scripts can be launched from inside
their own folder regardless of the current working directory.

---

## 2. Environment Setup

- **Python**: 3.12
- **Key libraries**: PyTorch 2.12.1 (Basic + Sparse), TensorFlow 2.21.0 (Deep),
  NumPy 2.4.6, scikit-learn 1.9.0, pandas 3.0.3, matplotlib 3.11.0, seaborn 0.13.2,
  JupyterLab (for the preprocessing notebook).

```bash
# From the repository root
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

> **Reproducibility**: every training script uses a fixed random seed
> (`SEED = 42`). On CPU the results are effectively identical across reruns.
>
> **Shell note (Windows PowerShell)**: `&&` is not a valid separator. Run
> commands one per line, or join them with `;`.

---

## 3. Download the Dataset

The project uses the **KDD Cup 1999** dataset from the UCI KDD Archive:

- Landing page: <http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html>

Download and decompress the following files, then place them in the `dataset/`
folder using **exactly** these filenames (the preprocessing notebook expects the
original names):

| File in `dataset/` | Source file (decompress if `.gz`) | Role |
|--------------------|-----------------------------------|------|
| `kddcup.data_10_percent_corrected` | `kddcup.data_10_percent.gz` | Training source (10% subset) |
| `corrected` | `corrected.gz` | Official labeled final test set |
| `kddcup.names.txt` | `kddcup.names` | Feature schema / column names |
| `training_attack_types.txt` | `training_attack_types` | Attack → category mapping |

Example (macOS / Linux):

```bash
cd dataset
wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
wget http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz
wget http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
wget http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types
gunzip kddcup.data_10_percent.gz
gunzip corrected.gz
mv kddcup.data_10_percent kddcup.data_10_percent_corrected
mv kddcup.names kddcup.names.txt
mv training_attack_types training_attack_types.txt
```

> The raw dataset files and the generated `artifacts/` are already included in
> this repository, so the results can be replicated **without** re-downloading or
> re-preprocessing. The steps above are provided for completeness / a clean rebuild.

---

## 4. Data Preprocessing (shared by all models)

All models read from a single preprocessed file:
`artifacts/kdd99_preprocessed_data.npz`.

To regenerate it from the raw data, run the notebook `data_preprocessing.ipynb`
top to bottom (JupyterLab or `jupyter nbconvert --to notebook --execute`).

The preprocessing pipeline:

1. Loads `kddcup.data_10_percent_corrected` using the schema in `kddcup.names.txt`.
2. Removes exact duplicate rows (348,435) and a few conflicting rows (4).
3. One-hot encodes the 3 categorical features (`protocol_type`, `service`, `flag`).
4. Applies `log1p` + MinMax scaling to 9 skewed numeric features and MinMax
   scaling to the remaining numeric features (all scaled to `[0, 1]`).
5. Expands the original **41** features to **74** features (`float32`).
6. Fits the preprocessor **only on normal training data** to avoid leakage.

The `.npz` contains the following arrays (produced with `random_state = 42`):

| Array | Shape | Role |
|-------|-------|------|
| `X_train_normal` | (61481, 74) | Autoencoder training (normal only) |
| `X_validation_normal` | (13175, 74) | Early stopping / model selection |
| `X_calibration_normal` | (13175, 74) | Threshold calibration |
| `X_anomaly_reference` | (57751, 74) | Development evaluation (anomalies) |
| `anomaly_categories` | (57751,) | DoS / Probe / R2L / U2R labels |

The official `corrected` set is kept as an untouched final test set and is not
used for the reported development results.

---

## 5. Reproducing the Results

> Prerequisite: `artifacts/kdd99_preprocessed_data.npz` must exist (it is shipped
> with the repo; otherwise run Section 4 first). Each model writes its own
> artifacts with a distinct prefix, so the models never overwrite each other.

### 5.1 Basic Autoencoder (PyTorch)

Architecture `74 → 64 → 32 → 16 → 32 → 64 → 74` (latent 16, ReLU + MSE).
A single pipeline runner executes config selection → training → calibration →
evaluation → figures:

```bash
cd src
python run_basic_autoencoder_pipeline.py
```

Or run the stages individually (same order):

```bash
cd src
python select_basic_autoencoder_config.py
python train_final_basic_autoencoder.py
python calibrate_basic_threshold.py
python evaluate_basic_autoencoder.py
python visualize_basic_results.py
```

Outputs: `artifacts/models/basic_autoencoder.pt`,
`artifacts/thresholds/basic_threshold.json`,
`artifacts/evaluation/basic_metrics.{json,csv}`,
`artifacts/figures/basic_autoencoder/*.png`.

### 5.2 Deep Autoencoder (TensorFlow / Keras)

Deeper architecture `74 → 64 → 48 → 32 → 24 → 16 → 24 → 32 → 48 → 64 → 74`
(latent 16, ReLU + MSE, Sigmoid output). Run the stages in order:

```bash
cd deep_autoencoder
python train_final_deep_autoencoder.py
python calibrate_deep_threshold.py
python evaluate_deep_autoencoder.py
python visualize_deep_results.py
```

Outputs: `artifacts/models/deep_autoencoder.weights.h5` (+ `.json`),
`artifacts/thresholds/deep_threshold.json`,
`artifacts/evaluation/deep_metrics.{json,csv}`,
`artifacts/figures/deep_autoencoder/*.png`.

### 5.3 Sparse Autoencoder (PyTorch)

Same layer sizes as the basic model but with **Sigmoid encoder activations** and
a **KL-divergence sparsity penalty** on the encoder units:

`Loss = MSE(reconstruction) + β · Σ KL(ρ ‖ ρ̂_j)`, with sparsity target
**ρ = 0.1** and sparsity weight **β = 1e-3**.

```bash
cd sparse_autoencoder
python train_sparse_autoencoder.py
python calibrate_sparse_threshold.py
python evaluate_sparse_autoencoder.py
python visualize_sparse_results.py
```

Optional hyperparameter studies (written under `artifacts/experiments/`):

```bash
cd sparse_autoencoder
python experiment_sparse_beta.py   # sweep beta (sparsity weight)
python experiment_sparse_rho.py    # sweep rho (sparsity target)
```

Outputs: `artifacts/models/sparse_autoencoder.pt`,
`artifacts/thresholds/sparse_threshold.json`,
`artifacts/evaluation/sparse_metrics.{json,csv}`,
`artifacts/figures/sparse_autoencoder/*.png`,
`artifacts/experiments/sparse_{beta,rho}_sweep.{csv,json,png}`.

---

## 6. Expected Results

Development-set performance at the primary **p95** threshold (calibrated on
`X_calibration_normal`, evaluated on `X_anomaly_reference`; normal FPR ≈ 5%):

| Metric | Basic AE | Deep AE | Sparse AE |
|--------|----------|---------|-----------|
| Precision | 0.9886 | 0.9885 | 0.9886 |
| Recall | 0.9866 | 0.9767 | 0.9864 |
| F1 | 0.9876 | 0.9825 | 0.9875 |
| False Positive Rate | 0.050 | 0.050 | 0.050 |
| Recall — DoS | 0.988 | 0.985 | 0.987 |
| Recall — Probe | 0.981 | 0.984 | 0.988 |
| Recall — R2L | 0.949 | 0.511 | 0.952 |
| Recall — U2R | 0.942 | 0.942 | 0.981 |

R2L and U2R are the rare, hard attack categories. The exact numbers reported in
the final report are stored in `artifacts/evaluation/*_metrics.json` and the
figures under `artifacts/figures/`.

---

## 7. Repository

<https://github.com/ZeeChee-Guo/SEP740-DeepLearning>
