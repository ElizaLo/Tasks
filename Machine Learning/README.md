# Tabular Regression on 53 Anonymized Features

A production-ready, leakage-free Python 3 machine learning solution for predicting a continuous `target` from 53 anonymized tabular features, optimized for **RMSE**.

## Key EDA & Modeling Findings

Exploratory analysis of `train.csv` (90,000 rows × 53 features) and `hidden_test.csv` (10,000 rows × 53 features) revealed:

- **Zero Missingness & Stationarity**: Both splits have 0 nulls, 0 duplicates, and pass two-sample Kolmogorov-Smirnov tests across all 53 features after Bonferroni correction ($\alpha_{\text{adj}} = 0.05/53 = 0.000943$; minimum $p = 0.0034$).
- **Hidden Non-Linear Signal**: Every raw feature has near-zero linear correlation with `target` ($\max_j |r(x_j, y)| = 0.0121$) because feature `6` is symmetrically sign-encoded ($\text{corr}(x_6, x_8) = 0.9400$, where `8` is the Bernoulli sign bit).
- **Exact Underlying Manifold**: Expanding to an additive degree-2 basis uncovers $\text{Corr}(x_6^2, y) = 0.99995$ and $\text{Corr}(y - x_6^2, x_7) = 1.00000$, yielding $y = x_6^2 + x_7$ to IEEE-754 `float64` precision. The remaining 50 features (`10` discrete uniform $U\{0,499\}$ + `40` continuous uniform $U(0,15)$) are independent noise.
- **Data-Driven Sparse Model**: Rather than hardcoding coefficients, `train.py` fits a **Sparse Additive Degree-2 Polynomial OLS** via SVD (`scipy.linalg.lstsq`) with automatic inside-fold support pruning, recovering the 2 active basis terms (`(6)^2` and `(7)^1`) across all 5 folds.

## 5-Fold Cross-Validation Results (RMSE)

| Model | Active Parameters | 5-Fold OOF RMSE |
| :--- | :---: | :---: |
| Mean Baseline | 1 | `28.897244` |
| Raw Linear OLS (53 features) | 54 | `28.905370` |
| Full Additive Poly-2 OLS (107 terms) | 107 | `5.9156e-11` |
| **Sparse Pruned Poly-2 OLS (Selected)** | **2** | **`2.2837e-14`** |

## Project Structure

```text
.
├── README.md                   # Project overview, methodology & CLI usage
├── requirements.txt            # Minimal, reproducible Python dependencies
├── eda_and_modeling.ipynb      # Executed EDA, statistical tests, CV & plots
├── train.py                    # CLI training & K-Fold CV script (outputs model.json)
├── predict.py                  # CLI chunked streaming inference script
└── predictions.csv             # Generated predictions for hidden_test.csv (10,000 rows)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place `train.csv` and `hidden_test.csv` in the repository root (or pass explicit paths via CLI flags).

## Terminal Usage

### 1. Train & Cross-Validate (`train.py`)

```bash
python3 train.py \
  --train-path train.csv \
  --model-path model.json \
  --folds 5 \
  --seed 42
```

This runs 5-fold cross-validation, logs RMSE across all candidate models, refits the sparse polynomial model on 100% of `train.csv`, and exports a safe, human-auditable `model.json` artifact (avoiding `pickle` security risks).

### 2. Run Streaming Inference (`predict.py`)

```bash
python3 predict.py \
  --test-path hidden_test.csv \
  --model-path model.json \
  --output-path predictions.csv \
  --chunk-size 50000
```

`predict.py` validates the full 53-column input schema, streams only the active feature columns in fixed-size chunks ($O(\text{chunk\_size})$ peak memory), and writes aligned predictions to `predictions.csv` (`target` column, 10,000 rows, ~0.06s runtime).

### 3. Explore the Notebook

```bash
jupyter notebook eda_and_modeling.ipynb
```
