#!/usr/bin/env python3
"""Production training pipeline for tabular regression on 53 anonymized features.

This script implements a leakage-free, data-driven modeling pipeline:
1. Validates input schema, numeric finiteness, and target presence.
2. Evaluates candidate model families via deterministic K-Fold Cross-Validation (RMSE):
   - Mean baseline
   - Raw linear OLS (53 features)
   - Additive degree-2 polynomial OLS (107 basis terms: intercept + x_j + x_j^2)
   - Sparse pruned polynomial OLS (automatic inside-fold support selection)
3. Refits the optimal sparse polynomial model on 100% of the training data using
   numerically stable SVD least squares (``scipy.linalg.lstsq``) and exports a
   safe, human-auditable JSON model artifact (avoiding pickle security risks).

Example:
    python3 train.py --train-path train.csv --model-path model.json --folds 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasisTerm:
    """Single additive polynomial basis term ``weight * (x[feature] ** power)``."""

    feature: str
    power: int
    weight: float


@dataclass(frozen=True)
class ModelArtifact:
    """Serializable representation of the trained sparse polynomial model."""

    model_type: str
    created_at_utc: str
    target_col: str
    expected_features: list[str]
    intercept: float
    terms: list[BasisTerm]
    cv_metrics: dict[str, float]
    training_rows: int


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) in float64 precision."""
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def build_additive_poly2_design(
    x_raw: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Construct the additive degree-2 design matrix ``[1, x_0..x_{d-1}, x_0^2..x_{d-1}^2]``.

    Args:
        x_raw: Feature matrix of shape ``(n_samples, n_features)`` in float64.
        feature_names: Ordered list of raw feature column names.

    Returns:
        A tuple ``(design_matrix, term_specs)`` where ``design_matrix`` has shape
        ``(n_samples, 1 + 2 * n_features)`` and ``term_specs`` maps columns ``1..``
        to ``(feature_name, power)``.
    """
    n_samples = x_raw.shape[0]
    ones = np.ones((n_samples, 1), dtype=np.float64)
    design = np.hstack((ones, x_raw, x_raw * x_raw))
    term_specs: list[tuple[str, int]] = [(f, 1) for f in feature_names] + [
        (f, 2) for f in feature_names
    ]
    return design, term_specs


def fit_sparse_poly2(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    effect_threshold: float = 1e-6,
) -> tuple[float, list[BasisTerm]]:
    """Fit an additive degree-2 polynomial model with automatic sparse support pruning.

    Stage 1 fits full standardized OLS via SVD (``gelsd``) across all ``1 + 2*D`` basis
    terms. Stage 2 prunes noise terms whose standardized contribution ``|w_j * std(z_j)|``
    falls below ``effect_threshold`` and refits OLS strictly on the active support.

    Args:
        x_train: Training feature matrix ``(n_samples, n_features)``.
        y_train: Training target vector ``(n_samples,)``.
        feature_names: Raw feature column names.
        effect_threshold: Minimum standardized effect size to retain a basis term.

    Returns:
        A tuple ``(intercept, active_terms)``.
    """
    design, term_specs = build_additive_poly2_design(x_train, feature_names)

    # Standardize non-intercept columns for numerically well-conditioned SVD & fair pruning.
    basis = design[:, 1:]
    means = basis.mean(axis=0)
    stds = basis.std(axis=0)
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    z_norm = (basis - means) / safe_stds

    z_full = np.hstack((np.ones((x_train.shape[0], 1), dtype=np.float64), z_norm))
    coef_norm, _, _, _ = lstsq(z_full, y_train, lapack_driver="gelsd")

    # Select terms with meaningful standardized effect size.
    standardized_effects = np.abs(coef_norm[1:])
    active_indices = np.flatnonzero(standardized_effects >= effect_threshold)

    # Fallback: if all terms were below threshold, retain the full basis.
    if active_indices.size == 0:
        active_indices = np.arange(len(term_specs))

    # Refit unstandardized OLS strictly on the selected active support for exact weights.
    active_design = np.hstack(
        (np.ones((x_train.shape[0], 1), dtype=np.float64), basis[:, active_indices])
    )
    active_coef, _, _, _ = lstsq(active_design, y_train, lapack_driver="gelsd")

    intercept = float(active_coef[0])
    # Zero out numerically negligible intercept noise (< 1e-12) for clean sparsity.
    if abs(intercept) < 1e-12:
        intercept = 0.0

    active_terms = [
        BasisTerm(
            feature=term_specs[int(idx)][0],
            power=int(term_specs[int(idx)][1]),
            weight=float(weight),
        )
        for idx, weight in zip(active_indices, active_coef[1:])
    ]
    return intercept, active_terms


def predict_from_terms(
    df: pd.DataFrame,
    intercept: float,
    terms: list[BasisTerm],
) -> np.ndarray:
    """Evaluate a sparse polynomial model in O(n_samples * len(terms)) time & O(n_samples) RAM."""
    preds = np.full(len(df), intercept, dtype=np.float64)
    for term in terms:
        col_vals = df[term.feature].to_numpy(dtype=np.float64, copy=False)
        if term.power == 1:
            preds += term.weight * col_vals
        elif term.power == 2:
            preds += term.weight * (col_vals * col_vals)
        else:
            preds += term.weight * np.power(col_vals, term.power)
    return preds


def run_kfold_cv(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    *,
    n_splits: int = 5,
    seed: int = 42,
    effect_threshold: float = 1e-6,
) -> dict[str, float]:
    """Run deterministic K-Fold Cross-Validation comparing baseline and polynomial models."""
    x_all = df[feature_names].to_numpy(dtype=np.float64, copy=False)
    y_all = df[target_col].to_numpy(dtype=np.float64, copy=False)
    n_samples = len(y_all)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    folds = np.array_split(indices, n_splits)

    oof_mean = np.empty(n_samples, dtype=np.float64)
    oof_linear = np.empty(n_samples, dtype=np.float64)
    oof_poly2_full = np.empty(n_samples, dtype=np.float64)
    oof_poly2_sparse = np.empty(n_samples, dtype=np.float64)
    active_term_counts: list[int] = []

    for fold_idx in range(n_splits):
        val_idx = folds[fold_idx]
        tr_idx = np.concatenate([folds[j] for j in range(n_splits) if j != fold_idx])

        x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
        x_va = x_all[val_idx]

        # 1. Mean baseline
        oof_mean[val_idx] = float(np.mean(y_tr))

        # 2. Raw Linear OLS
        x_tr_lin = np.hstack((np.ones((len(tr_idx), 1)), x_tr))
        x_va_lin = np.hstack((np.ones((len(val_idx), 1)), x_va))
        w_lin, _, _, _ = lstsq(x_tr_lin, y_tr, lapack_driver="gelsd")
        oof_linear[val_idx] = x_va_lin @ w_lin

        # 3. Full Additive Degree-2 Polynomial OLS (107 terms)
        d_tr, _ = build_additive_poly2_design(x_tr, feature_names)
        d_va, _ = build_additive_poly2_design(x_va, feature_names)
        w_poly, _, _, _ = lstsq(d_tr, y_tr, lapack_driver="gelsd")
        oof_poly2_full[val_idx] = d_va @ w_poly

        # 4. Sparse Pruned Degree-2 Polynomial OLS (support selected strictly inside fold)
        intercept, terms = fit_sparse_poly2(
            x_tr, y_tr, feature_names, effect_threshold=effect_threshold
        )
        active_term_counts.append(len(terms))
        val_df = df.iloc[val_idx]
        oof_poly2_sparse[val_idx] = predict_from_terms(val_df, intercept, terms)

    metrics = {
        "cv_folds": float(n_splits),
        "oof_rmse_mean_baseline": compute_rmse(y_all, oof_mean),
        "oof_rmse_linear_ols": compute_rmse(y_all, oof_linear),
        "oof_rmse_poly2_full": compute_rmse(y_all, oof_poly2_full),
        "oof_rmse_poly2_sparse": compute_rmse(y_all, oof_poly2_sparse),
        "avg_selected_terms": float(np.mean(active_term_counts)),
    }
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train sparse degree-2 polynomial regression model on tabular data."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("train.csv"),
        help="Path to the training CSV file (default: train.csv).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model.json"),
        help="Output path for the serialized JSON model artifact (default: model.json).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Name of the target column in the training CSV (default: target).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic K-Fold splitting (default: 42).",
    )
    parser.add_argument(
        "--effect-threshold",
        type=float,
        default=1e-6,
        help="Standardized effect threshold for sparse basis pruning (default: 1e-6).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute training, cross-validation, and model artifact serialization."""
    t0 = time.perf_counter()
    args = parse_args(argv)

    if not args.train_path.is_file():
        logger.error("Training file not found: %s", args.train_path)
        return 1

    logger.info("Loading training dataset from %s ...", args.train_path)
    df = pd.read_csv(args.train_path)

    if args.target_col not in df.columns:
        logger.error("Target column %r not found in %s.", args.target_col, args.train_path)
        return 1

    feature_names = [c for c in df.columns if c != args.target_col]
    if not feature_names:
        logger.error("No feature columns found in %s.", args.train_path)
        return 1

    if df.isna().any().any():
        logger.error("Training dataset contains missing values; imputation required.")
        return 1

    if not np.isfinite(df.to_numpy(dtype=np.float64)).all():
        logger.error("Training dataset contains non-finite values (Inf/-Inf).")
        return 1

    logger.info(
        "Validated dataset: %d rows x %d features | target=%r",
        len(df),
        len(feature_names),
        args.target_col,
    )

    # 1. Cross-Validation Benchmark
    logger.info("Running %d-Fold Cross-Validation (seed=%d)...", args.folds, args.seed)
    cv_metrics = run_kfold_cv(
        df,
        feature_names,
        args.target_col,
        n_splits=args.folds,
        seed=args.seed,
        effect_threshold=args.effect_threshold,
    )
    logger.info("CV RMSE | Mean Baseline        : %.6f", cv_metrics["oof_rmse_mean_baseline"])
    logger.info("CV RMSE | Raw Linear OLS       : %.6f", cv_metrics["oof_rmse_linear_ols"])
    logger.info("CV RMSE | Full Poly-2 OLS (107): %.6e", cv_metrics["oof_rmse_poly2_full"])
    logger.info(
        "CV RMSE | Sparse Poly-2 OLS    : %.6e (avg %.1f active terms)",
        cv_metrics["oof_rmse_poly2_sparse"],
        cv_metrics["avg_selected_terms"],
    )

    # 2. Final Fit on 100% of Training Data
    x_full = df[feature_names].to_numpy(dtype=np.float64, copy=False)
    y_full = df[args.target_col].to_numpy(dtype=np.float64, copy=False)
    intercept, terms = fit_sparse_poly2(
        x_full, y_full, feature_names, effect_threshold=args.effect_threshold
    )

    train_preds = predict_from_terms(df, intercept, terms)
    train_rmse = compute_rmse(y_full, train_preds)
    cv_metrics["full_train_rmse"] = train_rmse

    logger.info("Final sparse model fitted | intercept=%.6e | active_terms=%d", intercept, len(terms))
    for term in terms:
        logger.info("  Term: (%s)^%d -> weight = %.12f", term.feature, term.power, term.weight)
    logger.info("Full training RMSE: %.6e", train_rmse)

    # 3. Serialize Safe JSON Artifact
    artifact = ModelArtifact(
        model_type="sparse_additive_poly2_ols",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        target_col=args.target_col,
        expected_features=feature_names,
        intercept=intercept,
        terms=terms,
        cv_metrics=cv_metrics,
        training_rows=len(df),
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(artifact)
    args.model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    logger.info("Saved model artifact to %s in %.2f s.", args.model_path, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())