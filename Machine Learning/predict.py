#!/usr/bin/env python3
"""Production streaming inference script for tabular regression predictions.

Loads the JSON model artifact produced by ``train.py`` and generates predictions
for ``hidden_test.csv`` in bounded O(chunk_size) memory by reading only the required
active columns (after verifying full input schema integrity).

Example:
    python3 predict.py --test-path hidden_test.csv --model-path model.json --output-path predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model_artifact(model_path: Path) -> dict[str, Any]:
    """Load and validate a serialized JSON model artifact."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact: dict[str, Any] = json.loads(model_path.read_text(encoding="utf-8"))
    required_keys = {"expected_features", "intercept", "terms", "target_col"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(f"Malformed model artifact; missing keys: {sorted(missing)}")
    return artifact


def predict_chunk(
    chunk: pd.DataFrame,
    intercept: float,
    terms: list[dict[str, Any]],
) -> np.ndarray:
    """Compute predictions for a single DataFrame chunk in float64 precision."""
    preds = np.full(len(chunk), float(intercept), dtype=np.float64)
    for term in terms:
        col_vals = chunk[term["feature"]].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(col_vals).all():
            raise ValueError(
                f"Non-finite value (NaN/Inf) detected in feature {term['feature']!r}."
            )
        power = int(term["power"])
        weight = float(term["weight"])
        if power == 1:
            preds += weight * col_vals
        elif power == 2:
            preds += weight * (col_vals * col_vals)
        else:
            preds += weight * np.power(col_vals, power)
    return preds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the inference script."""
    parser = argparse.ArgumentParser(
        description="Run streaming batch inference using a trained JSON model artifact."
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("hidden_test.csv"),
        help="Path to the input test CSV file (default: hidden_test.csv).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model.json"),
        help="Path to the trained JSON model artifact (default: model.json).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("predictions.csv"),
        help="Destination path for predictions CSV (default: predictions.csv).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Number of rows per streaming chunk for bounded RAM inference (default: 50000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Execute schema validation and chunked streaming inference."""
    t0 = time.perf_counter()
    args = parse_args(argv)

    if not args.test_path.is_file():
        logger.error("Test file not found: %s", args.test_path)
        return 1

    if args.chunk_size <= 0:
        logger.error("chunk-size must be positive, got %d.", args.chunk_size)
        return 1

    artifact = load_model_artifact(args.model_path)
    expected_features: list[str] = artifact["expected_features"]
    intercept: float = float(artifact["intercept"])
    terms: list[dict[str, Any]] = artifact["terms"]
    target_col: str = str(artifact["target_col"])

    # Validate header schema first without loading the entire CSV into RAM.
    header_cols = list(pd.read_csv(args.test_path, nrows=0).columns)
    missing_cols = [c for c in expected_features if c not in header_cols]
    if missing_cols:
        logger.error("Test file is missing required feature columns: %s", missing_cols[:10])
        return 1

    # Read ONLY active model columns during chunked streaming for maximum I/O & RAM efficiency.
    active_cols = sorted({str(t["feature"]) for t in terms})
    logger.info(
        "Streaming inference from %s | active_cols=%s | chunk_size=%d",
        args.test_path,
        active_cols,
        args.chunk_size,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    pred_min = float("inf")
    pred_max = float("-inf")
    pred_sum = 0.0

    write_header = True
    mode = "w"
    for chunk in pd.read_csv(args.test_path, usecols=active_cols, chunksize=args.chunk_size):
        preds = predict_chunk(chunk, intercept, terms)
        out_df = pd.DataFrame({target_col: preds})
        out_df.to_csv(args.output_path, mode=mode, header=write_header, index=False)
        write_header = False
        mode = "a"

        total_rows += len(preds)
        pred_min = min(pred_min, float(preds.min()))
        pred_max = max(pred_max, float(preds.max()))
        pred_sum += float(preds.sum())

    pred_mean = pred_sum / total_rows if total_rows > 0 else float("nan")
    elapsed = time.perf_counter() - t0
    logger.info(
        "Wrote %d predictions to %s in %.3f s | min=%.6f | mean=%.6f | max=%.6f",
        total_rows,
        args.output_path,
        elapsed,
        pred_min,
        pred_mean,
        pred_max,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())