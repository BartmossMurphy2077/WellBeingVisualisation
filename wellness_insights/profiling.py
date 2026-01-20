"""Profiling and auditing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils import compute_entropy, infer_column_types, is_id_like, pick_time_column


@dataclass
class ProfileResult:
    dataset: str
    n_rows: int
    n_cols: int
    duplicate_rows: int
    missingness: pd.DataFrame
    numeric_stats: pd.DataFrame
    categorical_stats: pd.DataFrame
    outlier_stats: pd.DataFrame
    target_candidates: pd.DataFrame
    data_dictionary: pd.DataFrame


def _infer_description(name: str) -> str:
    tokens = name.split("_")
    if not tokens:
        return name
    return " ".join(tokens)


def infer_target_candidates(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Infer potential wellbeing target columns based on heuristics."""
    candidates = []
    time_col = pick_time_column(df, config)
    for col in df.columns:
        if col == time_col:
            continue
        series = df[col]
        if is_id_like(series, col, config):
            continue
        score = 0
        reasons = []
        lname = col.lower()
        if any(k in lname for k in config.target_keywords):
            score += 3
            reasons.append("keyword_match")
        if any(k in lname for k in config.proxy_target_keywords):
            score += 1
            reasons.append("proxy_keyword")
        if pd.api.types.is_numeric_dtype(series):
            non_na = series.dropna()
            if non_na.nunique() > 1:
                score += 1
                reasons.append("variance")
            if not non_na.empty:
                min_val, max_val = non_na.min(), non_na.max()
                if min_val >= 0 and max_val <= 10:
                    score += 1
                    reasons.append("scale_0_10")
                elif min_val >= 0 and max_val <= 100:
                    score += 1
                    reasons.append("scale_0_100")
        if score > 0:
            candidates.append(
                {
                    "column": col,
                    "score": score,
                    "reasons": ",".join(reasons),
                }
            )
    if not candidates:
        return pd.DataFrame(columns=["column", "score", "reasons"])
    return pd.DataFrame(candidates).sort_values("score", ascending=False)


def build_data_dictionary(df: pd.DataFrame, dataset: str, config: Config) -> pd.DataFrame:
    """Infer a lightweight data dictionary."""
    numeric_cols, categorical_cols, datetime_cols, bool_cols = infer_column_types(df, config)
    rows = []
    for col in df.columns:
        dtype = "other"
        if col in numeric_cols:
            dtype = "numeric"
        elif col in categorical_cols:
            dtype = "categorical"
        elif col in datetime_cols:
            dtype = "datetime"
        elif col in bool_cols:
            dtype = "boolean"
        notes = []
        missing_pct = df[col].isna().mean()
        if missing_pct > 0.1:
            notes.append(f"missing {missing_pct:.1%}")
        if is_id_like(df[col], col, config):
            notes.append("id_like")
        rows.append(
            {
                "dataset": dataset,
                "column": col,
                "inferred_type": dtype,
                "description": _infer_description(col),
                "notes": ";".join(notes),
            }
        )
    return pd.DataFrame(rows)


def profile_dataset(df: pd.DataFrame, dataset: str, config: Config) -> ProfileResult:
    """Generate profiling statistics for a dataset."""
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())

    missingness = pd.DataFrame(
        {
            "dataset": dataset,
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values),
        }
    )

    numeric_cols, categorical_cols, _, _ = infer_column_types(df, config)

    numeric_stats_rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        numeric_stats_rows.append(
            {
                "dataset": dataset,
                "column": col,
                "mean": series.mean(),
                "std": series.std(),
                "median": series.median(),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "skew": series.skew(),
                "kurtosis": series.kurtosis(),
            }
        )
    numeric_stats = pd.DataFrame(numeric_stats_rows)

    categorical_stats_rows = []
    for col in categorical_cols:
        series = df[col]
        non_na = series.dropna()
        if non_na.empty:
            continue
        value_counts = non_na.value_counts()
        top_values = value_counts.head(5).to_dict()
        categorical_stats_rows.append(
            {
                "dataset": dataset,
                "column": col,
                "cardinality": non_na.nunique(),
                "entropy": compute_entropy(non_na),
                "top_values": ";".join(f"{k}:{v}" for k, v in top_values.items()),
            }
        )
    categorical_stats = pd.DataFrame(categorical_stats_rows)

    outlier_rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr
        outlier_pct = ((series < lower) | (series > upper)).mean()
        outlier_rows.append(
            {
                "dataset": dataset,
                "column": col,
                "outlier_pct": outlier_pct,
            }
        )
    outlier_stats = pd.DataFrame(outlier_rows)

    target_candidates = infer_target_candidates(df, config)
    data_dictionary = build_data_dictionary(df, dataset, config)

    return ProfileResult(
        dataset=dataset,
        n_rows=n_rows,
        n_cols=n_cols,
        duplicate_rows=duplicate_rows,
        missingness=missingness,
        numeric_stats=numeric_stats,
        categorical_stats=categorical_stats,
        outlier_stats=outlier_stats,
        target_candidates=target_candidates,
        data_dictionary=data_dictionary,
    )
