"""Relationship mining and statistical analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils import infer_column_types, is_id_like, winsorize_series


@dataclass
class RelationshipResults:
    correlation_pairs: pd.DataFrame
    target_relationships: pd.DataFrame
    missingness_relationships: pd.DataFrame
    group_comparisons: pd.DataFrame


def _upper_triangle_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    corr = df.corr(method=method)
    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: method})
    )
    return pairs


def compute_numeric_correlations(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Compute Pearson, Spearman, and winsorized Pearson for numeric columns."""
    df = df.loc[:, df.nunique(dropna=True) > 1]
    if df.shape[1] < 2:
        return pd.DataFrame(
            columns=["dataset", "feature_1", "feature_2", "pearson", "spearman", "winsorized_pearson"]
        )
    pearson = _upper_triangle_corr(df, "pearson")
    spearman = _upper_triangle_corr(df, "spearman")
    winsorized = df.apply(winsorize_series)
    winsorized_corr = _upper_triangle_corr(winsorized, "pearson").rename(columns={"pearson": "winsorized_pearson"})

    merged = pearson.merge(spearman, on=["feature_1", "feature_2"], how="outer").merge(
        winsorized_corr, on=["feature_1", "feature_2"], how="outer"
    )
    merged.insert(0, "dataset", dataset)
    return merged


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Correlation ratio (eta) for categorical-numeric association."""
    categories = categories.astype("category")
    measurements = measurements
    if measurements.empty:
        return float("nan")
    group_means = measurements.groupby(categories, observed=True).mean()
    counts = measurements.groupby(categories, observed=True).count()
    overall_mean = measurements.mean()
    numerator = (counts * (group_means - overall_mean) ** 2).sum()
    denominator = ((measurements - overall_mean) ** 2).sum()
    if denominator == 0:
        return float("nan")
    return float(np.sqrt(numerator / denominator))


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V for categorical-categorical association."""
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return float("nan")

    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.empty:
        return float("nan")
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0:
        return float("nan")
    r, k = confusion_matrix.shape
    return float(np.sqrt(chi2 / (n * (min(k - 1, r - 1)))))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    order = np.argsort(pvalues)
    ranked = np.empty(n)
    prev_bh = 0.0
    for i, idx in enumerate(order, start=1):
        bh_value = pvalues[idx] * n / i
        bh_value = min(bh_value, 1.0)
        prev_bh = max(prev_bh, bh_value)
        ranked[idx] = prev_bh
    return ranked


def _compute_group_comparisons(
    df: pd.DataFrame, target: str, categorical_cols: list[str]
) -> pd.DataFrame:
    rows = []
    try:
        from scipy import stats
    except Exception:
        return pd.DataFrame(columns=["target", "feature", "test", "statistic", "p_value", "effect_size"])

    for col in categorical_cols:
        if col == target:
            continue
        groups = [group.dropna().values for _, group in df.groupby(col)[target]]
        if len(groups) < 2:
            continue
        test_name = "kruskal"
        try:
            stat, p = stats.kruskal(*groups)
        except Exception:
            continue
        # Effect size approximation: epsilon squared
        n = sum(len(g) for g in groups)
        k = len(groups)
        if n <= k:
            effect = float("nan")
        else:
            effect = (stat - k + 1) / (n - k)
        rows.append(
            {
                "target": target,
                "feature": col,
                "test": test_name,
                "statistic": stat,
                "p_value": p,
                "effect_size": effect,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["target", "feature", "test", "statistic", "p_value", "effect_size"])
    result = pd.DataFrame(rows)
    result["p_fdr"] = benjamini_hochberg(result["p_value"].values)
    return result


def compute_target_relationships(
    df: pd.DataFrame, dataset: str, target: str, config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute relationships between predictors and target."""
    if target not in df.columns:
        empty = pd.DataFrame(
            columns=[
                "dataset",
                "target",
                "feature",
                "method",
                "score",
            ]
        )
        return empty, empty, empty

    target_series = df[target]
    numeric_cols, categorical_cols, _, _ = infer_column_types(df, config)
    numeric_cols = [col for col in numeric_cols if col != target and not is_id_like(df[col], col, config)]
    categorical_cols = [
        col for col in categorical_cols if col != target and not is_id_like(df[col], col, config)
    ]

    rows = []
    for col in numeric_cols:
        pair = df[[col, target]].dropna()
        if pair.empty:
            continue
        if pair[col].nunique() <= 1 or pair[target].nunique() <= 1:
            continue
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "feature": col,
                "method": "pearson",
                "score": pair[col].corr(pair[target], method="pearson"),
            }
        )
        rows.append(
            {
                "dataset": dataset,
                "target": target,
                "feature": col,
                "method": "spearman",
                "score": pair[col].corr(pair[target], method="spearman"),
            }
        )

    if pd.api.types.is_numeric_dtype(target_series):
        for col in categorical_cols:
            pair = df[[col, target]].dropna()
            if pair.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "feature": col,
                    "method": "correlation_ratio",
                    "score": correlation_ratio(pair[col], pair[target]),
                }
            )
    else:
        for col in categorical_cols:
            pair = df[[col, target]].dropna()
            if pair.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "feature": col,
                    "method": "cramers_v",
                    "score": cramers_v(pair[col], pair[target]),
                }
            )

    relationships = pd.DataFrame(rows)

    missing_rows = []
    if pd.api.types.is_numeric_dtype(target_series):
        for col in df.columns:
            if col == target:
                continue
            indicator = df[col].isna().astype(int)
            paired = pd.concat([indicator, target_series], axis=1).dropna()
            if paired.empty:
                continue
            if paired.iloc[:, 0].nunique() <= 1 or paired.iloc[:, 1].nunique() <= 1:
                continue
            missing_rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "feature": col,
                    "method": "missingness_corr",
                    "score": paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman"),
                }
            )
    missingness = pd.DataFrame(missing_rows)

    group_comparisons = _compute_group_comparisons(df, target, categorical_cols)
    if not group_comparisons.empty:
        group_comparisons.insert(0, "dataset", dataset)

    return relationships, missingness, group_comparisons


def analyze_relationships(
    df: pd.DataFrame, dataset: str, targets: list[str], config: Config
) -> RelationshipResults:
    """Compute correlation pairs and target-specific relationships."""
    numeric_cols, _, _, _ = infer_column_types(df, config)
    numeric_df = df[numeric_cols].dropna(axis=1, how="all")
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
    correlation_pairs = compute_numeric_correlations(numeric_df, dataset)

    relationship_frames = []
    missing_frames = []
    group_frames = []
    for target in targets:
        relationships, missingness, group_comparisons = compute_target_relationships(df, dataset, target, config)
        if not relationships.empty:
            relationship_frames.append(relationships)
        if not missingness.empty:
            missing_frames.append(missingness)
        if not group_comparisons.empty:
            group_frames.append(group_comparisons)

    return RelationshipResults(
        correlation_pairs=correlation_pairs,
        target_relationships=pd.concat(relationship_frames, ignore_index=True)
        if relationship_frames
        else pd.DataFrame(
            columns=["dataset", "target", "feature", "method", "score"]
        ),
        missingness_relationships=pd.concat(missing_frames, ignore_index=True)
        if missing_frames
        else pd.DataFrame(
            columns=["dataset", "target", "feature", "method", "score"]
        ),
        group_comparisons=pd.concat(group_frames, ignore_index=True)
        if group_frames
        else pd.DataFrame(
            columns=["dataset", "target", "feature", "test", "statistic", "p_value", "effect_size", "p_fdr"]
        ),
    )
