"""Reporting utilities to produce markdown reports and plots."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .modeling import ModelArtifacts


def _format_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "No data available."
    return df.head(max_rows).to_string(index=False)


def _write_markdown(path: Path, sections: Iterable[str]) -> None:
    content = "\n\n".join(sections).strip() + "\n"
    path.write_text(content, encoding="utf-8")


def plot_missingness(missingness: pd.DataFrame, out_path: Path) -> None:
    if missingness.empty:
        return
    missingness = missingness.sort_values("missing_pct", ascending=False).head(30)
    plt.figure(figsize=(10, 6))
    plt.barh(missingness["column"], missingness["missing_pct"])
    plt.gca().invert_yaxis()
    plt.xlabel("Missing percentage")
    plt.title("Top Missingness by Column")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_correlations(corr_pairs: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    if corr_pairs.empty:
        return
    corr_pairs = corr_pairs.copy()
    corr_pairs["abs_pearson"] = corr_pairs["pearson"].abs()
    top = corr_pairs.sort_values("abs_pearson", ascending=False).head(top_n)
    labels = top["feature_1"] + " vs " + top["feature_2"]
    plt.figure(figsize=(10, 6))
    plt.barh(labels, top["abs_pearson"])
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute Pearson correlation")
    plt.title("Top Correlations (Absolute)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cluster_projection(projection: pd.DataFrame, out_path: Path) -> None:
    if projection.empty:
        return
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        projection["pc1"], projection["pc2"], c=projection["cluster"], cmap="tab10", alpha=0.7
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cluster Projection (PCA)")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_interactions(df: pd.DataFrame, interactions: pd.DataFrame, out_dir: Path, top_n: int = 5) -> None:
    if interactions.empty:
        return
    interactions = interactions.sort_values("delta_r2", ascending=False).head(top_n)
    for _, row in interactions.iterrows():
        interaction = row["interaction"]
        if "*" not in interaction:
            continue
        f1, f2 = interaction.split("*", 1)
        if f1 not in df.columns or f2 not in df.columns:
            continue
        plt.figure(figsize=(8, 6))
        plt.scatter(df[f1], df[f2], alpha=0.5, s=10)
        plt.xlabel(f1)
        plt.ylabel(f2)
        plt.title(f"Interaction: {interaction}")
        plt.tight_layout()
        out_path = out_dir / f"interaction_{f1}_{f2}.png"
        plt.savefig(out_path, dpi=200)
        plt.close()


def plot_partial_dependence(
    df: pd.DataFrame,
    artifacts: list[ModelArtifacts],
    feature_importance: pd.DataFrame,
    out_dir: Path,
    top_n: int = 3,
) -> None:
    try:
        from sklearn.inspection import PartialDependenceDisplay
    except Exception:
        return

    for artifact in artifacts:
        target = artifact.target
        importance = feature_importance[
            (feature_importance["target"] == target) & (feature_importance["model"] == "elasticnet")
        ]
        if importance.empty:
            continue
        top_features = (
            importance.sort_values("importance_mean", ascending=False)
            .head(top_n)["feature"]
            .tolist()
        )
        if not top_features:
            continue
        plt.figure(figsize=(10, 4 * len(top_features)))
        try:
            PartialDependenceDisplay.from_estimator(
                artifact.tree_model,
                df,
                features=top_features,
                kind="average",
            )
            plt.suptitle(f"Partial Dependence for {target}")
            plt.tight_layout()
            out_path = out_dir / f"pdp_{target}.png"
            plt.savefig(out_path, dpi=200)
        except Exception:
            pass
        plt.close()


def write_audit_report(
    profile_results: list, join_notes: list[str], out_path: Path
) -> None:
    sections = ["# Data Audit Report"]
    sections.append("## Join Summary\n" + "\n".join(f"- {note}" for note in join_notes))

    inventory_rows = [
        {
            "dataset": p.dataset,
            "rows": p.n_rows,
            "cols": p.n_cols,
            "duplicates": p.duplicate_rows,
        }
        for p in profile_results
    ]
    inventory_df = pd.DataFrame(inventory_rows)
    sections.append("## File Inventory\n```\n" + _format_table(inventory_df) + "\n```")

    for profile in profile_results:
        sections.append(f"## Dataset: {profile.dataset}")
        sections.append(
            "### Missingness\n```\n" + _format_table(profile.missingness, max_rows=15) + "\n```"
        )
        sections.append(
            "### Numeric Distributions\n```\n"
            + _format_table(profile.numeric_stats, max_rows=15)
            + "\n```"
        )
        sections.append(
            "### Categorical Distributions\n```\n"
            + _format_table(profile.categorical_stats, max_rows=15)
            + "\n```"
        )
        sections.append(
            "### Outlier Diagnostics\n```\n"
            + _format_table(profile.outlier_stats, max_rows=15)
            + "\n```"
        )
        sections.append(
            "### Potential Targets\n```\n"
            + _format_table(profile.target_candidates, max_rows=10)
            + "\n```"
        )

    _write_markdown(out_path, sections)


def write_insights_report(
    dataset: str,
    relationships: pd.DataFrame,
    feature_importance: pd.DataFrame,
    interactions: pd.DataFrame,
    cluster_profiles: pd.DataFrame,
    cluster_meta: dict,
    out_path: Path,
) -> None:
    sections = ["# Insights Report"]

    summary_bullets = []
    top_rel = relationships.copy()
    if not top_rel.empty:
        top_rel["abs_score"] = top_rel["score"].abs()
        top_rel = top_rel.sort_values("abs_score", ascending=False).head(5)
        for _, row in top_rel.iterrows():
            summary_bullets.append(
                f"- Strong association: {row['feature']} vs {row['target']} ({row['method']}={row['score']:.3f})."
            )
    if interactions is not None and not interactions.empty:
        top_inter = interactions.sort_values("delta_r2", ascending=False).head(3)
        for _, row in top_inter.iterrows():
            summary_bullets.append(
                f"- Interaction: {row['interaction']} (delta_r2={row['delta_r2']:.3f}, method={row['method']})."
            )
    if cluster_meta:
        summary_bullets.append(
            f"- Segmentation suggests k={cluster_meta.get('k')} with silhouette={cluster_meta.get('silhouette'):.3f}."
        )
    if not summary_bullets:
        summary_bullets.append("- No strong relationships detected; data may be sparse or weakly related.")
    sections.append("## Executive Summary\n" + "\n".join(summary_bullets))

    if not feature_importance.empty:
        top_features = (
            feature_importance.sort_values("importance_mean", ascending=False)
            .head(10)
            .copy()
        )
        sections.append(
            "## Most Influential Factors\n```\n" + _format_table(top_features) + "\n```"
        )

    if interactions is not None and not interactions.empty:
        sections.append(
            "## Interactions That Matter\n```\n"
            + _format_table(interactions.sort_values("delta_r2", ascending=False).head(15))
            + "\n```"
        )

    if not cluster_profiles.empty:
        sections.append("## Segments\n```\n" + _format_table(cluster_profiles.head(20)) + "\n```")
    else:
        sections.append("## Segments\nNo stable clusters identified or insufficient data.")

    sections.append(
        "## Data Limitations and Risk\n"
        "- Observational data; associations are not causal.\n"
        "- Potential confounding, measurement bias, and missingness may influence results.\n"
        "- High-cardinality categorical variables were downsampled or excluded from modeling.\n"
        "- Some analyses rely on sampled data for performance."
    )

    _write_markdown(out_path, sections)
