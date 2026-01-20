#!/usr/bin/env python3
"""Run the wellness insights analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from wellness_insights.analysis import analyze_relationships
from wellness_insights.cleaning import clean_dataframe
from wellness_insights.clustering import run_clustering
from wellness_insights.config import Config, default_run_id
from wellness_insights.io import load_csvs, try_unify
from wellness_insights.modeling import run_modeling
from wellness_insights.profiling import profile_dataset
from wellness_insights.reporting import (
    plot_cluster_projection,
    plot_interactions,
    plot_missingness,
    plot_partial_dependence,
    plot_top_correlations,
    write_audit_report,
    write_insights_report,
)
from wellness_insights.utils import ensure_directory, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wellness insights analysis")
    parser.add_argument("--data-dir", required=True, type=Path, help="Path to dataset directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for artifacts")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    config = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        run_id=args.run_id or default_run_id(),
        random_seed=args.seed,
    )

    ensure_directory(config.run_dir)
    figures_dir = config.run_dir / "figures"
    ensure_directory(figures_dir)

    logger = setup_logging(config.run_dir / "run.log")
    logger.info("Starting wellness insights analysis")

    try:
        datafiles = load_csvs(config)
        if not datafiles:
            logger.error("No CSV files found in data directory.")
            return 1
        for datafile in datafiles:
            datafile.df = clean_dataframe(datafile.df, config)

        join_result = try_unify(datafiles, config)
        if join_result.df is not None:
            logger.info("Unified dataset created.")
            unified_df = clean_dataframe(join_result.df, config)
        else:
            logger.info("No unified dataset created.")
            unified_df = None

        profile_results = []
        data_dictionary_frames = []
        for datafile in datafiles:
            profile = profile_dataset(datafile.df, datafile.name, config)
            profile_results.append(profile)
            data_dictionary_frames.append(profile.data_dictionary)

        if unified_df is not None:
            unified_profile = profile_dataset(unified_df, "unified", config)
            profile_results.append(unified_profile)
            data_dictionary_frames.append(unified_profile.data_dictionary)
        else:
            unified_profile = None

        # Relationship analysis
        correlation_frames = []
        relationship_frames = []
        missingness_frames = []
        group_frames = []
        datasets_for_analysis = [("unified", unified_df, unified_profile)] if unified_df is not None else []
        if not datasets_for_analysis:
            datasets_for_analysis = [
                (profile.dataset, datafile.df, profile)
                for datafile, profile in zip(datafiles, profile_results)
            ]

        for dataset_name, dataset_df, profile in datasets_for_analysis:
            targets = profile.target_candidates["column"].tolist() if not profile.target_candidates.empty else []
            relationships = analyze_relationships(dataset_df, dataset_name, targets, config)
            correlation_frames.append(relationships.correlation_pairs)
            relationship_frames.append(relationships.target_relationships)
            missingness_frames.append(relationships.missingness_relationships)
            group_frames.append(relationships.group_comparisons)

        correlation_frames = [df for df in correlation_frames if not df.empty]
        relationship_frames = [df for df in relationship_frames if not df.empty]
        missingness_frames = [df for df in missingness_frames if not df.empty]
        group_frames = [df for df in group_frames if not df.empty]

        correlation_pairs = (
            pd.concat(correlation_frames, ignore_index=True)
            if correlation_frames
            else pd.DataFrame(
                columns=["dataset", "feature_1", "feature_2", "pearson", "spearman", "winsorized_pearson"]
            )
        )
        target_relationships = (
            pd.concat(relationship_frames, ignore_index=True)
            if relationship_frames
            else pd.DataFrame(columns=["dataset", "target", "feature", "method", "score"])
        )
        missingness_relationships = (
            pd.concat(missingness_frames, ignore_index=True)
            if missingness_frames
            else pd.DataFrame(columns=["dataset", "target", "feature", "method", "score"])
        )
        group_comparisons = (
            pd.concat(group_frames, ignore_index=True)
            if group_frames
            else pd.DataFrame(
                columns=["dataset", "target", "feature", "test", "statistic", "p_value", "effect_size", "p_fdr"]
            )
        )

        # Modeling and clustering on unified dataset if available
        model_feature_importance = pd.DataFrame(
            columns=["dataset", "target", "feature", "importance_mean", "importance_std", "model", "metric_mean"]
        )
        interaction_effects = pd.DataFrame(
            columns=["dataset", "target", "interaction", "delta_r2", "direction", "method"]
        )
        cluster_profiles = pd.DataFrame(columns=["cluster", "feature", "value", "overall", "delta"])
        cluster_projection = pd.DataFrame(columns=["pc1", "pc2", "cluster"])
        cluster_meta = {}
        artifacts = []

        if unified_df is not None and unified_profile is not None:
            targets = unified_profile.target_candidates["column"].tolist() if not unified_profile.target_candidates.empty else []
            if targets:
                model_feature_importance, interaction_effects, artifacts = run_modeling(
                    unified_df, "unified", targets, config
                )
            cluster_result = run_clustering(unified_df, "unified", targets, config)
            cluster_profiles = cluster_result.profiles
            cluster_projection = cluster_result.projection
            cluster_meta = cluster_result.meta

        # Write artifacts
        summary_stats = pd.concat(
            [p.numeric_stats for p in profile_results], ignore_index=True
        ) if profile_results else pd.DataFrame()
        data_dictionary = (
            pd.concat(data_dictionary_frames, ignore_index=True)
            if data_dictionary_frames
            else pd.DataFrame(columns=["dataset", "column", "inferred_type", "description", "notes"])
        )

        _write_csv(summary_stats, config.run_dir / "summary_stats.csv")
        _write_csv(correlation_pairs, config.run_dir / "correlation_pairs.csv")
        _write_csv(model_feature_importance, config.run_dir / "feature_importance.csv")
        _write_csv(interaction_effects, config.run_dir / "interaction_effects.csv")
        _write_csv(cluster_profiles, config.run_dir / "cluster_profiles.csv")
        _write_csv(data_dictionary, config.run_dir / "data_dictionary.csv")

        # Plotting
        if profile_results:
            plot_missingness(profile_results[0].missingness, figures_dir / "missingness.png")
        plot_top_correlations(correlation_pairs, figures_dir / "top_correlations.png")
        plot_interactions(unified_df if unified_df is not None else datafiles[0].df, interaction_effects, figures_dir)
        plot_cluster_projection(cluster_projection, figures_dir / "clusters.png")
        if unified_df is not None and artifacts:
            plot_partial_dependence(unified_df, artifacts, model_feature_importance, figures_dir)

        # Reports
        write_audit_report(profile_results, join_result.notes, config.run_dir / "audit_report.md")
        write_insights_report(
            "unified" if unified_df is not None else profile_results[0].dataset,
            target_relationships,
            model_feature_importance,
            interaction_effects,
            cluster_profiles,
            cluster_meta,
            config.run_dir / "insights_report.md",
        )

        # Console findings
        findings = []
        if not target_relationships.empty:
            temp = target_relationships.copy()
            temp["abs_score"] = temp["score"].abs()
            findings.extend(
                temp.sort_values("abs_score", ascending=False)
                .head(5)
                .apply(lambda r: f"{r['feature']} vs {r['target']} ({r['method']}={r['score']:.3f})", axis=1)
                .tolist()
            )
        if not interaction_effects.empty:
            findings.extend(
                interaction_effects.sort_values("delta_r2", ascending=False)
                .head(5)
                .apply(lambda r: f"{r['interaction']} (delta_r2={r['delta_r2']:.3f})", axis=1)
                .tolist()
            )
        logger.info("Top findings:")
        for item in findings[:10]:
            logger.info(" - %s", item)

        logger.info("Run complete. Outputs at %s", config.run_dir)
        print(f"Run folder: {config.run_dir}")
        return 0
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
