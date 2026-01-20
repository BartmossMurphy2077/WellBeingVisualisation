"""Configuration for wellness insights analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


def default_run_id() -> str:
    """Return a timestamped run id."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class Config:
    """Central config values for analysis runs."""

    data_dir: Path
    out_dir: Path
    run_id: str = field(default_factory=default_run_id)
    random_seed: int = 42
    sample_max_rows: int = 200_000
    min_rows_for_modeling: int = 200
    max_full_bytes: int = 200 * 1024 * 1024
    join_unique_threshold: float = 0.98
    join_overlap_threshold: float = 0.6
    test_size: float = 0.2
    max_cat_levels: int = 50
    min_cat_freq: int = 20
    top_k_features: int = 10
    top_k_interactions: int = 10
    max_k_clusters: int = 6
    outlier_iqr_multiplier: float = 1.5
    time_col_candidates: tuple[str, ...] = (
        "time",
        "date",
        "year",
        "month",
        "day",
        "timestamp",
    )
    group_col_candidates: tuple[str, ...] = (
        "geo",
        "country",
        "country_code",
        "region",
        "state",
        "id",
        "iso",
    )
    target_keywords: tuple[str, ...] = (
        "wellbeing",
        "well_being",
        "happiness",
        "life_satisfaction",
        "stress",
        "anxiety",
        "mood",
        "sleep_quality",
        "depression",
        "mental_health",
        "health_score",
        "qol",
        "quality_of_life",
        "burnout",
    )
    proxy_target_keywords: tuple[str, ...] = (
        "suicide",
        "mortality",
        "death",
        "deaths",
        "bmi",
        "alcohol",
        "cholesterol",
        "cancer",
        "unemployment",
        "working_hours",
    )

    @property
    def run_dir(self) -> Path:
        """Return the output run directory path."""
        return self.out_dir / f"run_{self.run_id}"
