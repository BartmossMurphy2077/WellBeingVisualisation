"""Unit tests for utility functions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wellness_insights.config import Config
from wellness_insights.profiling import infer_target_candidates
from wellness_insights.utils import safe_read_csv, standardize_column_name


def test_standardize_column_name() -> None:
    assert standardize_column_name(" Body Mass Index (Men) ") == "body_mass_index_men"
    assert standardize_column_name("123 Score") == "col_123_score"


def test_infer_target_candidates() -> None:
    df = pd.DataFrame(
        {
            "life_satisfaction": [1, 2, 3],
            "age": [20, 30, 40],
            "geo": ["a", "b", "c"],
            "time": [2020, 2021, 2022],
            "suicide_rate": [5, 6, 7],
        }
    )
    config = Config(data_dir=Path("."), out_dir=Path("."))
    candidates = infer_target_candidates(df, config)
    cols = candidates["column"].tolist()
    assert "life_satisfaction" in cols
    assert "suicide_rate" in cols


def test_safe_read_csv(tmp_path: Path) -> None:
    data = "col1;col2\n1;2\n3;4\n"
    path = tmp_path / "sample.csv"
    path.write_text(data, encoding="latin-1")
    df, notes = safe_read_csv(path)
    assert df.shape == (2, 2)
    assert set(df.columns) == {"col1", "col2"}
