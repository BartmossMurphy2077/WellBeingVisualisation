"""Data cleaning and normalization routines."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import Config
from .utils import coerce_numeric, maybe_parse_datetime


def _maybe_numeric(series: pd.Series, threshold: float = 0.9) -> Tuple[pd.Series, bool]:
    if series.dtype.kind != "O":
        return series, False
    numeric = coerce_numeric(series)
    ratio = numeric.notna().mean()
    if ratio >= threshold:
        return numeric, True
    return series, False


def clean_dataframe(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Clean a dataframe with light-touch normalization."""
    cleaned = df.copy()
    for col in cleaned.columns:
        if cleaned[col].dtype.kind == "O":
            cleaned[col] = cleaned[col].astype(str).str.strip().replace({"nan": None})
        parsed = maybe_parse_datetime(cleaned[col], name=col)
        if parsed is not None:
            cleaned[col] = parsed
            continue
        coerced, changed = _maybe_numeric(cleaned[col])
        if changed:
            cleaned[col] = coerced
    return cleaned
