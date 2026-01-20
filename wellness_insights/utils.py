"""Utility helpers for wellness insights analysis."""

from __future__ import annotations

import csv
import logging
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Config


def setup_logging(log_path: Path) -> logging.Logger:
    """Configure a logger that writes to both console and file."""
    logger = logging.getLogger("wellness_insights")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")


def standardize_column_name(name: str) -> str:
    """Convert a column name to snake_case ASCII."""
    if name is None:
        name = ""
    name = name.strip().lower()
    name = _NON_ALNUM_RE.sub("_", name)
    name = _MULTI_UNDERSCORE_RE.sub("_", name).strip("_")
    if not name:
        name = "col"
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with standardized column names."""
    return df.rename(columns={col: standardize_column_name(col) for col in df.columns})


def _sniff_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except Exception:
        return ","


def safe_read_csv(path: Path) -> Tuple[pd.DataFrame, list[str]]:
    """Read a CSV with fallback encodings and delimiter sniffing."""
    notes: list[str] = []
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    delimiter = ","
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            sample = handle.read(4096)
            if sample:
                delimiter = _sniff_delimiter(sample)
    except Exception as exc:
        notes.append(f"Delimiter sniff failed: {exc}")

    engines = ["c", "python"] if delimiter == "," else ["python", "c"]
    last_error: Optional[Exception] = None
    for enc in encodings:
        for engine in engines:
            try:
                read_kwargs = {
                    "encoding": enc,
                    "sep": delimiter,
                    "engine": engine,
                }
                if engine == "c":
                    read_kwargs["low_memory"] = False
                df = pd.read_csv(path, **read_kwargs)
                if enc != "utf-8":
                    notes.append(f"Used fallback encoding: {enc}")
                if engine != engines[0]:
                    notes.append(f"Used fallback engine: {engine}")
                return df, notes
            except Exception as exc:
                last_error = exc
                notes.append(f"Read failed with encoding {enc} (engine {engine}): {exc}")

    msg = f"Failed to read {path}. Last error: {last_error}"
    raise RuntimeError(msg)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a series to numeric, preserving NaNs."""
    return pd.to_numeric(series, errors="coerce")


def _looks_like_date(value: str) -> bool:
    value = value.strip()
    if not value:
        return False
    if re.match(r"^\\d{4}([-/]\\d{1,2}([-/]\\d{1,2})?)?$", value):
        return True
    return any(token in value for token in ("-", "/", ":", "T"))


def maybe_parse_datetime(
    series: pd.Series, threshold: float = 0.8, name: Optional[str] = None
) -> Optional[pd.Series]:
    """Attempt to parse datetime for an object series with heuristic checks."""
    if series.dtype.kind != "O":
        return None
    if name:
        lname = name.lower()
        if any(token in lname for token in ("date", "time", "year", "month", "day")):
            candidate = True
        else:
            candidate = False
    else:
        candidate = False

    sample = series.dropna().astype(str).head(50)
    if not candidate:
        candidate = any(_looks_like_date(val) for val in sample)
    if not candidate:
        return None

    year_like = sample.apply(lambda v: bool(re.match(r"^\\d{4}$", v))).mean()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format",
            category=UserWarning,
        )
        if year_like >= 0.9:
            parsed = pd.to_datetime(series, errors="coerce", format="%Y")
        else:
            parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().mean() >= threshold:
        return parsed
    return None


def infer_column_types(
    df: pd.DataFrame, config: Config
) -> Tuple[list[str], list[str], list[str], list[str]]:
    """Infer numeric, categorical, datetime, boolean columns."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # Treat low-cardinality numeric columns as categorical
    categorical_cols: list[str] = []
    for col in numeric_cols[:]:
        nunique = df[col].nunique(dropna=True)
        if nunique <= config.max_cat_levels:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    categorical_cols.extend(object_cols)
    return numeric_cols, categorical_cols, datetime_cols, bool_cols


def is_id_like(series: pd.Series, name: str, config: Config) -> bool:
    """Heuristic to detect id-like columns."""
    lname = name.lower()
    if any(token in lname for token in ("id", "code", "iso")):
        return True
    if lname in config.group_col_candidates:
        return True
    non_na = series.dropna()
    if non_na.empty:
        return False
    ratio = non_na.nunique() / len(non_na)
    return ratio >= config.join_unique_threshold


def compute_entropy(values: pd.Series) -> float:
    """Compute entropy for categorical distributions."""
    counts = values.value_counts(dropna=True)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log2(probs)).sum())


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """Winsorize a numeric series for robust correlation."""
    if series.empty:
        return series
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def pick_time_column(df: pd.DataFrame, config: Config) -> Optional[str]:
    """Pick the most likely time column."""
    for col in df.columns:
        if col.lower() in config.time_col_candidates:
            return col
    for col in df.columns:
        parsed = maybe_parse_datetime(df[col])
        if parsed is not None:
            return col
    return None


def pick_group_column(df: pd.DataFrame, config: Config) -> Optional[str]:
    """Pick a likely grouping column such as geo or id."""
    for col in df.columns:
        if col.lower() in config.group_col_candidates:
            return col
    for col in df.columns:
        if df[col].dtype.kind in ("O", "U", "S"):
            nunique = df[col].nunique(dropna=True)
            if 1 < nunique <= config.max_cat_levels:
                return col
    return None


def time_group_split(
    df: pd.DataFrame,
    time_col: Optional[str],
    group_col: Optional[str],
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create train/test indices respecting time and groups where possible."""
    rng = np.random.default_rng(seed)
    n_rows = len(df)
    indices = np.arange(n_rows)
    if time_col is None:
        rng.shuffle(indices)
        split = int((1 - test_size) * n_rows)
        return indices[:split], indices[split:]

    # Ensure time is sorted
    df_sorted = df.sort_values(time_col)
    if group_col is None:
        split = int((1 - test_size) * n_rows)
        train_idx = df_sorted.index[:split].to_numpy()
        test_idx = df_sorted.index[split:].to_numpy()
        return train_idx, test_idx

    train_indices = []
    test_indices = []
    for _, group in df_sorted.groupby(group_col):
        if len(group) < 2:
            train_indices.extend(group.index.to_numpy())
            continue
        split = max(1, int((1 - test_size) * len(group)))
        train_indices.extend(group.index[:split].to_numpy())
        test_indices.extend(group.index[split:].to_numpy())
    return np.array(train_indices), np.array(test_indices)


def sample_for_modeling(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Sample rows for modeling if the dataset is large."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_feature_name(name: str) -> str:
    """Normalize feature names for reporting."""
    return name.replace("__", ".")


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe divide helper."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def list_csv_files(data_dir: Path) -> list[Path]:
    """Return sorted list of CSV files in the data directory."""
    return sorted(data_dir.glob("*.csv"))


def flatten_list(items: Iterable[Iterable[str]]) -> list[str]:
    """Flatten nested iterables into a list."""
    return [item for sub in items for item in sub]
