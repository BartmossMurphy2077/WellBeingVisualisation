"""I/O utilities for loading and unifying datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import Config
from .utils import list_csv_files, safe_read_csv, standardize_columns


@dataclass
class DataFile:
    """Container for a loaded CSV file."""

    name: str
    path: Path
    df: pd.DataFrame
    notes: list[str]


@dataclass
class JoinResult:
    """Join result metadata."""

    df: Optional[pd.DataFrame]
    keys: list[str]
    confidence: float
    notes: list[str]


def load_csvs(config: Config) -> list[DataFile]:
    """Load and standardize all CSVs from the data directory."""
    datafiles: list[DataFile] = []
    for path in list_csv_files(config.data_dir):
        df, notes = safe_read_csv(path)
        df = standardize_columns(df)
        datafiles.append(DataFile(name=path.stem, path=path, df=df, notes=notes))
    return datafiles


def _key_overlap_score(
    datafiles: list[DataFile], keys: list[str], sample_max: int, seed: int
) -> float:
    rng = np.random.default_rng(seed)
    sets = []
    for datafile in datafiles:
        df = datafile.df[keys].dropna()
        if len(df) > sample_max:
            df = df.sample(n=sample_max, random_state=seed)
        key_tuples = list(map(tuple, df.to_numpy()))
        sets.append(set(key_tuples))
    if not sets:
        return 0.0
    intersection = set.intersection(*sets)
    mean_size = sum(len(s) for s in sets) / len(sets)
    if mean_size == 0:
        return 0.0
    return len(intersection) / mean_size


def _key_uniqueness_ratio(df: pd.DataFrame, keys: list[str]) -> float:
    if df.empty:
        return 0.0
    dupes = df.duplicated(subset=keys).sum()
    return 1 - dupes / len(df)


def _make_unique_columns(
    df: pd.DataFrame, existing: set[str], prefix: str, keys: list[str]
) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in df.columns:
        if col in keys:
            continue
        if col in existing:
            rename[col] = f"{prefix}_{col}"
    return df.rename(columns=rename)


def try_unify(datafiles: list[DataFile], config: Config) -> JoinResult:
    """Attempt to unify datafiles on shared keys."""
    notes: list[str] = []
    if not datafiles:
        return JoinResult(df=None, keys=[], confidence=0.0, notes=["No files loaded."])

    common_cols = set(datafiles[0].df.columns)
    for datafile in datafiles[1:]:
        common_cols &= set(datafile.df.columns)

    if not common_cols:
        return JoinResult(df=None, keys=[], confidence=0.0, notes=["No common columns."])

    candidates: list[list[str]] = []
    if {"geo", "time"} <= common_cols:
        candidates.append(["geo", "time"])
    for col in common_cols:
        if col in ("geo", "time"):
            continue
        if col.lower() in config.time_col_candidates or col.lower() in config.group_col_candidates:
            candidates.append([col])

    if not candidates and len(common_cols) >= 2:
        common_list = sorted(common_cols)
        candidates.append(common_list[:2])

    best_keys: list[str] = []
    best_score = 0.0
    for keys in candidates:
        min_unique = min(_key_uniqueness_ratio(df.df, keys) for df in datafiles)
        overlap = _key_overlap_score(datafiles, keys, sample_max=50000, seed=config.random_seed)
        score = min_unique * overlap
        notes.append(f"Join candidate {keys} unique={min_unique:.3f} overlap={overlap:.3f}")
        if score > best_score:
            best_score = score
            best_keys = keys

    if not best_keys or best_score < config.join_overlap_threshold:
        notes.append("No confident join keys found.")
        return JoinResult(df=None, keys=best_keys, confidence=best_score, notes=notes)

    unified = datafiles[0].df.copy()
    existing_cols = set(unified.columns)
    for datafile in datafiles[1:]:
        df_next = _make_unique_columns(datafile.df, existing_cols, datafile.name, best_keys)
        existing_cols.update(df_next.columns)
        unified = unified.merge(df_next, on=best_keys, how="outer")

    notes.append(f"Unified datasets on keys {best_keys} with score {best_score:.3f}")
    return JoinResult(df=unified, keys=best_keys, confidence=best_score, notes=notes)
