"""Clustering and segmentation analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config
from .utils import infer_column_types, is_id_like, pick_group_column, pick_time_column, sample_for_modeling


@dataclass
class ClusterResult:
    labels: pd.Series
    profiles: pd.DataFrame
    projection: pd.DataFrame
    meta: dict


def run_clustering(df: pd.DataFrame, dataset: str, targets: list[str], config: Config) -> ClusterResult:
    """Run clustering on non-target features."""
    time_col = pick_time_column(df, config)
    group_col = pick_group_column(df, config)

    numeric_cols, categorical_cols, _, _ = infer_column_types(df, config)
    numeric_cols = [
        col
        for col in numeric_cols
        if col not in targets and col != time_col and col != group_col and not is_id_like(df[col], col, config)
    ]
    categorical_cols = [
        col
        for col in categorical_cols
        if col not in targets and col != time_col and col != group_col and not is_id_like(df[col], col, config)
    ]

    filtered_categorical = []
    for col in categorical_cols:
        if df[col].nunique(dropna=True) <= config.max_cat_levels:
            filtered_categorical.append(col)
    categorical_cols = filtered_categorical

    if not numeric_cols and not categorical_cols:
        empty = pd.DataFrame(columns=["cluster", "feature", "value", "overall", "delta"])
        proj = pd.DataFrame(columns=["pc1", "pc2", "cluster"])
        return ClusterResult(labels=pd.Series(dtype=int), profiles=empty, projection=proj, meta={})

    df_cluster = sample_for_modeling(df, config.sample_max_rows, config.random_seed).reset_index(drop=True)
    X = df_cluster[numeric_cols + categorical_cols].copy()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=config.min_cat_freq,
                    sparse=False,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)
    X_processed = preprocessor.fit_transform(X)

    best_k = 2
    best_score = -1.0
    max_k = min(config.max_k_clusters, len(df_cluster) - 1)
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=config.random_seed, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        score = silhouette_score(X_processed, labels)
        if score > best_score:
            best_score = score
            best_k = k

    final_kmeans = KMeans(n_clusters=best_k, random_state=config.random_seed, n_init=10)
    labels = final_kmeans.fit_predict(X_processed)
    labels_series = pd.Series(labels, name="cluster")

    # Cluster profiles on original scale
    profiles = []
    overall_means = df_cluster[numeric_cols].mean(numeric_only=True)
    for cluster_id in np.unique(labels):
        cluster_mask = labels_series == cluster_id
        cluster_df = df_cluster.loc[cluster_mask]
        for col in numeric_cols:
            mean_val = cluster_df[col].mean()
            overall_val = overall_means.get(col, np.nan)
            profiles.append(
                {
                    "cluster": cluster_id,
                    "feature": col,
                    "value": mean_val,
                    "overall": overall_val,
                    "delta": mean_val - overall_val,
                }
            )
        for col in categorical_cols:
            value_counts = cluster_df[col].value_counts(normalize=True)
            top = value_counts.head(1)
            if not top.empty:
                cat_val = top.index[0]
                profiles.append(
                    {
                        "cluster": cluster_id,
                        "feature": col,
                        "value": f"{cat_val} ({top.iloc[0]:.2%})",
                        "overall": "",
                        "delta": "",
                    }
                )

    profiles_df = pd.DataFrame(profiles)

    # Projection for visualization
    pca = PCA(n_components=2, random_state=config.random_seed)
    coords = pca.fit_transform(X_processed)
    projection = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "cluster": labels,
        }
    )

    meta = {"k": best_k, "silhouette": best_score}
    return ClusterResult(labels=labels_series, profiles=profiles_df, projection=projection, meta=meta)
