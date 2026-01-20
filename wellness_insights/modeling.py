"""Modeling and interaction analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config
from .utils import infer_column_types, is_id_like, pick_group_column, pick_time_column, sample_for_modeling


@dataclass
class ModelArtifacts:
    target: str
    model: Pipeline
    tree_model: Optional[Pipeline]
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    model_type: str


def _prepare_features(
    df: pd.DataFrame,
    target: str,
    config: Config,
    time_col: Optional[str],
    group_col: Optional[str],
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = df.dropna(subset=[target]).copy()
    numeric_cols, categorical_cols, _, _ = infer_column_types(df, config)
    numeric_cols = [
        col
        for col in numeric_cols
        if col != target
        and col != time_col
        and col != group_col
        and not is_id_like(df[col], col, config)
    ]
    categorical_cols = [
        col
        for col in categorical_cols
        if col != target
        and col != time_col
        and col != group_col
        and not is_id_like(df[col], col, config)
    ]

    # Remove high-cardinality categorical columns
    filtered_categorical = []
    for col in categorical_cols:
        if df[col].nunique(dropna=True) <= config.max_cat_levels:
            filtered_categorical.append(col)
    categorical_cols = filtered_categorical

    feature_cols = numeric_cols + categorical_cols
    X = df[feature_cols].copy()
    y = df[target]
    return X, y, numeric_cols, categorical_cols


def _build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str], config: Config
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
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

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_model(is_classification: bool, seed: int) -> Pipeline:
    if is_classification:
        model = LogisticRegression(max_iter=2000)
    else:
        model = ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=seed, max_iter=10000)
    return model


def _build_tree_model(is_classification: bool, seed: int) -> Pipeline:
    if is_classification:
        model = RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2
        )
    else:
        model = RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=-1, min_samples_leaf=2
        )
    return model


def _cv_splits(
    df: pd.DataFrame, time_col: Optional[str], group_col: Optional[str], seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = 3
    if time_col and time_col in df.columns:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        return list(splitter.split(df))
    if group_col and group_col in df.columns:
        unique_groups = df[group_col].nunique()
        if unique_groups >= 2:
            splitter = GroupKFold(n_splits=min(n_splits, unique_groups))
            return list(splitter.split(df, groups=df[group_col]))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df))


def _is_classification_target(y: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 10 and np.allclose(y.dropna() % 1, 0):
            return True
        return False
    return True


def _metric(is_classification: bool, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> float:
    if is_classification:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            return roc_auc_score(y_true, y_proba)
        return accuracy_score(y_true, y_pred)
    return r2_score(y_true, y_pred)


def _collect_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
    return names


def _map_feature_to_original(
    encoded_names: list[str], numeric_cols: list[str], categorical_cols: list[str]
) -> list[str]:
    mapped = []
    cat_prefixes = [f"{col}_" for col in categorical_cols]
    for name in encoded_names:
        if "__" in name:
            name = name.split("__", 1)[1]
        original = name
        for col in numeric_cols:
            if name == col:
                original = col
                break
        for col in categorical_cols:
            if name.startswith(f"{col}_"):
                original = col
                break
        mapped.append(original)
    return mapped


def _aggregate_importance(
    names: list[str], original_names: list[str], importances: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame({"feature": original_names, "importance": importances})
    agg = df.groupby("feature", as_index=False)["importance"].sum()
    return agg


def run_modeling(
    df: pd.DataFrame, dataset: str, targets: list[str], config: Config
) -> tuple[pd.DataFrame, pd.DataFrame, list[ModelArtifacts]]:
    """Fit models per target and compute feature importance and interactions."""
    feature_importance_rows = []
    interaction_rows = []
    artifacts: list[ModelArtifacts] = []

    time_col = pick_time_column(df, config)
    group_col = pick_group_column(df, config)

    for target in targets:
        if target not in df.columns:
            continue
        df_target = df.dropna(subset=[target])
        if len(df_target) < config.min_rows_for_modeling:
            continue

        df_target = sample_for_modeling(df_target, config.sample_max_rows, config.random_seed)
        if time_col and time_col in df_target.columns:
            df_target = df_target.sort_values(time_col).reset_index(drop=True)
        else:
            df_target = df_target.reset_index(drop=True)
        X, y, numeric_cols, categorical_cols = _prepare_features(
            df_target, target, config, time_col, group_col
        )
        if X.empty:
            continue
        is_classification = _is_classification_target(y)

        preprocessor = _build_preprocessor(numeric_cols, categorical_cols, config)
        tree_preprocessor = _build_preprocessor(numeric_cols, categorical_cols, config)
        base_model = _build_model(is_classification, config.random_seed)
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])

        tree_model = _build_tree_model(is_classification, config.random_seed)
        tree_pipeline = Pipeline(steps=[("preprocess", tree_preprocessor), ("model", tree_model)])

        splits = _cv_splits(df_target, time_col, group_col, config.random_seed)
        if not splits:
            continue

        fold_importances = []
        fold_metrics = []
        for train_idx, test_idx in splits:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = None
            if is_classification:
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                except Exception:
                    y_proba = None
            fold_metrics.append(_metric(is_classification, y_test, y_pred, y_proba))

            result = permutation_importance(
                pipeline, X_test, y_test, n_repeats=5, random_state=config.random_seed
            )
            feature_names = _collect_feature_names(preprocessor)
            if not feature_names:
                continue
            original_names = _map_feature_to_original(feature_names, numeric_cols, categorical_cols)
            aggregated = _aggregate_importance(feature_names, original_names, result.importances_mean)
            fold_importances.append(aggregated.set_index("feature")["importance"])

        if fold_importances:
            importance_df = pd.concat(fold_importances, axis=1).fillna(0.0)
            importance_mean = importance_df.mean(axis=1)
            importance_std = importance_df.std(axis=1)
            for feature, mean_val in importance_mean.items():
                feature_importance_rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "feature": feature,
                        "importance_mean": mean_val,
                        "importance_std": float(importance_std.loc[feature]),
                        "model": "elasticnet" if not is_classification else "logistic",
                        "metric_mean": float(np.mean(fold_metrics)) if fold_metrics else float("nan"),
                    }
                )

        # Fit final models for artifacts and tree-based importance
        pipeline.fit(X, y)
        tree_pipeline.fit(X, y)

        feature_names = _collect_feature_names(preprocessor)
        artifacts.append(
            ModelArtifacts(
                target=target,
                model=pipeline,
                tree_model=tree_pipeline,
                feature_names=feature_names,
                numeric_features=numeric_cols,
                categorical_features=categorical_cols,
                model_type="classification" if is_classification else "regression",
            )
        )

        # Tree-based permutation importance for interaction proxy
        tree_feature_names = _collect_feature_names(tree_preprocessor)
        if tree_feature_names:
            try:
                tree_result = permutation_importance(
                    tree_pipeline, X, y, n_repeats=5, random_state=config.random_seed
                )
                original_names = _map_feature_to_original(
                    tree_feature_names, numeric_cols, categorical_cols
                )
                tree_importance = _aggregate_importance(
                    tree_feature_names, original_names, tree_result.importances_mean
                )
                for _, row in tree_importance.iterrows():
                    feature_importance_rows.append(
                        {
                            "dataset": dataset,
                            "target": target,
                            "feature": row["feature"],
                            "importance_mean": row["importance"],
                            "importance_std": float("nan"),
                            "model": "random_forest",
                            "metric_mean": float("nan"),
                        }
                    )
            except Exception:
                pass

        # Interaction effects for numeric features
        if len(numeric_cols) >= 2 and pd.api.types.is_numeric_dtype(df_target[target]):
            numeric_corr = (
                df_target[numeric_cols + [target]]
                .corr()[target]
                .drop(target)
                .abs()
                .sort_values(ascending=False)
            )
            top_numeric = numeric_corr.head(config.top_k_features).index.tolist()
            pairs = []
            for i, f1 in enumerate(top_numeric):
                for f2 in top_numeric[i + 1 :]:
                    pairs.append((f1, f2))
            for f1, f2 in pairs[: config.top_k_interactions]:
                sub = df_target[[f1, f2, target]].dropna()
                if sub.empty:
                    continue
                x1 = sub[f1].values
                x2 = sub[f2].values
                y_sub = sub[target].values
                # Standardize for stability
                x1 = (x1 - x1.mean()) / (x1.std() or 1)
                x2 = (x2 - x2.mean()) / (x2.std() or 1)
                base = np.column_stack([x1, x2])
                interaction = np.column_stack([x1, x2, x1 * x2])
                scores = []
                for train_idx, test_idx in splits:
                    base_train, base_test = base[train_idx], base[test_idx]
                    int_train, int_test = interaction[train_idx], interaction[test_idx]
                    y_train, y_test = y_sub[train_idx], y_sub[test_idx]
                    from sklearn.linear_model import LinearRegression

                    base_model = LinearRegression().fit(base_train, y_train)
                    int_model = LinearRegression().fit(int_train, y_train)
                    base_pred = base_model.predict(base_test)
                    int_pred = int_model.predict(int_test)
                    base_r2 = r2_score(y_test, base_pred)
                    int_r2 = r2_score(y_test, int_pred)
                    scores.append(int_r2 - base_r2)
                coef_sign = float(np.sign(np.cov(x1 * x2, y_sub)[0, 1])) if len(y_sub) > 1 else 0.0
                interaction_rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "interaction": f"{f1}*{f2}",
                        "delta_r2": float(np.mean(scores)) if scores else float("nan"),
                        "direction": "positive" if coef_sign >= 0 else "negative",
                        "method": "linear_interaction",
                    }
                )

        # Tree-based interaction proxy
        if len(numeric_cols) >= 2 and pd.api.types.is_numeric_dtype(df_target[target]):
            top_features = numeric_cols[: config.top_k_features]
            sub = df_target[top_features + [target]].dropna()
            if len(sub) >= config.min_rows_for_modeling:
                X_sub = sub[top_features].copy()
                for col in X_sub.columns:
                    X_sub[col] = X_sub[col].fillna(X_sub[col].median())
                y_sub = sub[target].values
                rf = RandomForestRegressor(
                    n_estimators=200, random_state=config.random_seed, n_jobs=-1, min_samples_leaf=2
                )
                rf.fit(X_sub, y_sub)
                try:
                    baseline = rf.predict(X_sub)
                    baseline_metric = r2_score(y_sub, baseline)
                except Exception:
                    baseline_metric = float("nan")
                for i, f1 in enumerate(top_features):
                    for f2 in top_features[i + 1 : config.top_k_interactions]:
                        X_perm = X_sub.copy()
                        rng = np.random.default_rng(config.random_seed)
                        X_perm[f1] = rng.permutation(X_perm[f1].values)
                        X_perm[f2] = rng.permutation(X_perm[f2].values)
                        try:
                            pred_both = rf.predict(X_perm)
                            metric_both = r2_score(y_sub, pred_both)
                        except Exception:
                            metric_both = float("nan")
                        interaction_rows.append(
                            {
                                "dataset": dataset,
                                "target": target,
                                "interaction": f"{f1}*{f2}",
                                "delta_r2": baseline_metric - metric_both
                                if baseline_metric == baseline_metric
                                else float("nan"),
                                "direction": "unknown",
                                "method": "rf_joint_permutation",
                            }
                        )

    feature_importance = pd.DataFrame(feature_importance_rows)
    interaction_effects = pd.DataFrame(interaction_rows)
    return feature_importance, interaction_effects, artifacts
