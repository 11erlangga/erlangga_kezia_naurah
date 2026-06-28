import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from .train import build_catboost


def apply_feature_drop(
    X,
    cat_features,
    drop_cols=None,
):
    drop_cols = drop_cols or []

    X_new = X.drop(columns=drop_cols, errors="ignore")

    cat_new = [c for c in cat_features if c not in drop_cols]

    return X_new, cat_new


def evaluate_holdout(
    X_train,
    X_val,
    y_train,
    y_val,
    cat_features,
    drop_cols=None,
    params=None,
):
    X_train_new, cat_new = apply_feature_drop(
        X_train,
        cat_features,
        drop_cols,
    )

    X_val_new = X_val.drop(columns=drop_cols or [], errors="ignore")

    model = build_catboost(params)

    model.fit(
        X_train_new,
        y_train,
        cat_features=cat_new,
    )

    pred = model.predict(X_val_new)

    rmse = np.sqrt(mean_squared_error(y_val, pred))

    return rmse


def cross_validate_catboost(
    X,
    y,
    cat_features,
    splitter,
    params=None,
):
    scores = []

    groups = None

    if isinstance(splitter, GroupKFold):
        groups = X["source_id"]

    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model = build_catboost(params)

        model.fit(
            X_train,
            y_train,
            cat_features=cat_features,
        )

        pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, pred))

        scores.append(rmse)

    return np.mean(scores), np.std(scores), scores  # mean_rmse, std_rmse, scores


def evaluate_feature_set(
    X,
    y,
    cat_features,
    splitter,
    drop_cols=None,
    params=None,
):
    X_new, cat_new = apply_feature_drop(
        X,
        cat_features,
        drop_cols,
    )

    return cross_validate_catboost(
        X=X_new,
        y=y,
        cat_features=cat_new,
        splitter=splitter,
        params=params,
    )


def run_experiments(X, y, cat_feats, splitter, experiments):
    """Jalankan evaluate_feature_set untuk beberapa skenario drop_cols.

    Parameters
    ----------
    experiments : dict[str, list[str]]
        Mapping nama eksperimen -> kolom yang di-drop.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping nama eksperimen -> (mean_rmse, std_rmse).
    """
    results = {}
    for name, drop_cols in experiments.items():
        mean_rmse, std_rmse, _ = evaluate_feature_set(
            X, y, cat_feats, splitter, drop_cols=drop_cols
        )
        results[name] = (mean_rmse, std_rmse)
    return results
