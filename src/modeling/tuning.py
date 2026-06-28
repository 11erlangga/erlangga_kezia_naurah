import functools

import numpy as np
import optuna
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold

from .train import build_catboost

FIXED_PARAMS = {
    "eval_metric": "RMSE",
}


def objective(trial, X, y, cat_feats, n_splits=5, extra_params=None):
    """Hyperparam space dari https://forecastegy.com/posts/catboost-hyperparameter-tuning-guide-with-optuna/

    Menggunakan StratifiedKFold berdasarkan source_id — setiap fold dijamin
    mengandung distribusi source_id yang proporsional. Konsekuensinya: satu
    source_id bisa muncul di train dan val sekaligus (tidak ada jaminan
    no-leakage antar source).

    Args:
        trial: Optuna trial object.
        X: Feature dataframe (harus mengandung kolom 'source_id').
        y: Target series.
        cat_feats: List nama kolom kategorik.
        n_splits: Jumlah fold. Defaults to 5.
        extra_params: Dict parameter tambahan (misal GPU_PARAMS) yang di-merge
            ke params sebelum training. Defaults to None.

    Raises:
        optuna.TrialPruned: Jika trial di-prune oleh MedianPruner.

    Returns:
        float: Mean RMSE across folds.
    """
    tuned_params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }
    params = {**FIXED_PARAMS, **tuned_params, **(extra_params or {})}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    strata = X["source_id"]
    rmse_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, strata)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_catboost(params)
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feats,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
        )

        rmse = root_mean_squared_error(y_val, model.predict(X_val))
        rmse_scores.append(rmse)

        trial.report(np.mean(rmse_scores), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(rmse_scores)


def tune_catboost(X, y, cat_feats, n_trials=100, n_splits=5, extra_params=None):
    """Cari hyperparameter CatBoost terbaik via Optuna + StratifiedKFold CV.

    Stratifikasi dilakukan berdasarkan kolom 'source_id' di X — distribusi
    source_id dijaga proporsional di setiap fold.

    Args:
        X: Feature dataframe.
        y: Target series.
        cat_feats: List nama kolom kategorik.
        n_trials: Jumlah trial Optuna. Defaults to 100.
        n_splits: Jumlah fold CV. Defaults to 5.
        extra_params: Dict parameter tambahan (misal GPU_PARAMS) yang di-merge
            ke setiap trial. Defaults to None.

    Returns:
        optuna.Study: Study object hasil optimasi.

    Note:
        study.best_params TIDAK menyertakan FIXED_PARAMS (eval_metric) karena
        itu bukan hasil trial.suggest_*. Untuk rekonstruksi model final,
        gabungkan manual: {**FIXED_PARAMS, **study.best_params}.
    """
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(
        functools.partial(
            objective,
            X=X,
            y=y,
            cat_feats=cat_feats,
            n_splits=n_splits,
            extra_params=extra_params,
        ),
        n_trials=n_trials,
    )
    return study
