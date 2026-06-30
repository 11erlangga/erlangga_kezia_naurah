import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from .train import build_catboost


def apply_feature_drop(
    X,
    cat_features,
    drop_cols=None,
):
    """Menghapus kolom tertentu dari dataset dan memperbarui daftar fitur kategorik.

    Args:
        X (pd.DataFrame): Dataset fitur (features).
        cat_features (list[str]): Daftar nama kolom yang bertipe kategorik.
        drop_cols (list[str], optional): Daftar nama kolom yang ingin dihapus.
            Defaults to None.

    Returns:
        tuple[pd.DataFrame, list[str]]: Sebuah tuple berisi:
            - pd.DataFrame: Dataset baru yang kolomnya sudah dihapus.
            - list[str]: Daftar fitur kategorik baru (tanpa kolom yang dihapus).
    """
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
    """Mengevaluasi model CatBoost menggunakan pendekatan holdout (train-val split).

    Args:
        X_train (pd.DataFrame): Dataset fitur untuk training.
        X_val (pd.DataFrame): Dataset fitur untuk validasi.
        y_train (pd.Series): Target untuk training.
        y_val (pd.Series): Target untuk validasi.
        cat_features (list[str]): Daftar nama kolom kategorik.
        drop_cols (list[str], optional): Daftar kolom yang akan diabaikan/dihapus
            saat training dan inferensi. Defaults to None.
        params (dict, optional): Parameter custom untuk inisialisasi CatBoost.
            Defaults to None.

    Returns:
        float: Nilai RMSE (Root Mean Squared Error) pada data validasi.
    """
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
    """Melakukan evaluasi cross-validation untuk model CatBoost.

    Jika `splitter` yang digunakan adalah `GroupKFold`, maka pengelompokan
    akan dilakukan berdasarkan kolom `source_id`.

    Args:
        X (pd.DataFrame): Dataset fitur.
        y (pd.Series): Dataset target.
        cat_features (list[str]): Daftar nama kolom kategorik.
        splitter (object): Objek cross-validator dari scikit-learn
            (misalnya KFold, StratifiedKFold, GroupKFold).
        params (dict, optional): Parameter custom untuk model CatBoost.
            Defaults to None.

    Returns:
        tuple[float, float, list[float]]: Sebuah tuple berisi:
            - float: Rata-rata (mean) RMSE dari semua fold.
            - float: Standar deviasi (std) RMSE dari semua fold.
            - list[float]: Daftar skor RMSE untuk masing-masing fold.
    """
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
    """Mengevaluasi satu kombinasi set fitur (setelah drop kolom tertentu) dengan CV.

    Args:
        X (pd.DataFrame): Dataset fitur.
        y (pd.Series): Dataset target.
        cat_features (list[str]): Daftar nama kolom kategorik.
        splitter (object): Objek cross-validator scikit-learn.
        drop_cols (list[str], optional): Kolom yang ingin di-drop sebelum CV.
            Defaults to None.
        params (dict, optional): Parameter model. Defaults to None.

    Returns:
        tuple[float, float, list[float]]: Hasil fungsi `cross_validate_catboost`
            (mean RMSE, std RMSE, daftar skor).
    """
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


def run_experiments(
    X,
    y,
    cat_feats,
    splitter,
    experiments,
    params=None,
):
    """Jalankan evaluasi (evaluate_feature_set) untuk beberapa skenario penghapusan kolom.

    Args:
        X (pd.DataFrame): Dataset fitur.
        y (pd.Series): Dataset target.
        cat_feats (list[str]): Daftar nama kolom kategorik.
        splitter (object): Objek cross-validator scikit-learn.
        experiments (dict[str, list[str]]): Mapping nama eksperimen ke daftar
            kolom yang akan di-drop.
        params (dict, optional): Parameter model CatBoost. Defaults to None.

    Returns:
        dict[str, tuple[float, float]]: Dictionary hasil eksperimen dengan
            mapping nama eksperimen ke tuple (mean_rmse, std_rmse).
    """
    results = {}
    for name, drop_cols in experiments.items():
        mean_rmse, std_rmse, _ = evaluate_feature_set(
            X,
            y,
            cat_feats,
            splitter,
            drop_cols=drop_cols,
            params=params,
        )
        results[name] = (mean_rmse, std_rmse)
    return results


def get_oof_predictions(X, y, cat_features, splitter, params=None):
    """Kumpulkan Out-Of-Fold (OOF) predictions sejajar dengan index asli X.

    Berbeda dengan cross_validate_catboost yang hanya mengembalikan RMSE agregat,
    fungsi ini menghasilkan prediksi per-baris sehingga bisa digunakan untuk error
    analysis seperti analisis residual, segmentasi, atau identifikasi kasus ekstrem.

    Args:
        X (pd.DataFrame): Dataset fitur.
        y (pd.Series): Dataset target.
        cat_features (list[str]): Daftar nama kolom kategorik.
        splitter (object): Objek cross-validator scikit-learn (StratifiedKFold,
            GroupKFold, dll).
        params (dict, optional): Parameter model CatBoost. Defaults to None.

    Returns:
        pd.Series: Prediksi OOF dengan index yang sama seperti target `y`.
    """
    oof_preds = np.full(len(y), np.nan)
    groups = X["source_id"] if isinstance(splitter, GroupKFold) else None

    for train_idx, val_idx in splitter.split(X, X["source_id"], groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]

        model = build_catboost(params)
        model.fit(X_train, y_train, cat_features=cat_features)

        oof_preds[val_idx] = model.predict(X_val)

    return pd.Series(oof_preds, index=y.index, name="oof_pred")
