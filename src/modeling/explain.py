import shap
from catboost import Pool


def compute_shap_values(model, X, cat_feats):
    """Menghitung SHAP values untuk model CatBoost menggunakan TreeExplainer.

    Data dibungkus sebagai ``catboost.Pool`` karena ``shap.TreeExplainer`` dapat gagal pada model CatBoost dengan fitur kategorik apabila input diberikan sebagai DataFrame biasa. Lihat: https://github.com/shap/shap/issues/1061.

    Args:
        model (CatBoostRegressor):
            Model CatBoostRegressor yang sudah di-fit.
        X (pd.DataFrame):
            Data yang akan digunakan untuk menghitung SHAP values.
        cat_feats (list[str]):
            Daftar nama kolom kategorik. Harus sama dengan fitur kategorik
            yang digunakan saat ``model.fit()``.

    Returns:
        shap.Explanation:
            Objek SHAP yang dapat langsung digunakan untuk
            ``shap.plots.bar()``, ``shap.summary_plot()``, atau
            ``shap.plots.scatter()``.
    """
    pool = Pool(X, cat_features=cat_feats)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pool)
    shap_values.feature_names = list(X.columns)
    shap_values.data = X.values
    return shap_values
