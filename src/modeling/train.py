from catboost import CatBoostRegressor

from src.config import CATBOOST_DEFAULT_PARAMS


def build_catboost(params=None):
    """Membuat dan mengembalikan instance model CatBoostRegressor.

    Fungsi ini akan memuat parameter default dari `CATBOOST_DEFAULT_PARAMS`.
    Jika argumen `params` diberikan, nilai-nilai di dalamnya akan menimpa
    (override) atau menambahkan parameter default tersebut.

    Args:
        params (dict, optional): Dictionary berisi parameter custom untuk
            menginisialisasi model CatBoost. Defaults to None.

    Returns:
        CatBoostRegressor: Objek model CatBoostRegressor yang belum dilatih
            (unfitted) dan siap digunakan.
    """
    model_params = CATBOOST_DEFAULT_PARAMS.copy()

    if params:
        model_params.update(params)

    return CatBoostRegressor(**model_params)
