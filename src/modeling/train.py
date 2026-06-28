from catboost import CatBoostRegressor

from src.config import CATBOOST_DEFAULT_PARAMS


def build_catboost(params=None):
    model_params = CATBOOST_DEFAULT_PARAMS.copy()

    if params:
        model_params.update(params)

    return CatBoostRegressor(**model_params)
