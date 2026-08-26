import pandas as pd
from module_olist.modeling.pipeline import (
    create_lightgbm_pipeline,
    create_xgboost_pipeline,
    create_gradient_boosting_pipeline
)


class ProbaWrapper:
    """
    Wrapper que faz predict_proba retornar apenas a probabilidade da classe
    positiva (coluna 1), tornando a saída 1D em vez de (n, 2).
    """
    def __init__(self, model):
        self._model = model

    def predict_proba(self, X):
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self._model.predict(X)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
):

    models = {
        'LightGBM': create_lightgbm_pipeline(),
        'XGBoost': create_xgboost_pipeline(),
        'Gradient Boosting': create_gradient_boosting_pipeline()
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = ProbaWrapper(model)

    return trained_models