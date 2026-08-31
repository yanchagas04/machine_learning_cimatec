import pandas as pd
from sklearn.pipeline import Pipeline
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


def get_pipelines() -> dict[str, Pipeline]:
    """
    Retorna instâncias frescas (não treinadas) dos pipelines de cada modelo.

    Usado pela cross-validation, que precisa de estimadores brutos para
    realizar o fit/predict internamente em cada fold.
    """
    return {
        'LightGBM': create_lightgbm_pipeline(),
        'XGBoost': create_xgboost_pipeline(),
        'Gradient Boosting': create_gradient_boosting_pipeline(),
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict[str, ProbaWrapper]:
    """
    Treina cada pipeline no conjunto de treino e aplica o ProbaWrapper,
    que normaliza a saída de predict_proba para 1D.
    """
    trained_models = {}

    for name, model in get_pipelines().items():
        model.fit(X_train, y_train)
        trained_models[name] = ProbaWrapper(model)

    return trained_models