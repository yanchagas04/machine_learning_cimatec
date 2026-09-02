from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
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


def train_best_model(
    best_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ProbaWrapper:
    """
    Treina somente o pipeline do modelo selecionado pela CV e retorna
    um ProbaWrapper com a saída de predict_proba normalizada para 1D.

    Args:
        best_name: Nome do modelo vencedor (deve ser chave de get_pipelines()).
        X_train: Features de treino.
        y_train: Target de treino.

    Returns:
        ProbaWrapper com o modelo treinado.
    """
    pipelines = get_pipelines()

    if best_name not in pipelines:
        raise ValueError(
            f"Modelo '{best_name}' não encontrado. "
            f"Opções disponíveis: {list(pipelines)}"
        )

    logger.info(f"Treinando modelo selecionado: {best_name}...")
    pipeline = pipelines[best_name]
    pipeline.fit(X_train, y_train)
    logger.info(f"Treinamento de {best_name} concluído.")

    return ProbaWrapper(pipeline)


def save_model(
    model: ProbaWrapper,
    name: str,
    models_dir: Path,
) -> Path:
    """
    Persiste o pipeline treinado em disco usando joblib.

    O arquivo é salvo como `<models_dir>/<name_sanitizado>.pkl`,
    onde espaços são substituídos por underscores e letras são
    convertidas para minúsculas.

    Args:
        model: ProbaWrapper com o pipeline treinado.
        name: Nome do modelo (usado para compor o nome do arquivo).
        models_dir: Diretório de destino (deve existir).

    Returns:
        Path do arquivo .pkl salvo.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = name.lower().replace(" ", "_") + ".pkl"
    model_path = models_dir / filename

    # Persiste o pipeline interno (sklearn-compatível), não o wrapper
    joblib.dump(model._model, model_path)

    logger.info(f"Modelo salvo em: {model_path}")
    return model_path


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