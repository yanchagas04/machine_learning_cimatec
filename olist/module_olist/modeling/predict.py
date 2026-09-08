from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from module_olist.modeling.split import FEATURES

# Threshold padrão encontrado pelo evaluate_models para o LightGBM
DEFAULT_THRESHOLD = 0.13


def load_model(model_path: Path):
    """
    Carrega um pipeline treinado serializado com joblib.

    Args:
        model_path: Caminho para o arquivo .pkl do modelo.

    Returns:
        Pipeline sklearn carregado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {model_path}\n"
            "Execute o pipeline de treino (python -m module_olist.main) primeiro."
        )

    logger.info(f"Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    logger.info("Modelo carregado com sucesso.")
    return model


def predict(
    model,
    X: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """
    Realiza a predição de atraso de entrega para um conjunto de pedidos.

    Aplica o threshold ótimo encontrado durante a avaliação para converter
    probabilidades em classes binárias (0 = no prazo, 1 = atrasado).

    Args:
        model: Pipeline sklearn carregado via load_model().
        X: DataFrame com as features esperadas pelo modelo.
              Deve conter as colunas definidas em FEATURES.
        threshold: Ponto de corte para classificação binária.
                   Padrão: 0.13 (otimizado para F1 no hold-out).

    Returns:
        DataFrame com as colunas:
        - ``proba_late``: probabilidade de atraso (0.0 – 1.0)
        - ``is_late_pred``: classe predita (0 ou 1)
    """
    # Garante que apenas as features esperadas sejam enviadas ao modelo
    missing = [col for col in FEATURES if col not in X.columns]
    if missing:
        raise ValueError(
            f"Features ausentes no DataFrame de entrada: {missing}\n"
            f"Features esperadas: {FEATURES}"
        )

    logger.info(f"Realizando predição para {len(X):,} amostras (threshold={threshold})...")

    proba = model.predict_proba(X[FEATURES])[:, 1]
    pred  = (proba >= threshold).astype(np.int8)

    result = pd.DataFrame(
        {"proba_late": proba, "is_late_pred": pred},
        index=X.index,
    )

    n_late = pred.sum()
    logger.info(
        f"Predição concluída — "
        f"{n_late:,} atrasados ({n_late / len(pred) * 100:.1f}%) "
        f"de {len(pred):,} pedidos."
    )

    return result
