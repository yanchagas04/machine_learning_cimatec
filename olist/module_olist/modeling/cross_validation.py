import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold, cross_validate

from module_olist.modeling.train import get_pipelines


def cross_validate_models(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Avalia cada pipeline com Stratified K-Fold Cross-Validation.

    A CV é executada sobre o conjunto completo (X, y) antes do split
    treino/teste, garantindo uma estimativa não-viesada da capacidade de
    generalização de cada modelo.

    Args:
        X: DataFrame com as features.
        y: Series com o target binário (is_late).
        n_splits: Número de folds (padrão = 5).
        random_state: Semente para reproduzibilidade.

    Returns:
        Dicionário com os arrays de score por métrica para cada modelo.
        Exemplo: {"LightGBM": {"test_roc_auc": array([...]), ...}, ...}
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    pipelines = get_pipelines()
    results = {}

    logger.info("=" * 60)
    logger.info(f"Iniciando Cross-Validation ({n_splits}-Fold Stratified)")
    logger.info("=" * 60)

    for name, pipeline in pipelines.items():
        logger.info(f"Avaliando: {name}...")

        cv_results = cross_validate(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,  # paraleliza nos núcleos disponíveis
            return_train_score=False,
        )

        results[name] = cv_results

        # Loga o resumo por métrica (média ± desvio padrão)
        logger.info(f"  Resultados de {name} ({n_splits} folds):")
        for metric in scoring:
            scores = cv_results[f"test_{metric}"]
            logger.info(
                f"    {metric:<12}: {scores.mean():.4f} ± {scores.std():.4f}"
                f"  |  folds: {np.round(scores, 4).tolist()}"
            )

    logger.info("=" * 60)
    logger.info("Cross-Validation concluída.")
    logger.info("=" * 60)

    return results
