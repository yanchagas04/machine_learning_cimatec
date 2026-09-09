import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger

from module_olist.config import (
    FIGURES_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
)
from module_olist.modeling.predict import load_model
from module_olist.modeling.split import FEATURES
from module_olist.modeling.interpret import (
    prepare_data_for_shap,
    create_explainer,
    calculate_shap_values,
)


def _save_waterfall(shap_values, idx: int, filename: str, title: str) -> None:
    """Gera e salva um waterfall plot SHAP para uma amostra específica."""
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    plt.title(title, fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Waterfall salvo em: {FIGURES_DIR / filename}")


def main():
    # ── 1. Carrega o dataset intermediário (já com features criadas) ──────────
    dataset_path = INTERIM_DATA_DIR / "dataset.csv"
    logger.info(f"Carregando dataset de: {dataset_path}")
    data = pd.read_csv(dataset_path)

    X = data[FEATURES]
    logger.info(f"Amostras para explicação: {len(X):,}")

    # ── 2. Carrega o pipeline treinado ────────────────────────────────────────
    model_path = MODELS_DIR / "lightgbm.pkl"
    pipeline = load_model(model_path)

    # ── 3. Transforma X com o preprocessor do pipeline ───────────────────────
    # prepare_data_for_shap aplica o ColumnTransformer e devolve um DataFrame
    # com os nomes corretos das features (numéricas + one-hot encoded).
    logger.info("Preparando dados para SHAP (aplicando preprocessor)...")
    X_transformed = prepare_data_for_shap(pipeline, X)

    # ── 4. Cria o explainer SHAP ──────────────────────────────────────────────
    logger.info("Criando TreeExplainer...")
    explainer = create_explainer(pipeline, X)

    # ── 5. Calcula os SHAP values ─────────────────────────────────────────────
    logger.info("Calculando SHAP values...")
    shap_values = calculate_shap_values(explainer, X_transformed)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 6. Plot global: beeswarm ──────────────────────────────────────────────
    logger.info("Gerando gráfico SHAP summary (global)...")
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico salvo em: {FIGURES_DIR / 'shap_summary.png'}")

    # ── 7. Plots locais: waterfall por amostra representativa ─────────────────
    # Seleciona amostras com as probabilidades mais extremas para cada classe,
    # garantindo exemplos ilustrativos e bem definidos.
    logger.info("Selecionando amostras representativas para explicação local...")
    probas = pipeline.predict_proba(X)[:, 1]

    # Amostra com maior probabilidade de atraso (caso "atrasado")
    idx_late    = int(probas.argmax())
    # Amostra com menor probabilidade de atraso (caso "no prazo")
    idx_on_time = int(probas.argmin())

    logger.info(
        f"  Amostra 'atrasado'  → índice {idx_late}  | proba={probas[idx_late]:.3f}"
    )
    logger.info(
        f"  Amostra 'no prazo'  → índice {idx_on_time} | proba={probas[idx_on_time]:.3f}"
    )

    logger.info("Gerando waterfall plots locais...")
    _save_waterfall(
        shap_values, idx_late,
        filename="shap_waterfall_late.png",
        title=f"Explicação local — Pedido ATRASADO (proba={probas[idx_late]:.3f})",
    )
    _save_waterfall(
        shap_values, idx_on_time,
        filename="shap_waterfall_on_time.png",
        title=f"Explicação local — Pedido NO PRAZO (proba={probas[idx_on_time]:.3f})",
    )


if __name__ == "__main__":
    main()