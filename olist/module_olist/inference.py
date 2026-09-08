"""
inference.py — Entry point para inferência em novos dados.

Fluxo:
    1. Carrega os dados brutos (orders, itens, customers)
    2. Aplica o mesmo pipeline de features usado no treino
    3. Carrega o modelo salvo em models/
    4. Chama predict() e salva o resultado em data/processed/predictions.csv

Uso:
    uv run python -m module_olist.inference
"""
from pathlib import Path

from loguru import logger

from module_olist.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from module_olist.dataset import load_data
from module_olist.features import create_dataset, create_features
from module_olist.modeling.predict import DEFAULT_THRESHOLD, load_model, predict
from module_olist.modeling.split import FEATURES


def run_inference(
    orders_path: Path | None = None,
    itens_path: Path | None = None,
    customers_path: Path | None = None,
    model_path: Path | None = None,
    output_path: Path | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """
    Executa o pipeline de inferência completo.

    Carrega dados brutos, aplica engenharia de features, carrega o modelo
    e salva as predições em CSV.

    Args:
        orders_path:    Caminho para olist_orders_dataset.csv.
                        Se None, usa RAW_DATA_DIR padrão.
        itens_path:     Caminho para olist_order_items_dataset.csv.
                        Se None, usa RAW_DATA_DIR padrão.
        customers_path: Caminho para olist_customers_dataset.csv.
                        Se None, usa RAW_DATA_DIR padrão.
        model_path:     Caminho para o .pkl do modelo.
                        Se None, usa MODELS_DIR / lightgbm.pkl.
        output_path:    Caminho de saída das predições em CSV.
                        Se None, usa PROCESSED_DATA_DIR / predictions.csv.
        threshold:      Ponto de corte para classificação (padrão: 0.13).
    """
    # ── Caminhos padrão ──────────────────────────────────────────────────────
    orders_path    = orders_path    or RAW_DATA_DIR  / "olist_orders_dataset.csv"
    itens_path     = itens_path     or RAW_DATA_DIR  / "olist_order_items_dataset.csv"
    customers_path = customers_path or RAW_DATA_DIR  / "olist_customers_dataset.csv"
    model_path     = model_path     or MODELS_DIR    / "lightgbm.pkl"
    output_path    = output_path    or PROCESSED_DATA_DIR / "predictions.csv"

    logger.info("=" * 60)
    logger.info("Iniciando pipeline de inferência")
    logger.info("=" * 60)

    # ── 1. Carregamento dos dados brutos ─────────────────────────────────────
    logger.info("Carregando dados brutos...")
    orders, itens, customers = load_data(orders_path, itens_path, customers_path)

    if orders is None or itens is None or customers is None:
        logger.error("Falha ao carregar os dados. Inferência encerrada.")
        return

    # ── 2. Junções e engenharia de features ──────────────────────────────────
    logger.info("Criando dataset e aplicando features...")
    data = create_dataset(orders, itens, customers)
    data = create_features(data)

    logger.info(f"Dataset de inferência: {data.shape[0]:,} linhas x {data.shape[1]} colunas")

    # ── 3. Carregamento do modelo ─────────────────────────────────────────────
    model = load_model(model_path)

    # ── 4. Predição ──────────────────────────────────────────────────────────
    X = data[FEATURES]
    predictions = predict(model, X, threshold=threshold)

    # ── 5. Combina order_id com predições e salva ─────────────────────────────
    output = data[["order_id"]].copy()
    output["proba_late"]   = predictions["proba_late"].values
    output["is_late_pred"] = predictions["is_late_pred"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    logger.info(f"Predições salvas em: {output_path}")
    logger.info("=" * 60)
    logger.info("Inferência concluída.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_inference()
