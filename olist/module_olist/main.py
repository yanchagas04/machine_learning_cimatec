from module_olist.modeling.evaluate import evaluate_models
from pathlib import Path
from loguru import logger

from module_olist.dataset import load_data, save_dataset
from module_olist.features import create_dataset, create_features
from module_olist.modeling.split import split_data, FEATURES, TARGET
from module_olist.modeling.train import train_best_model, save_model
from module_olist.modeling.cross_validation import cross_validate_models

def main():
    """
    Pipeline principal: carrega os dados brutos, realiza as junções,
    aplica engenharia de features e salva o resultado em data/interim.
    """

    # Raiz do projeto
    project_root = Path(__file__).resolve().parents[1]

    # Caminho para os dados brutos
    orders_path    = project_root / "data" / "raw" / "olist_orders_dataset.csv"
    itens_path     = project_root / "data" / "raw" / "olist_order_items_dataset.csv"
    customers_path = project_root / "data" / "raw" / "olist_customers_dataset.csv"

    # Caminho de saída
    output_path = project_root / "data" / "interim" / "dataset.csv"

    # Diretório para salvar o modelo treinado
    models_dir = project_root / "models"

    logger.info("Iniciando pipeline de dados...")

    # 1. Carregamento dos dados brutos
    logger.info("Carregando dados brutos...")
    orders, itens, customers = load_data(orders_path, itens_path, customers_path)

    if orders is None or itens is None or customers is None:
        logger.error("Falha ao carregar os dados. Pipeline encerrado.")
        return

    # 2. Junções: cria target, agrega itens e faz merge com clientes
    logger.info("Criando dataset com junções...")
    data = create_dataset(orders, itens, customers)

    # 3. Engenharia de features
    logger.info("Criando features...")
    data = create_features(data)

    logger.info(f"Dataset final: {data.shape[0]:,} linhas x {data.shape[1]} colunas")

    # 4. Persistência do dataset intermediário
    logger.info(f"Salvando dataset em: {output_path}")
    save_dataset(data, output_path)

    logger.info("Pipeline concluído com sucesso!")

    # 5. Cross-Validation — avaliação robusta antes do split
    # Retorna os scores de todos os modelos e o nome do melhor (por ROC AUC médio).
    X = data[FEATURES]
    y = data[TARGET]
    _, best_name = cross_validate_models(X, y)

    # 6. Split treino/teste (hold-out final)
    X_train, X_test, y_train, y_test = split_data(data)

    # 7. Treina APENAS o modelo selecionado pela CV
    best_model = train_best_model(best_name, X_train, y_train)

    # 8. Persiste o modelo treinado em models/<nome>.pkl
    save_model(best_model, best_name, models_dir)

    # 9. Avaliação no hold-out com busca de threshold ótimo
    evaluate_models({best_name: best_model}, X_test, y_test)


if __name__ == "__main__":
    main()
