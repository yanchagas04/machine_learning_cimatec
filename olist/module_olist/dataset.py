import pandas as pd
from pathlib import Path
from loguru import logger

def load_data(orders_path: Path, itens_path: Path, customers_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de pedidos, itens e clientes a partir dos arquivos CSV.

    Args:
        orders_path (Path): Caminho para o arquivo CSV de pedidos.
        itens_path (Path): Caminho para o arquivo CSV de itens.
        customers_path (Path): Caminho para o arquivo CSV de clientes.

    Return:
        Orders: Dataframe de pedidos
        Itens: Dataframe de itens
        Customers: Dataframe de clientes
    """

    try:
        orders = pd.read_csv(
            orders_path,
            sep=",",
            encoding="utf-8",
            parse_dates=["order_purchase_timestamp", 
                        "order_approved_at", 
                        "order_delivered_carrier_date",
                        "order_delivered_customer_date",
                        "order_estimated_delivery_date"]
        )
        
        itens = pd.read_csv(itens_path)
        customers = pd.read_csv(customers_path)
        
        logger.info(f"Orders loaded: {len(orders)}")
        logger.info(f"Itens loaded: {len(itens)}")
        logger.info(f"Customers loaded: {len(customers)}")
            
        return orders, itens, customers

    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None, None, None

def save_dataset(dataset: pd.DataFrame, output_path: Path) -> None:
    """
    Salva os dados no caminho especificado.
    
    Args:
        dataset (pd.DataFrame): Dataframe a ser salvo.
        path (Path): Caminho para salvar o dataframe.
    """
    try:
        dataset.to_csv(output_path, sep=",", encoding="utf-8", index=False)
        logger.info(f"Dataset salvo em {output_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar dataset: {e}")

def create_target(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a variável-alvo do problema.
    
    Args:
        orders (pd.DataFrame): Dataframe de pedidos.
    
    Returns:
        pd.DataFrame: Dataframe com a variável-alvo.
    """
    # Seleciona apenas os pedidos que podem ser utilizados para construir
    # o histórico de entregas atrasadas e realizadas dentro do prazo.
    delivered_orders = orders.loc[
        # Mantém somente pedidos que foram efetivamente entregues.
        orders["order_status"].eq("delivered")

        # Remove pedidos sem a data real em que o cliente recebeu a compra.
        # Essa data é necessária para saber se o pedido atrasou.
        & orders["order_delivered_customer_date"].notna()

        # Remove pedidos sem a data de entrega prometida ao cliente.
        # Sem essa informação, não é possível comparar o previsto com o realizado.
        & orders["order_estimated_delivery_date"].notna()

        # Mantém somente pedidos com a data de aprovação do pagamento.
        # Esse é o momento definido para realizar a previsão.
        & orders["order_approved_at"].notna()
    ].copy()  # Cria uma cópia independente para evitar alterações no DataFrame original.


    # Cria a variável-alvo do problema:
    # 1 → pedido entregue depois da data prometida;
    # 0 → pedido entregue dentro do prazo ou antes da data prometida.
    delivered_orders["is_late"] = (
        delivered_orders["order_delivered_customer_date"] > delivered_orders["order_estimated_delivery_date"]
    ).astype("int8")  # Armazena 0 e 1 usando um tipo inteiro que ocupa menos memória.


    # Apresenta a quantidade total de pedidos antes da aplicação dos filtros.
    logger.info(f"Pedidos originais: {len(orders):,}")


    # Apresenta quantos pedidos permaneceram após aplicação do filtro
    logger.info(f"Pedidos filtrados: {len(delivered_orders):,}")
        

    # Mostra a quantidade de pedidos em cada classe:
    # 0 = entregue no prazo;
    # 1 = entregue com atraso.
    logger.info(
        delivered_orders["is_late"].value_counts(dropna=False)
    )

    return delivered_orders

def agregate_itens(items: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os itens por pedido.
    
    Args:
        items (pd.DataFrame): Dataframe de itens.
    
    Returns:
        pd.DataFrame: Dataframe com os itens agregados por pedido.
    """
    # A tabela de itens possui uma linha para cada item presente no pedido.
    # Portanto, um mesmo order_id pode aparecer várias vezes.
    #
    # Como o objetivo é construir uma base com uma linha por pedido,
    # precisamos agrupar os itens antes de integrar essa tabela às demais.
    items_agg = (
        items.groupby(
            "order_id",       # Agrupa todos os itens pertencentes ao mesmo pedido.
            as_index=False,   # Mantém order_id como uma coluna comum.
        )
        .agg(
            # Conta quantas linhas de itens existem em cada pedido.
            # Um pedido com três produtos registrados terá item_count igual a 3.
            # nome_da_nova_coluna=("coluna_original", "função")
            item_count=("order_item_id", "count"),

            # Conta quantos vendedores diferentes participam do pedido.
            # O nunique evita contar o mesmo vendedor mais de uma vez.
            # nome_da_nova_coluna=("coluna_original", "função")
            seller_count=("seller_id", "nunique"),

            # Soma os preços dos itens para obter o valor total dos produtos
            # presentes no pedido.
            # nome_da_nova_coluna=("coluna_original", "função")
            total_price=("price", "sum"),

            # Soma o frete de todos os itens para obter o valor total de frete
            # associado ao pedido.
            total_freight=("freight_value", "sum"),
        )
    )


    # Verifica se cada pedido aparece somente uma vez após a agregação.
    #
    # Se a condição for falsa, o Python interromperá a execução e lançará
    # um AssertionError. Essa checagem ajuda a garantir que a unidade de
    # análise da nova tabela é realmente o pedido.
    assert items_agg["order_id"].is_unique

    return items_agg
