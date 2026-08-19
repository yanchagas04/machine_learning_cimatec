from module_olist.dataset import agregate_itens
from module_olist.dataset import create_target
import pandas as pd

def create_features(data: pd.DataFrame) -> pd.DataFrame:

    data = data.copy()
    # Calcula quantos dias a empresa prometeu para realizar a entrega,
    # considerando como início o momento da aprovação do pagamento.
    data["promised_days"] = (
        data["order_estimated_delivery_date"]  # Data prometida para a entrega.
        - data["order_approved_at"]            # Data de aprovação do pagamento.
    ).dt.total_seconds().div(86_400)          # Converte segundos para dias => 24 * 60 * 60 = 86.400 segundos


    # Extrai o número do mês em que a compra foi realizada.
    # Exemplo: janeiro = 1, fevereiro = 2, ..., dezembro = 12.
    data["purchase_month"] = (
        data["order_purchase_timestamp"].dt.month
    )


    # Extrai o dia da semana em que a compra foi realizada.
    #
    # O Pandas representa os dias da seguinte forma:
    # 0 = segunda-feira
    # 1 = terça-feira
    # 2 = quarta-feira
    # 3 = quinta-feira
    # 4 = sexta-feira
    # 5 = sábado
    # 6 = domingo
    data["purchase_weekday"] = (
        data["order_purchase_timestamp"].dt.dayofweek
    )


    # Extrai a hora em que a compra foi realizada.
    # Os valores variam de 0 a 23.
    #
    # Exemplo:
    # 0  = meia-noite
    # 8  = 8 horas
    # 14 = 14 horas
    # 23 = 23 horas
    data["purchase_hour"] = (
        data["order_purchase_timestamp"].dt.hour
    )

    return data

def create_dataset(orders, itens, customers):
    """
    Cria o dataset final.
    
    Args:
        orders (pd.DataFrame): Dataframe de pedidos.
        itens (pd.DataFrame): Dataframe de itens.
        customers (pd.DataFrame): Dataframe de clientes.
    
    Returns:
        pd.DataFrame: Dataset final.
    """
    orders = create_target(orders)
    items_agg = agregate_itens(itens)

    data = orders.merge(
        items_agg,
        on="order_id",
        how="left",
        validate="one_to_one"
    )

    data = data.merge(
        customers[['customer_id', 'customer_city', 'customer_state']],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    return data