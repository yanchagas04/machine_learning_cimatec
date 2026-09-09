import pandas as pd
import shap 
from scipy import sparse

def prepare_data_for_shap(pipeline, x):
    """
    Prepara o DataFrame para cálculo do SHAP

    Retira o label do DataFrame, aplica o preprocessor e retorna um DataFrame
    com nomes de features decodificados
    
    Args:
        pipeline: Pipeline de treino completo
        x: DataFrame de treino com features e labels
    """
    #1. Extrai o preprocessor
    preprocessor = pipeline.named_steps["preprocessor"]

    #2. Aplica a transformação
    x_transformed = preprocessor.transform(x)

    if sparse.issparse(x_transformed):
        x_transformed = x_transformed.toarray()

    #Recupera nomes das features
    feature_names = preprocessor.get_feature_names_out()

    #Converte para um DataFrame
    x_transformed = pd.DataFrame(
        x_transformed,
        columns=feature_names,
        index=x.index
    )

    return x_transformed

def create_explainer(pipeline, x_train_raw):
    """
    Cria um explainer SHAP para o modelo

    Args:
        pipeline: Pipeline de treino completo
        x_train_raw: DataFrame de treino com features e labels
    """
    model = pipeline.named_steps["model"]

    #Criar um explainer que entenda as previsões feitas pelo modelo
    explainer = shap.TreeExplainer(model)

    return explainer


def calculate_shap_values(explainer, x_transformed):
    """
    Calcula SHAP values de forma eficiente

    Calcula para um subconjunto de dados para agilizar o processo
    """
    
    shap_values = explainer(x_transformed)

    return shap_values