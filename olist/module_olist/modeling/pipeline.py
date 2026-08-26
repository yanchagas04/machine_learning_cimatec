from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

NUMERICAL_FEATURES = [
    "promised_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight"
]

CATEGORICAL_FEATURES = [
    "purchase_month",
    "purchase_weekday",
    "purchase_hour",
    "customer_state"
]

# montar esteira de pre-processamento
def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # aplicar transformações para features numéricas
            ("numeric", "passthrough", NUMERICAL_FEATURES),
            # aplicar transformações para features categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Cria um pipeline do pré-processamento e treinamento do modelo Gradient Boosting.
    """
    preprocessor = create_preprocessor()
    model = GradientBoostingClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # taxa de aprendizado
        max_depth=3, # profundidade máxima das árvores
        random_state=42 # semente para reproduzibilidade
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("model", model)]
    )

def create_xgboost_pipeline() -> Pipeline:
    """
    Cria um pipeline do pré-processamento e treinamento do modelo XGBoost.
    """
    preprocessor = create_preprocessor()
    model = XGBClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # taxa de aprendizado
        max_depth=3, # profundidade máxima das árvores
        random_state=42, # semente para reproduzibilidade
        use_label_encoder=False, # desabilitar o uso do codificador de rótulos
        eval_metric="logloss" # métrica de avaliação
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("model", model)]
    )

def create_lightgbm_pipeline() -> Pipeline:
    """
    Cria um pipeline do pré-processamento e treinamento do modelo LightGBM.
    """
    preprocessor = create_preprocessor()
    model = LGBMClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # taxa de aprendizado
        max_depth=3, # profundidade máxima das árvores
        random_state=42 # semente para reproduzibilidade
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor),
               ("model", model)]
    )