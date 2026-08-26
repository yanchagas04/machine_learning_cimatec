from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

NUMERIC_FEATURES = [
    "promise_days",
    "item_count",
    "seller_count",
    "total_price",
    "total_freight"
]

CATEGORICAL_FEATURES = [
    "purchase_hour",
    "purchase_weekday",
    "purchase_month",
    "customer_state"
]

def create_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            # Nas colunas numéricas
            ("numeric", "passthrough", NUMERIC_FEATURES),
            # Nas colunas categóricas
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )

def create_gradient_boosting_pipeline() -> Pipeline:
    """
    Create a pipeline that includes preprocessing and training model.
    """

    preprocessor = create_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # Taxa de aprendizado
        max_depth=3, # Profundidade máxima das árvores
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
                ("model", model)]
    )

    return pipeline


def create_xgboost_pipeline() -> Pipeline:

    preprocessor = create_preprocessor()

    model = XGBClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # Taxa de aprendizado
        max_depth=3, # Profundidade máxima das árvores
        random_state=42
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
                ("model", model)]
    )

    return pipeline


def create_lightgbm_pipeline() -> Pipeline:
    """
    Create a pipeline that includes preprocessing and training model.
    """

    preprocessor = create_preprocessor()

    model = LGBMClassifier(
        n_estimators=100, # número de árvores na floresta
        learning_rate=0.1, # Taxa de aprendizado
        max_depth=3, # Profundidade máxima das árvores
        random_state=42
    )

    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor),
                ("model", model)]
    )

    return pipeline