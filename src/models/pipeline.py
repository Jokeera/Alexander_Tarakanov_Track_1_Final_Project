"""ML Pipeline creation for credit scoring (final)"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns():
    """Список признаков, ожидаемый после stage FEATURES (train_features.csv/test_features.csv)."""

    # Числовые признаки (engineered + базовые числовые)
    numeric_features = [
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
        "utilization_last", "pay_delay_sum", "pay_delay_max",
        "bill_trend", "pay_trend", "bill_avg", "pay_amt_avg", "pay_to_bill_ratio",
    ]

    # Категориальные (низкая кардинальность → можно плотный OHE)
    categorical_features = ["sex", "education", "marriage", "age_bin"]

    return numeric_features, categorical_features


def create_preprocessor():
    """Пре-процессинг: числовые → median+scaler; категориальные → most_frequent+OHE."""

    numeric_features, categorical_features = get_feature_columns()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Важно:
    # - handle_unknown='ignore' для стабильности на инференсе (новые бины/категории)
    # - drop=None (НЕ дропаем первый столбец, чтобы не ловить рассинхроны оффлайн/онлайн)
    # - sparse=False (категорий мало → плотная матрица ок; совместимо со старыми sklearn)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def create_pipeline(model_type: str = "logistic_regression", params: dict | None = None):
    """Полный sklearn Pipeline: preprocess → classifier."""
    if params is None:
        params = {}

    preprocessor = create_preprocessor()

    # Выбор модели (параметры приходят из params.yaml через train.py)
    if model_type == "logistic_regression":
        # Поддерживает predict_proba при solver='lbfgs' (см. params.yaml)
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def get_model_params(model_type: str, config: dict) -> dict:
    """Вернуть параметры модели из config[model] по ключу алгоритма."""
    # ожидается, что в config уже есть под-ключи: logistic_regression / random_forest / gradient_boosting
    return config.get(model_type, {})
