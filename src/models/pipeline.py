"""ML Pipeline creation for credit scoring"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_feature_columns():
    """Define feature columns for preprocessing"""
    
    # Numerical features
    numeric_features = [
        "limit_bal", "age",
        "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
        "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5", "bill_amt6",
        "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
        "utilization_last", "pay_delay_sum", "pay_delay_max",
        "bill_trend", "pay_trend", "bill_avg", "pay_amt_avg", "pay_to_bill_ratio"
    ]
    
    # Categorical features
    categorical_features = [
        "sex", "education", "marriage", "age_bin"
    ]
    
    return numeric_features, categorical_features


def create_preprocessor():
    """Create preprocessing pipeline"""
    
    numeric_features, categorical_features = get_feature_columns()
    
    # Numerical pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def create_pipeline(model_type="logistic_regression", params=None):
    """Create full ML pipeline with preprocessing and model"""
    
    if params is None:
        params = {}
    
    # Get preprocessor
    preprocessor = create_preprocessor()
    
    # Select model
    if model_type == "logistic_regression":
        model = LogisticRegression(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    return pipeline


def get_model_params(model_type, config):
    """Extract model parameters from config"""
    if model_type in config:
        return config[model_type]
    return {}