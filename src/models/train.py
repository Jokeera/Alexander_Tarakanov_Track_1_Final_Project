"""
Stage: TRAIN
Обучение модели (Pipeline) с подбором гиперпараметров,
логированием результатов в MLflow и сохранением артефактов.
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated",
)


from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .pipeline import create_pipeline, get_model_params


# ==========================
# ✅ Utility functions
# ==========================

def load_params():
    """Load full pipeline config from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data(train_path: str, test_path: str):
    """
    Load already engineered datasets (train_features.csv / test_features.csv).
    Target column strictly named 'target' — enforced earlier in pipeline.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return (
        train_df.drop(columns=["target"]),
        train_df["target"],
        test_df.drop(columns=["target"]),
        test_df["target"],
    )


def calculate_metrics(y_true, y_pred, y_proba):
    """Business-relevant binary classification metrics"""
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def plot_roc_curve(y_true, y_proba, save_path):
    """ROC-AUC curve — важный визуальный индикатор качества для скоринга"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Confusion matrix — важно для оценки Recall/Precision деградации"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


# ==========================
# ✅ Core Model Training Logic
# ==========================

def train_model(X_train, y_train, X_test, y_test, params):
    """Train model → tune hyperparameters → log → save artifacts"""

    model_type = params["model"]["algorithm"]
    model_params = get_model_params(model_type, params["model"])

    # MLflow — experiment tracking
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        print(f"🚀 Training model: {model_type}")

        # Base pipeline
        pipeline = create_pipeline(model_type, model_params)

        # Hyperparameter grids
        if model_type == "logistic_regression":
            param_grid = {"classifier__C": [0.1, 1.0, 10.0]}
        elif model_type == "random_forest":
            param_grid = {"classifier__n_estimators": [100], "classifier__max_depth": [10, 15]}
        elif model_type == "gradient_boosting":
            param_grid = {"classifier__n_estimators": [50, 100], "classifier__learning_rate": [0.05, 0.1]}
        else:
            param_grid = {}

        # GridSearchCV if applicable
        if param_grid:
            from sklearn.model_selection import GridSearchCV
            print("🔍 Hyperparameter tuning...")
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=params["training"]["cv_folds"],
                scoring=params["training"]["scoring"],
                n_jobs=-1,
                verbose=1,
            )
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
            mlflow.log_metric("best_cv_score", grid.best_score_)
            for k, v in grid.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
        else:
            print("ℹ️ No tuning → using base model params")
            pipeline.fit(X_train, y_train)

        # Predictions & metrics
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred, y_proba)

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        print("\n📊 Test Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # Visual report
        Path("models").mkdir(exist_ok=True)
        mlflow.log_artifact(plot_roc_curve(y_test, y_proba, "models/roc_curve.png"))
        mlflow.log_artifact(plot_confusion_matrix(y_test, y_pred, "models/confusion_matrix.png"))

        # Save trained model
        joblib.dump(pipeline, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl")

        # Export metrics JSON for CI
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("✅ Model trained and saved successfully")

        return pipeline, metrics


def main():
    params = load_params()

    # ✅ use exact feature files from params
    X_train, y_train, X_test, y_test = load_data(
        params["data"]["train_features_path"],
        params["data"]["test_features_path"],
    )

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    _, metrics = train_model(X_train, y_train, X_test, y_test, params)
    print(f"\n🎯 Final ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
