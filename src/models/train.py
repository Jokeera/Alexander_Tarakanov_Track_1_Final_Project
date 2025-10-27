"""
Stage: TRAIN
Обучение модели + подбор гиперпараметров + логирование в MLflow + сохранение артефактов
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


# ============
# Helpers
# ============

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def load_data(train_path: str, test_path: str):
    """Load pre-engineered feature datasets"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred, y_proba):
    """Binary classification metrics"""
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred, pos_label=1),
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "pr_auc": average_precision_score(y_true, y_proba)
    }


def plot_roc_curve(y_true, y_proba, save_path):
    """Plot ROC-AUC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


# ============
# Core Training
# ============

def train_model(X_train, y_train, X_test, y_test, params):
    model_type = params["model"]["algorithm"]
    model_params = get_model_params(model_type, params["model"])

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)

        print(f"🚀 Training model: {model_type}")

        # ✅ Base pipeline with params (no parameter loss!)
        base_pipeline = create_pipeline(model_type, model_params)

        # ✅ Hyperparameter grids
        if model_type == "logistic_regression":
            param_grid = {
                "classifier__C": [0.1, 1.0, 10.0]
            }
        elif model_type == "random_forest":
            param_grid = {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [10, 15]
            }
        elif model_type == "gradient_boosting":
            param_grid = {
                "classifier__n_estimators": [50, 100],
                "classifier__learning_rate": [0.05, 0.1]
            }
        else:
            param_grid = {}

        # ✅ GridSearchCV (if grid exists)
        if param_grid:
            from sklearn.model_selection import GridSearchCV
            print("🔍 Running hyperparameter tuning...")

            grid = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=params["training"]["cv_folds"],
                scoring=params["training"]["scoring"],
                n_jobs=-1,
                verbose=1,
            )
            grid.fit(X_train, y_train)
            pipeline = grid.best_estimator_
            mlflow.log_metrics({"best_cv_score": grid.best_score_})

            for k, v in grid.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
        else:
            print("⚙️ No hyperparameter grid → fitting base model")
            pipeline = base_pipeline.fit(X_train, y_train)

        # ✅ Predictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # ✅ Metrics
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        print("\n📊 Test Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # ✅ Visualizations
        Path("models").mkdir(exist_ok=True)
        roc_path = plot_roc_curve(y_test, y_proba, "models/roc_curve.png")
        cm_path = plot_confusion_matrix(y_test, y_pred, "models/confusion_matrix.png")
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)

        # ✅ Save model
        model_path = "models/model.pkl"
        joblib.dump(pipeline, model_path)
        mlflow.log_artifact(model_path)

        # ✅ Save metrics JSON
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("✅ Model trained and saved")

        return pipeline, metrics


def main():
    params = load_params()

    # ✅ read feature datasets directly from params
    X_train, y_train, X_test, y_test = load_data(
        params["data"]["train_features_path"],
        params["data"]["test_features_path"]
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    pipeline, metrics = train_model(X_train, y_train, X_test, y_test, params)
    print(f"\n🎯 Final Test ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
