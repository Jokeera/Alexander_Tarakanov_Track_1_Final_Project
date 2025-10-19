"""Model training script with MLflow tracking"""

import pandas as pd
import numpy as np
import yaml
import json
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from .pipeline import create_pipeline, get_model_params


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params


def load_data(train_path, test_path):
    """Load train and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]
    
    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate classification metrics"""
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1_score": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_proba)
    }
    return metrics


def plot_roc_curve(y_true, y_proba, save_path="models/roc_curve.png"):
    """Plot and save ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path


def plot_confusion_matrix(y_true, y_pred, save_path="models/confusion_matrix.png"):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return save_path


def train_model(X_train, y_train, X_test, y_test, params):
    """Train model with MLflow tracking"""
    
    # Get model configuration
    model_type = params["model"]["algorithm"]
    model_params = get_model_params(model_type, params["model"])
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", model_type)
        for key, value in model_params.items():
            mlflow.log_param(key, value)
        
        # Create base pipeline
        print(f"Training {model_type} with hyperparameter tuning...")
        base_pipeline = create_pipeline(model_type, {})
        
        # Define hyperparameter grid (small for fast execution)
        if model_type == "logistic_regression":
            param_grid = {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__penalty': ['l2']
            }
        elif model_type == "random_forest":
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [10, 15],
                'classifier__min_samples_split': [20, 30]
            }
        elif model_type == "gradient_boosting":
            param_grid = {
                'classifier__n_estimators': [50, 100],
                'classifier__learning_rate': [0.05, 0.1],
                'classifier__max_depth': [3, 5]
            }
        else:
            param_grid = {}
        
        # GridSearchCV for hyperparameter tuning
        if param_grid:
            from sklearn.model_selection import GridSearchCV
            grid_search = GridSearchCV(
                base_pipeline,
                param_grid,
                cv=params["training"]["cv_folds"],
                scoring=params["training"]["scoring"],
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            
            # Log best parameters
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            for key, value in grid_search.best_params_.items():
                mlflow.log_param(f"best_{key}", value)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
        else:
            # If no param_grid, use provided params
            pipeline = create_pipeline(model_type, model_params)
            pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = pipeline.predict(X_train)
        y_proba_train = pipeline.predict_proba(X_train)[:, 1]
        
        y_pred_test = pipeline.predict(X_test)
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train, y_proba_train)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_proba_test)
        
        # Log train metrics
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        
        # Print metrics
        print("\nTrain Metrics:")
        for metric_name, metric_value in train_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print("\nTest Metrics:")
        for metric_name, metric_value in test_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Create visualizations
        roc_path = plot_roc_curve(y_test, y_proba_test)
        cm_path = plot_confusion_matrix(y_test, y_pred_test)
        
        # Log artifacts
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(cm_path)
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Save model locally
        Path("models").mkdir(exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(pipeline, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save metrics to JSON
        metrics_dict = {
            "train": train_metrics,
            "test": test_metrics
        }
        metrics_path = "models/metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
        
        return pipeline, test_metrics


def main():
    """Main training function"""
    # Load parameters
    params = load_params()
    
    # Load data
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(
        params["data"]["train_path"].replace("train.csv", "train_features.csv"),
        params["data"]["test_path"].replace("test.csv", "test_features.csv")
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Target distribution in train: {y_train.value_counts(normalize=True).to_dict()}")
    
    # Train model
    pipeline, metrics = train_model(X_train, y_train, X_test, y_test, params)
    
    print("\n✅ Training completed successfully!")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()