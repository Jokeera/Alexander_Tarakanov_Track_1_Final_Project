"""
Stage: TRAIN — FINAL
Обучение + CV + дисбаланс классов + выбор порога + MLflow (модель/артефакты/метрики).
"""
from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, yaml, mlflow, mlflow.sklearn

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .pipeline import create_pipeline, get_model_params


# ------------------ utils ------------------
def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(train_path: str, test_path: str):
    tr = pd.read_csv(train_path); ts = pd.read_csv(test_path)
    return tr.drop(columns=["target"]), tr["target"], ts.drop(columns=["target"]), ts["target"]

def calc_metrics(y_true, y_pred, y_proba):
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc":  average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
    }

def plot_roc(y_true, y_proba, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", linewidth=2)
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def plot_pr(y_true, y_proba, path):
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, linewidth=2, label=f"AP={ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()
    return path

def plot_cm(y_true, y_pred, path, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()
    return path


# ------------------ core ------------------
def train_model(X_train, y_train, X_test, y_test, params):
    model_type  = params["model"]["algorithm"]
    model_cfg   = get_model_params(model_type, params["model"])
    random_state = params["training"].get("random_state", 42)

    # дисбаланс (веса классов)
    # LR/RF — class_weight="balanced"; GB — будем давать sample_weight
    if model_type == "logistic_regression":
        model_cfg.setdefault("max_iter", 500)
        model_cfg.setdefault("class_weight", "balanced")
        model_cfg.setdefault("solver", "liblinear")
        model_cfg.setdefault("random_state", random_state)
    elif model_type == "random_forest":
        model_cfg.setdefault("n_estimators", 300)
        model_cfg.setdefault("max_depth", 12)
        model_cfg.setdefault("class_weight", "balanced")
        model_cfg.setdefault("n_jobs", -1)
        model_cfg.setdefault("random_state", random_state)
    elif model_type == "gradient_boosting":
        model_cfg.setdefault("n_estimators", 200)
        model_cfg.setdefault("learning_rate", 0.05)
        model_cfg.setdefault("max_depth", 3)
        model_cfg.setdefault("random_state", random_state)

    # MLflow
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    cv_folds = params["training"].get("cv_folds", 5)
    scoring  = params["training"].get("scoring", "roc_auc")

    # сетки гиперов
    if model_type == "logistic_regression":
        param_grid = {"classifier__C": [0.1, 1.0, 3.0, 10.0]}
    elif model_type == "random_forest":
        param_grid = {"classifier__n_estimators": [200, 300, 500],
                      "classifier__max_depth": [10, 12, 16]}
    elif model_type == "gradient_boosting":
        param_grid = {"classifier__n_estimators": [150, 200, 300],
                      "classifier__learning_rate": [0.03, 0.05, 0.1],
                      "classifier__max_depth": [2, 3]}
    else:
        param_grid = {}

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        for k, v in model_cfg.items():
            mlflow.log_param(f"init_{k}", v)

        pipeline = create_pipeline(model_type, model_cfg)

        # sample_weight для GB (и разрешено для RF)
        sample_weight = None
        if model_type in {"gradient_boosting", "random_forest"}:
            # веса классов (balanced)
            # w1 / w0 ≈ n0 / n1
            p1 = (y_train == 1).mean()
            w0, w1 = 0.5 / (1 - p1 + 1e-12), 0.5 / (p1 + 1e-12)
            sample_weight = np.where(y_train.values == 1, w1, w0)

        # CV с стратификацией
        if param_grid:
            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
            )
            fit_kwargs = {}
            if sample_weight is not None:
                fit_kwargs = {"classifier__sample_weight": sample_weight}
            grid.fit(X_train, y_train, **fit_kwargs)
            pipeline = grid.best_estimator_
            mlflow.log_metric("best_cv_score", grid.best_score_)
            for k, v in grid.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
        else:
            if sample_weight is not None:
                pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
            else:
                pipeline.fit(X_train, y_train)

        # инференс
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # подбор порога по F1
        prec, rec, thr = precision_recall_curve(y_test, y_proba)
        f1s = 2 * prec * rec / (prec + rec + 1e-12)
        best_idx = int(f1s.argmax())
        best_thr = float(thr[max(best_idx - 1, 0)])  # из-за длины thr на 1 меньше
        mlflow.log_metric("best_threshold_f1", best_thr)

        y_pred_default = (y_proba >= 0.5).astype(int)
        y_pred_best    = (y_proba >= best_thr).astype(int)

        metrics_default = calc_metrics(y_test, y_pred_default, y_proba)
        metrics_best    = calc_metrics(y_test, y_pred_best,    y_proba)

        # лог метрик
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics_default.items()})
        mlflow.log_metrics({f"test_best_{k}": v for k, v in metrics_best.items()})

        print("\n📊 Test @0.5:", {k: round(v,4) for k,v in metrics_default.items()})
        print("📊 Test @best-thr:", {k: round(v,4) for k,v in metrics_best.items()})

        # артефакты
        Path("models").mkdir(exist_ok=True)
        mlflow.log_artifact(plot_roc(y_test, y_proba, "models/roc_curve.png"))
        mlflow.log_artifact(plot_pr (y_test, y_proba, "models/pr_curve.png"))
        mlflow.log_artifact(plot_cm (y_test, y_pred_default, "models/confusion_default.png", "CM @0.5"))
        mlflow.log_artifact(plot_cm (y_test, y_pred_best,    "models/confusion_best.png", "CM @best-thr"))

        # сохранение модели
        joblib.dump(pipeline, "models/model.pkl")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")  # как модель, не просто файл
        with open("models/metrics.json", "w") as f:
            json.dump({
                "default": metrics_default,
                "best_thr": {"threshold": best_thr, **metrics_best}
            }, f, indent=2)

        print("✅ Model trained and saved")
        return pipeline, {"default": metrics_default, "best": metrics_best, "thr": best_thr}


def main():
    P = load_params()
    Xtr, ytr, Xte, yte = load_data(P["data"]["train_features_path"], P["data"]["test_features_path"])
    print(f"Train: {Xtr.shape} | Test: {Xte.shape}")
    _, metrics = train_model(Xtr, ytr, Xte, yte, P)
    print(f"\n🎯 ROC-AUC @0.5: {metrics['default']['roc_auc']:.4f} | "
          f"@best: {metrics['best']['roc_auc']:.4f} (thr={metrics['thr']:.3f})")

if __name__ == "__main__":
    main()
