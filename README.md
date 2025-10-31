PD-MODEL — CREDIT CARD DEFAULT PREDICTION
──────────────────────────────────────────────────────────────
Automated ML Pipeline using DVC · MLflow · FastAPI · Docker · Monitoring

Author: Alexander Tarakanov
Goal: Automate the end-to-end ML pipeline for predicting credit-card default using open-source MLOps tools.
Dataset: Default of Credit Card Clients (UCI Machine Learning Repository, Taiwan, 2005)
──────────────────────────────────────────────────────────────


PROJECT OVERVIEW
──────────────────────────────────────────────────────────────
This project implements a reproducible, modular and production-ready credit-scoring pipeline.
It automates the full ML lifecycle:
data preparation → feature engineering → model training → experiment tracking → API service → drift monitoring.


TECHNOLOGY STACK
──────────────────────────────────────────────────────────────
Data processing     : Pandas, Scikit-Learn
Pipeline automation : DVC
Experiment tracking : MLflow
API service         : FastAPI + Uvicorn
Containerization    : Docker
Monitoring          : PSI Drift Detection
Testing / CI        : Pytest + GitHub Actions


PROJECT STRUCTURE
──────────────────────────────────────────────────────────────
project
├── data/                → Raw and processed data (tracked with DVC)
├── models/              → Trained model, metrics, ROC/PR curves, drift reports
├── notebooks/           → EDA and exploratory analysis
├── src/
│   ├── data/            → Data loading, cleaning, validation
│   ├── features/        → Feature engineering scripts
│   ├── models/          → Training pipeline and inference logic
│   └── api/             → FastAPI scoring service
├── tests/               → Unit tests (pytest)
├── doc/
│   └── screenshots/     → Evidence of work execution:
│       • DVC pipeline run
│       • MLflow UI experiments
│       • Drift report generation
│       • FastAPI /health and /predict endpoints
│       • Docker container build and run
│       • CI test and validation results
├── params.yaml          → Central configuration
├── dvc.yaml             → Pipeline definition (prepare → features → train → drift)
├── Dockerfile           → Containerized API
└── README.md            → Project description and setup guide
──────────────────────────────────────────────────────────────


INSTALLATION
──────────────────────────────────────────────────────────────
conda create -n pdmodel-310 python=3.10 -y
conda activate pdmodel-310
pip install -r requirements.txt

Verify installation:
python --version
dvc --version
mlflow --version


RUNNING THE FULL PIPELINE
──────────────────────────────────────────────────────────────
dvc pull        # optional if remote is configured
dvc repro       # runs prepare → features → train → drift

Artifacts generated:
• models/model.pkl
• models/metrics.json
• models/roc_curve.png, models/pr_curve.png
• models/drift_report.json


EXPERIMENT TRACKING (MLFLOW)
──────────────────────────────────────────────────────────────
mlflow ui --port 5000
Open in browser → http://localhost:5000

Each experiment logs:
• parameters and hyperparameters
• metrics (ROC-AUC, PR-AUC, F1, Precision, Recall)
• plots and trained model artifact


FASTAPI SCORING SERVICE
──────────────────────────────────────────────────────────────
uvicorn src.api.app:app --reload --port 8000

Endpoints:
• /health   → API status
• /predict  → JSON input → default probability output
• /docs     → Swagger UI

Example request:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "limit_bal": 50000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 30,
    "pay_0": 0, "pay_2": 0, "pay_3": 0, "pay_4": 0, "pay_5": 0, "pay_6": 0,
    "bill_amt1": 3913, "bill_amt2": 3102, "bill_amt3": 689,
    "bill_amt4": 0, "bill_amt5": 0, "bill_amt6": 0,
    "pay_amt1": 0, "pay_amt2": 689, "pay_amt3": 0,
    "pay_amt4": 0, "pay_amt5": 0, "pay_amt6": 0
  }'

Example response:
{
  "predicted_class": 0,
  "predicted_proba": 0.29,
  "risk_level": "Low"
}


DOCKER DEPLOYMENT
──────────────────────────────────────────────────────────────
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api

Health check:
curl http://localhost:8000/health


DRIFT MONITORING (PSI)
──────────────────────────────────────────────────────────────
python -m src.monitoring.drift
Output: models/drift_report.json
If PSI < 0.25 → no significant drift detected.


UNIT TESTS AND CI
──────────────────────────────────────────────────────────────
pytest -q
All tests run automatically in GitHub Actions (including Black, Flake8, and Great Expectations validation).


EVALUATION (TEST SET)
──────────────────────────────────────────────────────────────
ROC-AUC      ≈ 0.77–0.78
PR-AUC       ≈ 0.55
F1-score     ≈ 0.50
Drift status : No significant drift


KEY INSIGHTS
──────────────────────────────────────────────────────────────
• Data imbalance (~22% defaults) handled via class_weight=balanced
• Behavioural features (pay_*, delays, utilization) are most predictive
• MLflow ensures reproducible experiment tracking
• DVC enables deterministic pipeline execution (dvc repro)
• FastAPI + Docker provide deployable inference service
• PSI monitoring detects potential concept drift


VALIDATION CHECKLIST
──────────────────────────────────────────────────────────────
Component             | Validation Scope
──────────────────────────────────────────────────────────────
Data preparation      : Clean train/test split and schema validation (Pandera / GE)
Feature engineering   : Aggregate, trend, and ratio features correctly generated
Model training        : End-to-end Sklearn pipeline with GridSearchCV tuning
MLflow tracking       : Parameters, metrics, and 5+ experiments logged successfully
DVC pipeline          : Full pipeline reproducibility (prepare → features → train → drift)
Testing & CI          : Automated pytest suite + GitHub Actions (Black, Flake8, GE)
FastAPI & Docker      : Working /health and /predict endpoints inside container
Drift monitoring      : PSI < 0.25 — no significant data drift detected
──────────────────────────────────────────────────────────────
All components passed validation successfully.


REPOSITORY
──────────────────────────────────────────────────────────────
https://github.com/Jokeera/Alexander_Tarakanov_Track_1_Final_Project
──────────────────────────────────────────────────────────────
