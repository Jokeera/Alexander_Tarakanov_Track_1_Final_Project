# PD-Model — Credit Card Default Prediction

Automated ML Pipeline using DVC · MLflow · FastAPI · Docker · Monitoring

**Author:** Alexander Tarakanov
**Goal:** Automate the end-to-end pipeline for predicting credit-card default using open-source MLOps tools.
**Dataset:** *Default of Credit Card Clients* (UCI Machine Learning Repository, 2005, Taiwan)

---

## Project Overview

This project implements a reproducible, modular and production-ready credit-scoring pipeline.
End-to-end ML lifecycle automation: data preparation → model training → experiment tracking → API service → drift monitoring.

---

## FastAPI Scoring Service

```bash
uvicorn src.api.app:app --reload --port 8000
```

**Endpoints:**

* Swagger UI → [http://localhost:8000/docs](http://localhost:8000/docs)
* Health check → [http://localhost:8000/health](http://localhost:8000/health)
* Predict → `POST /predict`

**Example request:**

```bash
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
```

**Example response:**

```json
{
  "predicted_class": 0,
  "predicted_proba": 0.29,
  "risk_level": "Low"
}
```

---

## Technology Stack

| Area                | Tools                   |
| ------------------- | ----------------------- |
| Data processing     | Pandas, Scikit-Learn    |
| Pipeline automation | DVC                     |
| Experiment tracking | MLflow                  |
| API service         | FastAPI + Uvicorn       |
| Containerization    | Docker                  |
| Monitoring          | PSI Drift Detection     |
| Testing             | Pytest + GitHub Actions |

---

## Repository Structure

```
project
├── data/                # Raw & processed data (DVC-tracked)
├── models/              # Trained model, metrics, drift reports
├── notebooks/           # EDA and exploration
├── src/
│   ├── data/            # Data preparation & validation
│   ├── features/        # Feature engineering
│   ├── models/          # Pipeline, training, prediction
│   └── api/             # FastAPI scoring service
├── tests/               # Unit tests for modules
├── params.yaml          # Configuration for entire pipeline
├── dvc.yaml             # ML pipeline definition
├── Dockerfile
└── README.md
```

---

## Installation

```bash
conda create -n pdmodel-310 python=3.10 -y
conda activate pdmodel-310
pip install -r requirements.txt
```

---

## Run Full Pipeline

```bash
dvc pull
dvc repro
```

Generated:

* `models/model.pkl`
* `models/metrics.json`
* ROC Curve / Confusion Matrix
* `models/drift_report.json`

---

## Experiment Tracking (MLflow)

```bash
mlflow ui --port 5000
```

→ Open [http://localhost:5000](http://localhost:5000)

---

## Unit Tests

```bash
pytest -q
```

---

## Docker Deployment

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

---

## Drift Monitoring (PSI)

```bash
python -m src.monitoring.drift
```

Output: `models/drift_report.json`

---

## Evaluation (Test Set Example)

| Metric       |                Value |
| ------------ | -------------------: |
| ROC-AUC      |                ~0.77 |
| F1-score     |           ~0.45–0.50 |
| Drift status | No significant drift |

---

## Key Insights

* Class imbalance (~22 % defaults) → ROC-AUC and PR-AUC as primary metrics
* Behavioural features (`pay_*`, delays, utilization) contribute most
* Monetary features require scaling and aggregate/trend features
* Fully automated and reproducible ML workflow

