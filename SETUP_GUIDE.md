Отлично. Ниже — финальная профессиональная версия `SETUP_GUIDE.md` без эмодзи, в корпоративно-техническом стиле, принятом в документации MLOps-проектов.
Её можно оставить как есть в репозитории — она читается чётко, формально и подходит для защиты или публикации.

---

# Setup Guide — PD-Model (Credit Card Default Prediction)

This document provides step-by-step instructions for setting up, running, and verifying the full ML pipeline for the credit-card default prediction model.
The project includes automated stages for data preparation, model training, experiment tracking, API deployment, and drift monitoring.

---

## 1. Environment Setup

### Option A — Conda (recommended)

```bash
conda create -n pdmodel-310 python=3.10 -y
conda activate pdmodel-310
pip install -r requirements.txt
```

### Option B — Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Verify installation:

```bash
python --version
dvc --version
mlflow --version
```

---

## 2. Project Initialization

```bash
git clone <your_repository_url>
cd Alexander_Tarakanov_Track_1_Final_Project
```

If DVC remote storage is configured:

```bash
dvc pull
```

---

## 3. Running the Full ML Pipeline

Execute the entire automated workflow:

```bash
dvc repro
```

Pipeline stages:

1. **prepare** — data loading, cleaning, and train/test split
2. **features** — feature engineering and dataset enrichment
3. **train** — model training with hyperparameter tuning and MLflow logging
4. **drift** — PSI drift monitoring and report generation

Generated artifacts:

```
models/model.pkl
models/metrics.json
models/drift_report.json
```

---

## 4. Experiment Tracking (MLflow)

Launch the local MLflow interface:

```bash
mlflow ui --port 5000
```

Access via browser: [http://localhost:5000](http://localhost:5000)

---

## 5. FastAPI Scoring Service

Run the scoring API locally:

```bash
uvicorn src.api.app:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Example prediction request:

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

---

## 6. Docker Deployment

Build and run the containerized version:

```bash
docker build -t credit-scoring-api .
docker run -d -p 8000:8000 credit-scoring-api
```

Verify container health:

```bash
curl http://localhost:8000/health
```

---

## 7. Drift Monitoring

Run the PSI drift analysis manually:

```bash
python -m src.monitoring.drift
```

Output file:

```
models/drift_report.json
```

If `psi < 0.25`, the drift is considered insignificant.

---

## 8. Unit Tests

Execute basic integrity and functionality tests:

```bash
pytest -q
```

---

## 9. Verification Checklist

| Component           | Description                                | Status |
| ------------------- | ------------------------------------------ | :----: |
| Data preparation    | DVC stage `prepare` completes successfully |    ✓   |
| Feature engineering | Derived features saved correctly           |    ✓   |
| Model training      | Model and metrics generated                |    ✓   |
| MLflow tracking     | Experiment visible in UI                   |    ✓   |
| API service         | Returns valid predictions                  |    ✓   |
| Docker container    | Passes health check                        |    ✓   |
| Drift monitoring    | Report generated successfully              |    ✓   |

---

## 10. References

* [DVC Documentation](https://dvc.org/doc)
* [MLflow Documentation](https://mlflow.org/)
* [FastAPI Documentation](https://fastapi.tiangolo.com/)
* [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

Хочешь, я сразу добавлю к этому файлу минимальный `verify_repo.sh` (Bash-скрипт), который автоматизирует проверку шагов из этого гайда — чтобы одним запуском проверять, что пайплайн, модель и API работают корректно?
