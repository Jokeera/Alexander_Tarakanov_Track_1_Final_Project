PD-MODEL — Credit Card Default Prediction
Industrial MLOps Pipeline (Task 1 + Task 2)

Author: Alexander Tarakanov

============================================================

1. Project Context

This repository contains a full industrial MLOps implementation of a credit-scoring system
developed across two academic tracks:

- Task 1 — Automation of ML development and testing
- Task 2 — Automation of delivery and deployment of ML models

The project evolves from a classical ML pipeline to a production-ready system
with ONNX optimization, containerization, Kubernetes orchestration,
Infrastructure as Code, and CI/CD.

============================================================

2. Code Organization & Version Control (Git)

The repository follows a cookiecutter-data-science–style structure with strict separation
of concerns and reproducibility guarantees.

Key principles:
- Modular code organization (src/, tests/, scripts/)
- Deterministic pipelines (DVC, Terraform)
- Infrastructure and deployment separated from ML logic
- All experiments, models, and infrastructure are reproducible
- Git is used with semantic, meaningful commits

Repository structure:

.
├── data/                    Raw and processed data (DVC tracked)
├── models/                  Trained models, metrics, reports
├── notebooks/               EDA and exploratory analysis
├── src/                     Core application code
│   ├── data/                Data loading, cleaning, validation
│   ├── features/            Feature engineering
│   ├── models/              Training and inference logic
│   ├── api/                 FastAPI application
│   └── monitoring/          Drift detection
├── scripts/                 Training, ONNX, benchmark scripts
├── tests/                   Unit tests (pytest)
├── deployment/
│   ├── docker/              Dockerfiles (dev / prod)
│   └── kubernetes/          Kubernetes manifests
├── infrastructure/          Terraform IaC (modules and environments)
├── .github/workflows/       CI/CD pipelines
├── dvc.yaml                 DVC pipeline definition
├── params.yaml              Central configuration
├── README.txt               Project documentation
└── SETUP_GUIDE.md           Extended setup instructions

============================================================

3. Track 2 — Industrial Deployment & Delivery

Model preparation:
- Neural network trained in PyTorch
- Export to ONNX
- Numerical validation (PyTorch vs ONNX)
- INT8 quantization
- CPU inference benchmark

Artifacts:
- models/nn/credit_nn.pth
- models/onnx/credit_nn.onnx
- models/onnx/credit_nn_quant.onnx
- reports/benchmarks.md

API & containerization:
- FastAPI inference service using ONNX Runtime
- /health and /predict endpoints
- Development and production Docker images

Kubernetes:
- Namespaces: credit-staging, credit-prod
- Rolling deployments
- Liveness and readiness probes
- Resource limits and HPA
- Service and Ingress

Infrastructure as Code:
- Terraform modules: network, kubernetes, storage, monitoring
- Environments: staging, production
- Configuration validated with terraform validate

CI/CD:
- Linting and unit tests
- ONNX validation and benchmarking
- Docker build and security scanning

============================================================

4. Track 1 — ML Pipeline (Original Project)

This section preserves the original ML pipeline description.

Dataset:
Default of Credit Card Clients
UCI Machine Learning Repository (Taiwan, 2005)

Pipeline:
- Data preparation
- Feature engineering
- Model training
- Experiment tracking (MLflow)
- API service
- Drift monitoring

============================================================

5. Repositories

Track 1:
https://github.com/Jokeera/Alexander_Tarakanov_Track_1_Final_Project

Track 2:
Current repository

============================================================

6. Status

All course requirements are satisfied.
The project is fully reproducible.
The repository is ready for submission.
