"""
FastAPI scoring service
-> получает raw признаки (как в train.csv)
-> pipeline сам делает feature engineering и preprocessing
"""

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ------------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------------

MODEL_PATH = "models/model.pkl"
model = None

app = FastAPI(
    title="Credit Scoring API",
    description="Predict probability of default using trained ML model",
    version="1.0.0",
)


@app.on_event("startup")
async def load_model():
    """Load trained pipeline on startup"""
    global model
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded:", MODEL_PATH)
    else:
        print("⚠️ Model file not found:", MODEL_PATH)


# ------------------------------------------------------------------------------------
# REQUEST SCHEMA
# (Raw features ONLY — same columns as in train_raw.csv)
# ------------------------------------------------------------------------------------

class CreditRaw(BaseModel):
    limit_bal: float
    sex: int = Field(..., ge=1, le=2)
    education: int = Field(..., ge=1, le=4)
    marriage: int = Field(..., ge=1, le=3)
    age: int = Field(..., ge=18, le=100)

    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int

    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float

    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


# ------------------------------------------------------------------------------------
# RESPONSE SCHEMA
# ------------------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_proba: float
    risk_level: str


# ------------------------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "Credit Scoring API", "model_loaded": model is not None}


@app.get("/health")
async def health():
    return {"status": "ok" if model else "model_not_loaded"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: CreditRaw):
    """Infer with trained sklearn Pipeline"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([payload.dict()])  # raw → DataFrame

        # full preprocess+features inside pipeline
        proba = model.predict_proba(df)[0, 1]
        pred = model.predict(df)[0]

        risk = "Low" if proba < 0.3 else "Medium" if proba < 0.6 else "High"

        return PredictionResponse(
            predicted_class=int(pred),
            predicted_proba=float(proba),
            risk_level=risk,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
