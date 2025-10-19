"""FastAPI application for credit scoring model"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="API for predicting credit card default probability",
    version="1.0.0"
)


# Load model at startup
MODEL_PATH = "models/model.pkl"
model = None


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    if not Path(MODEL_PATH).exists():
        print(f"Warning: Model not found at {MODEL_PATH}")
        model = None
    else:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")


# Request schema
class CreditFeatures(BaseModel):
    """Input features for credit scoring"""
    limit_bal: float = Field(..., description="Credit limit", ge=0)
    sex: int = Field(..., description="Gender (1=male, 2=female)", ge=1, le=2)
    education: int = Field(..., description="Education level (1-4)", ge=1, le=4)
    marriage: int = Field(..., description="Marital status (1-3)", ge=1, le=3)
    age: int = Field(..., description="Age in years", ge=18, le=100)
    
    pay_0: int = Field(..., description="Repayment status Sept", ge=-2, le=9)
    pay_2: int = Field(..., description="Repayment status Aug", ge=-2, le=9)
    pay_3: int = Field(..., description="Repayment status July", ge=-2, le=9)
    pay_4: int = Field(..., description="Repayment status June", ge=-2, le=9)
    pay_5: int = Field(..., description="Repayment status May", ge=-2, le=9)
    pay_6: int = Field(..., description="Repayment status April", ge=-2, le=9)
    
    bill_amt1: float = Field(..., description="Bill amount Sept")
    bill_amt2: float = Field(..., description="Bill amount Aug")
    bill_amt3: float = Field(..., description="Bill amount July")
    bill_amt4: float = Field(..., description="Bill amount June")
    bill_amt5: float = Field(..., description="Bill amount May")
    bill_amt6: float = Field(..., description="Bill amount April")
    
    pay_amt1: float = Field(..., description="Payment amount Sept", ge=0)
    pay_amt2: float = Field(..., description="Payment amount Aug", ge=0)
    pay_amt3: float = Field(..., description="Payment amount July", ge=0)
    pay_amt4: float = Field(..., description="Payment amount June", ge=0)
    pay_amt5: float = Field(..., description="Payment amount May", ge=0)
    pay_amt6: float = Field(..., description="Payment amount April", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "limit_bal": 20000,
                "sex": 2,
                "education": 2,
                "marriage": 1,
                "age": 24,
                "pay_0": 2,
                "pay_2": 2,
                "pay_3": -1,
                "pay_4": -1,
                "pay_5": -2,
                "pay_6": -2,
                "bill_amt1": 3913,
                "bill_amt2": 3102,
                "bill_amt3": 689,
                "bill_amt4": 0,
                "bill_amt5": 0,
                "bill_amt6": 0,
                "pay_amt1": 0,
                "pay_amt2": 689,
                "pay_amt3": 0,
                "pay_amt4": 0,
                "pay_amt5": 0,
                "pay_amt6": 0
            }
        }


# Response schema
class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_class: int = Field(..., description="Predicted class (0 or 1)")
    predicted_proba: float = Field(..., description="Probability of default")
    risk_level: str = Field(..., description="Risk level (Low/Medium/High)")


def create_features(input_data: CreditFeatures) -> pd.DataFrame:
    """Create engineered features from input"""
    # Convert to dict
    data = input_data.dict()
    
    # Age binning
    age = data["age"]
    if age < 30:
        age_bin = "young"
    elif age < 45:
        age_bin = "middle"
    elif age < 60:
        age_bin = "senior"
    else:
        age_bin = "elderly"
    data["age_bin"] = age_bin
    
    # Utilization rate
    utilization = data["bill_amt1"] / data["limit_bal"] if data["limit_bal"] > 0 else 0
    data["utilization_last"] = np.clip(utilization, 0, 2)
    
    # Payment delay features
    pay_cols = [data["pay_0"], data["pay_2"], data["pay_3"], data["pay_4"], data["pay_5"], data["pay_6"]]
    data["pay_delay_sum"] = sum(1 for p in pay_cols if p > 0)
    data["pay_delay_max"] = max(pay_cols)
    
    # Bill trend
    bill_diff = data["bill_amt1"] - data["bill_amt3"]
    bill_base = abs(data["bill_amt3"]) + 1
    data["bill_trend"] = np.clip(bill_diff / bill_base, -5, 5)
    
    # Payment trend
    pay_diff = data["pay_amt1"] - data["pay_amt3"]
    pay_base = data["pay_amt3"] + 1
    data["pay_trend"] = np.clip(pay_diff / pay_base, -5, 5)
    
    # Average bill and payment
    bill_cols = [data[f"bill_amt{i}"] for i in range(1, 7)]
    pay_amt_cols = [data[f"pay_amt{i}"] for i in range(1, 7)]
    data["bill_avg"] = np.mean(bill_cols)
    data["pay_amt_avg"] = np.mean(pay_amt_cols)
    
    # Payment to bill ratio
    pay_to_bill = data["pay_amt_avg"] / data["bill_avg"] if data["bill_avg"] > 0 else 0
    data["pay_to_bill_ratio"] = np.clip(pay_to_bill, 0, 5)
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    return df


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Credit Scoring API",
        "version": "1.0.0",
        "status": "healthy" if model is not None else "model not loaded"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CreditFeatures):
    """Make prediction endpoint"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create features
        features_df = create_features(features)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0, 1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return PredictionResponse(
            predicted_class=int(prediction),
            predicted_proba=float(probability),
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)