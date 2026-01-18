from fastapi import FastAPI
from pydantic import BaseModel,Field
from typing import Optional
import pandas as pd
import joblib
from pathlib import Path
import sys


# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="Claim Denial Risk API",
    description="Real-time claim denial risk prediction using trained ML pipeline",
    version="1.0"
)

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

# Make src importable so pickle can find ClaimFeatureEngineer
sys.path.append(str(SRC_PATH))

MODEL_PATH = PROJECT_ROOT / "models" / "claim_denial_model.pkl"

# Pipeline placeholder (loaded at startup)
pipeline = None


# -----------------------------
# Load model on startup
# -----------------------------
@app.on_event("startup")
def load_model():
    """
    Load trained ML pipeline once when API starts.
    Prevents OpenAPI schema errors and avoids reloading per request.
    """
    global pipeline
    pipeline = joblib.load(MODEL_PATH)


# -----------------------------
# Input schema (raw claim data)
# -----------------------------
class ClaimRequest(BaseModel):
    billed_amount: float
    length_of_stay: Optional[float] = Field(default=None)
    age: int
    insurance_type: str
    visit_type: str
    department_x: str
    admission_type: Optional[str] = Field(default=None)
    diagnosis_code: Optional[str] = Field(default=None)
    years_experience: Optional[int] = Field(default=None)
    readmitted_flag: Optional[str] = Field(default="No")

    


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_claim_denial(claim: ClaimRequest):
    """
    Predicts claim denial risk for a single incoming claim.
    """

    # Convert request to DataFrame (pipeline expects tabular input)
    df = pd.DataFrame([claim.dict()])

    # Full pipeline execution:
    # feature engineering -> preprocessing -> model
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]

    return {
        "is_denied_prediction": int(prediction),
        "denial_risk_probability": round(float(probability), 4)
    }


# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def health_check():
    return {"status": "Claim Denial Risk API is running"}