"""
main.py
-------
Phase 2: FastAPI Backend

Loads the serialized RandomForest model and StandardScaler on startup,
exposes a /api/v1/predict POST endpoint, and returns a deterministic
loan-eligibility decision.

Run locally:
    uvicorn main:app --reload --port 8000

The Dockerfile in this directory containerizes the service.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    annual_income: float = Field(..., gt=0, description="Gross annual income in USD")
    credit_score: float = Field(..., ge=300, le=850, description="FICO credit score")
    debt_to_income_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Total monthly debt / gross monthly income"
    )
    employment_length_years: float = Field(
        ..., ge=0, description="Years in current employment"
    )
    loan_amount_requested: float = Field(
        ..., gt=0, description="Requested loan amount in USD"
    )

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "annual_income": 75000,
                "credit_score": 720,
                "debt_to_income_ratio": 0.25,
                "employment_length_years": 5,
                "loan_amount_requested": 15000,
            }
        },
    }


class PredictionResponse(BaseModel):
    decision: str
    probability_approved: float
    risk_factors: list[str]


# ---------------------------------------------------------------------------
# App state – loaded once at startup
# ---------------------------------------------------------------------------

class _ModelStore:
    model: Any = None
    scaler: Any = None
    feature_columns: list[str] = []
    positive_label: Any = None


store = _ModelStore()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts into memory once at startup."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Model files not found. Run `python train_model.py` first."
        )
    artifacts = joblib.load(MODEL_PATH)
    if isinstance(artifacts, dict) and "model" in artifacts:
        store.model = artifacts["model"]
        store.scaler = artifacts.get("scaler")
        store.feature_columns = list(artifacts.get("feature_columns", []))
        store.positive_label = artifacts.get("positive_label")
    else:
        store.model = artifacts
        store.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        store.feature_columns = []
        store.positive_label = None
    yield
    # Cleanup (none required)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Loan Eligibility Advisor API",
    version="1.0.0",
    description="Real-time loan eligibility prediction powered by a RandomForest model.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helper: derive risk factors from the application
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "Low credit score (< 620)": lambda a: a.credit_score < 620,
    "High debt-to-income ratio (> 40%)": lambda a: a.debt_to_income_ratio > 0.40,
    "Short employment history (< 2 years)": lambda a: a.employment_length_years < 2,
    "High loan-to-income ratio (> 40%)": lambda a: (
        a.loan_amount_requested / a.annual_income
    ) > 0.40,
    "Low annual income (< $30,000)": lambda a: a.annual_income < 30_000,
}


def _derive_risk_factors(application: LoanApplication) -> list[str]:
    return [label for label, check in RISK_THRESHOLDS.items() if check(application)]


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

APPROVAL_THRESHOLD = 0.60
CONDITIONAL_THRESHOLD = 0.40


def _make_decision(prob: float, risk_factors: list[str]) -> str:
    if prob >= APPROVAL_THRESHOLD and len(risk_factors) == 0:
        return "Approved"
    if prob >= CONDITIONAL_THRESHOLD:
        return "Conditional"
    return "Rejected"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok"}


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict loan eligibility",
)
def predict(application: LoanApplication) -> PredictionResponse:
    """
    Accepts a loan application payload, scales the features using the fitted
    StandardScaler, runs inference with the RandomForest model, and returns
    an eligibility decision along with identified risk factors.
    """
    if store.model is None or store.scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    application_payload = application.model_dump()
    feature_frame = pd.DataFrame([application_payload])
    encoded_frame = pd.get_dummies(feature_frame)

    if store.feature_columns:
        encoded_frame = encoded_frame.reindex(columns=store.feature_columns, fill_value=0)

    feature_matrix = encoded_frame.to_numpy(dtype=float)
    scaled = store.scaler.transform(feature_matrix)

    class_probabilities = store.model.predict_proba(scaled)[0]
    positive_index = 1
    if store.positive_label is not None and hasattr(store.model, "classes_"):
        matches = np.where(store.model.classes_ == store.positive_label)[0]
        if len(matches) == 1:
            positive_index = int(matches[0])

    prob_approved = float(class_probabilities[positive_index])
    risk_factors = _derive_risk_factors(application)
    decision = _make_decision(prob_approved, risk_factors)

    return PredictionResponse(
        decision=decision,
        probability_approved=round(prob_approved, 4),
        risk_factors=risk_factors,
    )
