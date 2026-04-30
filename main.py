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
from typing import Any, Optional

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class LoanApplication(BaseModel):
    person_age: int = Field(..., ge=18, description="Applicant age")
    person_income: float = Field(..., gt=0, description="Annual income in USD")
    person_home_ownership: str = Field(..., description="OWN / MORTGAGE / RENT / OTHER")
    person_emp_length: float = Field(..., ge=0, description="Years in current employment")
    loan_intent: str = Field(..., description="Loan purpose")
    loan_grade: str = Field(..., description="Loan grade (A..G)")
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    loan_int_rate: float = Field(..., ge=0, description="Interest rate (percent)")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Loan amount / income as fraction")
    cb_person_default_on_file: str = Field(..., description="Y or N")
    cb_person_cred_hist_length: float = Field(..., ge=0, description="Credit history length (years)")

    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "person_age": 35,
                "person_income": 59000,
                "person_home_ownership": "RENT",
                "person_emp_length": 3.0,
                "loan_intent": "PERSONAL",
                "loan_grade": "C",
                "loan_amnt": 35000,
                "loan_int_rate": 15.23,
                "loan_percent_income": 0.59,
                "cb_person_default_on_file": "N",
                "cb_person_cred_hist_length": 3,
            }
        },
    }


class PredictionResponse(BaseModel):
    decision: str
    probability_approved: float
    risk_factors: list[str]


class ChatMessage(BaseModel):
    user_message: str = Field(..., min_length=1, description="User's chat message")
    application: Optional[dict] = Field(None, description="Optional loan application context")
    prediction: Optional[PredictionResponse] = Field(
        None,
        description="Optional model prediction context",
    )


class ChatResponse(BaseModel):
    response: str


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

# Gemini model instance – initialised once at startup if the key is present
_gemini_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artefacts and optional LLM client into memory once at startup."""
    global _gemini_model
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

    # Initialise Gemini model once if the API key is available
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and _GENAI_AVAILABLE:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_SYSTEM_PROMPT,
        )

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
    "High loan % of income (> 40%)": lambda a: a.loan_percent_income > 0.40,
    "Short employment history (< 2 years)": lambda a: a.person_emp_length < 2,
    "High interest rate (> 18%)": lambda a: a.loan_int_rate > 18.0,
    "High loan-to-income ratio (> 40%)": lambda a: (a.loan_amnt / (a.person_income or 1)) > 0.40,
    "Low annual income (< $30,000)": lambda a: a.person_income < 30_000,
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


def _format_application_summary(application: LoanApplication) -> str:
    return (
        f"Applicant age: {application.person_age}; Income: ${application.person_income}; "
        f"Home ownership: {application.person_home_ownership}; Employment years: {application.person_emp_length}; "
        f"Loan amount: ${application.loan_amnt}; Interest: {application.loan_int_rate}%; "
        f"Loan intent: {application.loan_intent}; Grade: {application.loan_grade}"
    )


# ---------------------------------------------------------------------------
# Chat – system prompt (guardrail)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an AI Financial Assistant restricted to the Loan Eligibility Advisor platform. "
    "You may only discuss loan parameters, DTI, credit scores, and the platform's predictive outputs. "
    "If the user asks about coding, trivia, politics, or general topics, you must hard-reject the query "
    "with the exact phrase: 'My parameters restrict me from discussing topics outside of loan eligibility "
    "and financial profiling.' Maintain a professional, corporate tone."
)

# Keywords that indicate the message is within the financial domain
_FINANCIAL_KEYWORDS = {
    "loan", "credit", "dti", "debt", "income", "score", "interest", "rate",
    "mortgage", "eligibility", "approval", "risk", "payment", "financial",
    "finance", "borrow", "lender", "amortize", "principal", "collateral",
    "fico", "equity", "refinance", "default", "apr",
}


def _mock_chat_response(user_message: str) -> str:
    """
    Fallback used when no LLM API key is configured.
    Demonstrates guardrail logic: off-topic queries are hard-rejected.
    """
    lowered = user_message.lower()
    is_financial = any(kw in lowered for kw in _FINANCIAL_KEYWORDS)
    if not is_financial:
        return (
            "My parameters restrict me from discussing topics outside of "
            "loan eligibility and financial profiling."
        )
    return (
        "Thank you for your financial query. "
        "Based on the Loan Eligibility Advisor platform, "
        "I can help you understand loan parameters such as credit score thresholds, "
        "debt-to-income (DTI) ratios, and their impact on loan approval decisions. "
        "Please provide specific details so I can offer a more tailored assessment."
    )


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


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Financial domain chatbot",
)
async def chat(message: ChatMessage) -> ChatResponse:
    """
    Accepts a user message and returns an AI-generated response restricted
    to the financial / loan-eligibility domain.

    If GEMINI_API_KEY is present in the environment the request is forwarded
    to Gemini; otherwise a mocked guardrail response is returned.
    """
    # Build a prompt that includes the application and prediction context when provided.
    prompt_lines = []
    app_obj = None
    if message.application:
        try:
            # Try to coerce into LoanApplication for consistent risk-factor derivation
            app_obj = LoanApplication(**message.application)
            prompt_lines.append(f"Application: {_format_application_summary(app_obj)}")
        except Exception:
            # If parsing fails, fall back to raw dict embedding
            prompt_lines.append(f"Application: {message.application}")

    if message.prediction:
        prompt_lines.append(
            "Model decision: "
            f"{message.prediction.decision}; "
            f"Approved probability: {message.prediction.probability_approved}; "
            f"Risk factors: {', '.join(message.prediction.risk_factors) if message.prediction.risk_factors else 'None'}"
        )

    prompt_lines.append(f"User: {message.user_message}")
    prompt_text = "\n".join(prompt_lines)

    if _gemini_model is not None:
        try:
            result = _gemini_model.generate_content(prompt_text)
            text = result.text if result.text else (
                "My parameters restrict me from discussing topics outside of "
                "loan eligibility and financial profiling."
            )
            return ChatResponse(response=text)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

    # Fallback mock: if an application was provided, include a short summary and derived risk factors
    if app_obj is not None:
        risk_factors = _derive_risk_factors(app_obj)
        rf_text = ", ".join(risk_factors) if risk_factors else "None identified"
        prediction_text = (
            f"Decision: {message.prediction.decision}; "
            f"Probability: {message.prediction.probability_approved}; "
            f"Prediction risk factors: {', '.join(message.prediction.risk_factors) if message.prediction else 'None'}"
            if message.prediction
            else "No model prediction supplied."
        )
        mock = (
            f"I reviewed the provided application. Summary: {_format_application_summary(app_obj)}. "
            f"Model context: {prediction_text} "
            f"Application risk factors: {rf_text}. "
            "In plain terms: focus on reducing DTI and interest burden; consider lowering requested amount or improving income/stability to improve approval chances. "
            "Ask me specific follow-ups about any of these points."
        )
        return ChatResponse(response=mock)

    return ChatResponse(response=_mock_chat_response(message.user_message))
