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
import re
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


class ChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, description="User's chat message")


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Chat guardrail & LLM integration
# ---------------------------------------------------------------------------

CHAT_SYSTEM_PROMPT = (
    "You are an AI Financial Assistant restricted to the Loan Eligibility Advisor platform. "
    "You may only discuss loan parameters, DTI, credit scores, and the platform's predictive outputs. "
    "If the user asks about coding, trivia, politics, or general topics, you must hard-reject the query "
    "with the exact phrase: 'My parameters restrict me from discussing topics outside of loan eligibility "
    "and financial profiling.' Maintain a professional, corporate tone."
)

_FINANCIAL_KEYWORDS = re.compile(
    r"\b(loan|credit(?: score)?|dti|debt.to.income|income|interest rate|"
    r"mortgage|borrow|lend|repayment|amortization|apr|fico|eligibilit|"
    r"approv|reject|risk|collateral|default|principal|financial|bank|"
    r"fund|invest|asset|liabilit|balance|budget|afford|payment|instalment|"
    r"installment|rate|term|tenure|refinanc|consolidat)\b",
    re.IGNORECASE,
)

_GUARDRAIL_REPLY = (
    "My parameters restrict me from discussing topics outside of "
    "loan eligibility and financial profiling."
)


def _mock_llm_response(user_message: str) -> str:
    """
    Fallback used when no LLM API key is configured.
    Demonstrates the guardrail by checking for financial keywords.
    """
    if _FINANCIAL_KEYWORDS.search(user_message):
        # Provide a helpful, on-topic mock response
        lower = user_message.lower()
        if "credit score" in lower or "fico" in lower:
            return (
                "A credit score (FICO) ranges from 300 to 850. "
                "Scores above 670 are generally considered good, and scores above 740 "
                "are considered very good. Higher scores improve your loan approval odds "
                "and typically result in lower interest rates."
            )
        if "dti" in lower or "debt-to-income" in lower or "debt to income" in lower:
            return (
                "The Debt-to-Income (DTI) ratio is calculated as your total monthly "
                "debt payments divided by your gross monthly income. "
                "Most lenders prefer a DTI below 43%. A lower DTI signals "
                "stronger repayment capacity and improves eligibility."
            )
        if "approv" in lower or "eligib" in lower:
            return (
                "Loan approval depends on several factors including your credit score, "
                "DTI ratio, annual income, employment history, and requested loan amount. "
                "Use the Loan Eligibility Advisor form above for a personalised assessment."
            )
        return (
            "I can assist you with questions about loan parameters, DTI, credit scores, "
            "and the platform's eligibility outputs. Please ask a specific financial question."
        )
    return _GUARDRAIL_REPLY


async def _get_llm_response(user_message: str) -> str:
    """
    Attempt to call a real LLM. Tries OpenAI first, then Gemini.
    Falls back to the mock function if no API key is present.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if openai_key:
        try:
            from openai import AsyncOpenAI  # type: ignore
            client = AsyncOpenAI(api_key=openai_key)
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=512,
                temperature=0.3,
            )
            return completion.choices[0].message.content or _GUARDRAIL_REPLY
        except Exception:
            pass  # fall through to next provider

    if gemini_key:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=CHAT_SYSTEM_PROMPT,
            )
            result = model.generate_content(user_message)
            return result.text or _GUARDRAIL_REPLY
        except Exception:
            pass  # fall through to mock

    return _mock_llm_response(user_message)


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


@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="Financial assistant chatbot",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Accepts a user message and returns a domain-restricted AI response.
    A guardrail system prompt ensures the assistant only discusses
    loan eligibility, DTI, credit scores, and related financial topics.
    """
    reply = await _get_llm_response(request.user_message)
    return ChatResponse(response=reply)
