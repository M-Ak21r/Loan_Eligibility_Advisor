"""
main.py
-------
FastAPI backend for the Loan Eligibility Advisor.

Loads the serialized RandomForest artifacts on startup, exposes a
/api/v1/predict endpoint for deterministic loan-eligibility predictions, and
provides an optional domain-restricted chat endpoint.
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    import google.generativeai as genai

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

if load_dotenv is not None:
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
    loan_int_rate: float = Field(..., ge=0, description="Interest rate as a percentage")
    loan_percent_income: float = Field(
        ...,
        ge=0,
        le=1,
        description="Loan amount / income as a fraction",
    )
    cb_person_default_on_file: str = Field(..., description="Y or N")
    cb_person_cred_hist_length: float = Field(
        ...,
        ge=0,
        description="Credit history length in years",
    )

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
    application: Optional[dict[str, Any]] = Field(
        None,
        description="Optional loan application context",
    )
    prediction: Optional[PredictionResponse] = Field(
        None,
        description="Optional model prediction context",
    )


class ChatResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# App state loaded once at startup
# ---------------------------------------------------------------------------

class _ModelStore:
    model: Any = None
    scaler: Any = None
    feature_columns: list[str] = []
    approval_label: Any = None
    approval_profile: dict[str, dict[str, Any]] = {}


store = _ModelStore()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

_gemini_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts and optional LLM client into memory once."""
    global _gemini_model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model files not found. Run `python train_model.py` first.")

    artifacts = joblib.load(MODEL_PATH)
    if isinstance(artifacts, dict) and "model" in artifacts:
        store.model = artifacts["model"]
        store.scaler = artifacts.get("scaler")
        store.feature_columns = list(artifacts.get("feature_columns", []))
        store.approval_label = artifacts.get(
            "approval_label",
            artifacts.get("positive_label"),
        )
        store.approval_profile = artifacts.get("approval_profile", {})
    else:
        store.model = artifacts
        store.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        store.feature_columns = []
        store.approval_label = None
        store.approval_profile = {}

    if hasattr(store.model, "n_jobs"):
        store.model.n_jobs = 1

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and _GENAI_AVAILABLE:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_SYSTEM_PROMPT,
        )

    yield


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
# Model-driven probability and improvement suggestions
# ---------------------------------------------------------------------------

GRADE_ORDER = ("A", "B", "C", "D", "E", "F", "G")
MIN_SUGGESTION_LIFT = 0.01


def _score_application(application: LoanApplication) -> float:
    payload = application.model_dump()
    feature_frame = pd.DataFrame([payload])
    encoded_frame = pd.get_dummies(feature_frame)

    if store.feature_columns:
        encoded_frame = encoded_frame.reindex(columns=store.feature_columns, fill_value=0)

    encoded_frame = encoded_frame.astype(float)
    scaled = store.scaler.transform(encoded_frame)
    class_probabilities = store.model.predict_proba(scaled)[0]

    approval_index = 1
    if store.approval_label is not None and hasattr(store.model, "classes_"):
        matches = np.where(store.model.classes_ == store.approval_label)[0]
        if len(matches) == 1:
            approval_index = int(matches[0])

    return float(class_probabilities[approval_index])


def _format_currency(value: float) -> str:
    return f"${value:,.0f}"


def _make_application(payload: dict[str, Any]) -> LoanApplication:
    next_payload = payload.copy()
    next_payload["loan_percent_income"] = max(
        0,
        min(float(next_payload["loan_percent_income"]), 1),
    )
    return LoanApplication(**next_payload)


def _numeric_profile_value(column: str) -> Optional[float]:
    value = store.approval_profile.get("numeric_medians", {}).get(column)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _categorical_profile_value(column: str) -> Optional[str]:
    value = store.approval_profile.get("categorical_modes", {}).get(column)
    if value is None or pd.isna(value):
        return None
    return str(value)


def _grade_is_better(target_grade: str, current_grade: str) -> bool:
    target = target_grade.upper()
    current = current_grade.upper()
    if target not in GRADE_ORDER or current not in GRADE_ORDER:
        return False
    return GRADE_ORDER.index(target) < GRADE_ORDER.index(current)


def _candidate_improvements(application: LoanApplication) -> list[tuple[str, LoanApplication]]:
    base = application.model_dump()
    candidates: list[tuple[str, LoanApplication]] = []

    approved_loan_percent = _numeric_profile_value("loan_percent_income")
    if approved_loan_percent and application.loan_percent_income > approved_loan_percent:
        payload = base.copy()
        new_amount = max(1.0, application.person_income * approved_loan_percent)
        payload["loan_amnt"] = new_amount
        payload["loan_percent_income"] = approved_loan_percent
        candidates.append((
            "Lower the loan burden toward the approved-applicant profile "
            f"by requesting about {_format_currency(new_amount)}",
            _make_application(payload),
        ))

    approved_amount = _numeric_profile_value("loan_amnt")
    if approved_amount and application.loan_amnt > approved_amount:
        payload = base.copy()
        payload["loan_amnt"] = approved_amount
        payload["loan_percent_income"] = approved_amount / application.person_income
        candidates.append((
            f"Request a loan amount closer to {_format_currency(approved_amount)}",
            _make_application(payload),
        ))

    approved_interest = _numeric_profile_value("loan_int_rate")
    if approved_interest and application.loan_int_rate > approved_interest:
        payload = base.copy()
        payload["loan_int_rate"] = approved_interest
        candidates.append((
            f"Negotiate interest rate toward {approved_interest:.2f}%",
            _make_application(payload),
        ))

    approved_grade = _categorical_profile_value("loan_grade")
    if approved_grade and _grade_is_better(approved_grade, application.loan_grade):
        payload = base.copy()
        payload["loan_grade"] = approved_grade
        candidates.append((
            f"Improve credit profile enough to qualify for grade {approved_grade}",
            _make_application(payload),
        ))

    approved_emp_length = _numeric_profile_value("person_emp_length")
    if approved_emp_length and application.person_emp_length < approved_emp_length:
        payload = base.copy()
        payload["person_emp_length"] = approved_emp_length
        candidates.append((
            f"Show employment history closer to {approved_emp_length:.1f} years",
            _make_application(payload),
        ))

    approved_default_status = _categorical_profile_value("cb_person_default_on_file")
    if (
        approved_default_status
        and application.cb_person_default_on_file.upper() != approved_default_status.upper()
    ):
        payload = base.copy()
        payload["cb_person_default_on_file"] = approved_default_status
        candidates.append((
            "Resolve prior default history before applying",
            _make_application(payload),
        ))

    approved_income = _numeric_profile_value("person_income")
    if approved_income and application.person_income < approved_income:
        payload = base.copy()
        payload["person_income"] = approved_income
        payload["loan_percent_income"] = application.loan_amnt / approved_income
        candidates.append((
            f"Document income closer to {_format_currency(approved_income)} or add a co-applicant",
            _make_application(payload),
        ))

    combined_payload = base.copy()
    if approved_income and application.person_income < approved_income:
        combined_payload["person_income"] = approved_income
    if approved_loan_percent and application.loan_percent_income > approved_loan_percent:
        combined_payload["loan_amnt"] = max(
            1.0,
            combined_payload["person_income"] * approved_loan_percent,
        )
        combined_payload["loan_percent_income"] = approved_loan_percent
    if approved_interest and application.loan_int_rate > approved_interest:
        combined_payload["loan_int_rate"] = approved_interest
    if approved_grade and _grade_is_better(approved_grade, application.loan_grade):
        combined_payload["loan_grade"] = approved_grade
    if approved_emp_length and application.person_emp_length < approved_emp_length:
        combined_payload["person_emp_length"] = approved_emp_length
    if approved_default_status:
        combined_payload["cb_person_default_on_file"] = approved_default_status
    if combined_payload["person_income"] > 0:
        combined_payload["loan_percent_income"] = (
            combined_payload["loan_amnt"] / combined_payload["person_income"]
        )

    if combined_payload != base:
        candidates.append((
            "Apply the combined approved-profile changes before reapplying",
            _make_application(combined_payload),
        ))

    return candidates


def _derive_improvement_suggestions(
    application: LoanApplication,
    base_probability: float,
) -> list[str]:
    suggestions = []
    seen_labels = set()

    for label, candidate in _candidate_improvements(application):
        candidate_probability = _score_application(candidate)
        lift = candidate_probability - base_probability
        if lift < MIN_SUGGESTION_LIFT or label in seen_labels:
            continue
        seen_labels.add(label)
        suggestions.append((lift, label, candidate_probability))

    suggestions.sort(reverse=True, key=lambda item: item[0])
    return [
        f"{label} (estimated approval +{lift * 100:.1f} pts to {probability * 100:.1f}%)"
        for lift, label, probability in suggestions[:4]
    ]


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

APPROVAL_THRESHOLD = 0.60
CONDITIONAL_THRESHOLD = 0.40


def _make_decision(prob: float) -> str:
    if prob >= APPROVAL_THRESHOLD:
        return "Approved"
    if prob >= CONDITIONAL_THRESHOLD:
        return "Conditional"
    return "Rejected"


def _format_application_summary(application: LoanApplication) -> str:
    return (
        f"Applicant age: {application.person_age}; "
        f"Income: {_format_currency(application.person_income)}; "
        f"Home ownership: {application.person_home_ownership}; "
        f"Employment years: {application.person_emp_length}; "
        f"Loan amount: {_format_currency(application.loan_amnt)}; "
        f"Interest: {application.loan_int_rate:.2f}%; "
        f"Loan intent: {application.loan_intent}; "
        f"Grade: {application.loan_grade}"
    )


# ---------------------------------------------------------------------------
# Chat guardrail
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an AI Financial Assistant restricted to the Loan Eligibility Advisor platform. "
    "You may only discuss loan parameters, DTI, credit scores, and the platform's predictive outputs. "
    "When an application and model prediction are provided, treat them as the authoritative context "
    "for the answer. Reference the applicant's specific values, decision, approval probability, and "
    "model-tested improvement suggestions instead of giving generic advice. "
    "If the user asks about coding, trivia, politics, or general topics, you must hard-reject the query "
    "with the exact phrase: 'My parameters restrict me from discussing topics outside of loan eligibility "
    "and financial profiling.' Maintain a professional, corporate tone."
)

_FINANCIAL_KEYWORDS = {
    "loan",
    "credit",
    "dti",
    "debt",
    "income",
    "score",
    "interest",
    "rate",
    "mortgage",
    "eligibility",
    "approval",
    "risk",
    "payment",
    "financial",
    "finance",
    "borrow",
    "lender",
    "amortize",
    "principal",
    "collateral",
    "fico",
    "equity",
    "refinance",
    "default",
    "apr",
}

_GUARDRAIL_REPLY = (
    "My parameters restrict me from discussing topics outside of "
    "loan eligibility and financial profiling."
)


def _mock_chat_response(user_message: str) -> str:
    lowered = user_message.lower()
    is_financial = any(keyword in lowered for keyword in _FINANCIAL_KEYWORDS)
    if not is_financial:
        return _GUARDRAIL_REPLY
    return (
        "I can help with loan eligibility factors such as income, loan burden, "
        "interest rate, default history, employment length, and the model's "
        "approval output. Share an application result if you want a focused review."
    )


def _build_chat_context(
    application: Optional[dict[str, Any]],
    prediction: Optional[PredictionResponse],
) -> tuple[list[str], Optional[LoanApplication]]:
    prompt_lines = []
    app_obj = None

    if application:
        try:
            app_obj = LoanApplication(**application)
            prompt_lines.append(
                "Application context supplied by the user: "
                f"{_format_application_summary(app_obj)}"
            )
        except Exception:
            prompt_lines.append(f"Application context supplied by the user: {application}")

    if prediction:
        prompt_lines.append(
            "Model prediction context: "
            f"Decision: {prediction.decision}; "
            f"Approved probability: {prediction.probability_approved}; "
            "Model-tested improvement suggestions: "
            f"{', '.join(prediction.risk_factors) if prediction.risk_factors else 'None'}"
        )

    return prompt_lines, app_obj


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
    if store.model is None or store.scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    prob_approved = _score_application(application)
    risk_factors = _derive_improvement_suggestions(application, prob_approved)
    decision = _make_decision(prob_approved)

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
    prompt_lines, app_obj = _build_chat_context(message.application, message.prediction)
    if prompt_lines:
        prompt_lines.append(
            "Answer the user's question using the application and prediction context above."
        )
    prompt_lines.append(f"User: {message.user_message}")
    prompt_text = "\n".join(prompt_lines)

    if _gemini_model is not None:
        try:
            result = _gemini_model.generate_content(prompt_text)
            return ChatResponse(response=result.text or _GUARDRAIL_REPLY)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"LLM provider error: {exc}") from exc

    if app_obj is not None:
        if message.prediction:
            improvements = message.prediction.risk_factors
            probability = message.prediction.probability_approved
        else:
            probability = _score_application(app_obj)
            improvements = _derive_improvement_suggestions(app_obj, probability)

        improvement_text = (
            "; ".join(improvements)
            if improvements
            else "No high-impact model-tested changes were found for this application."
        )
        prediction_text = (
            f"Decision: {message.prediction.decision}; "
            f"Probability: {message.prediction.probability_approved}; "
            f"Suggested improvements: {', '.join(message.prediction.risk_factors)}"
            if message.prediction
            else "No model prediction supplied."
        )
        return ChatResponse(
            response=(
                f"I reviewed the provided application. Summary: {_format_application_summary(app_obj)}. "
                f"Model context: {prediction_text}. "
                f"Current estimated approval probability: {probability * 100:.1f}%. "
                f"The most useful model-tested next steps are: {improvement_text}. "
                "These are counterfactual estimates from the trained model, so treat them as guidance."
            )
        )

    return ChatResponse(response=_mock_chat_response(message.user_message))
