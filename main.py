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
    approval_label: Any = None
    approval_profile: dict[str, dict[str, Any]] = {}

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
        store.approval_label = artifacts.get(
            "approval_label",
            artifacts.get("positive_label"),
        )
        store.approval_profile = artifacts.get("approval_profile", {})
    else:
        store.model = artifacts
        store.scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
        store.feature_columns = []
<<<<<<< Updated upstream
        store.positive_label = None
=======
        store.approval_label = None
        store.approval_profile = {}

    if hasattr(store.model, "n_jobs"):
        store.model.n_jobs = 1
    # Initialise Gemini model once if the API key is available
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and _GENAI_AVAILABLE:
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=_SYSTEM_PROMPT,
        )

>>>>>>> Stashed changes
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
# Helpers: model-driven probability and improvement suggestions
# ---------------------------------------------------------------------------

<<<<<<< Updated upstream
RISK_THRESHOLDS = {
    "Low credit score (< 620)": lambda a: a.credit_score < 620,
    "High debt-to-income ratio (> 40%)": lambda a: a.debt_to_income_ratio > 0.40,
    "Short employment history (< 2 years)": lambda a: a.employment_length_years < 2,
    "High loan-to-income ratio (> 40%)": lambda a: (
        a.loan_amount_requested / a.annual_income
    ) > 0.40,
    "Low annual income (< $30,000)": lambda a: a.annual_income < 30_000,
}
=======
GRADE_ORDER = ("A", "B", "C", "D", "E", "F", "G")
MIN_SUGGESTION_LIFT = 0.01
>>>>>>> Stashed changes


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
    payload["loan_percent_income"] = max(
        0,
        min(float(payload["loan_percent_income"]), 1),
    )
    return LoanApplication(**payload)


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
    if approved_loan_percent and application.loan_percent_income > approved_loan_percent:
        combined_payload["loan_amnt"] = max(1.0, application.person_income * approved_loan_percent)
        combined_payload["loan_percent_income"] = approved_loan_percent
    if approved_interest and application.loan_int_rate > approved_interest:
        combined_payload["loan_int_rate"] = approved_interest
    if approved_grade and _grade_is_better(approved_grade, application.loan_grade):
        combined_payload["loan_grade"] = approved_grade
    if approved_emp_length and application.person_emp_length < approved_emp_length:
        combined_payload["person_emp_length"] = approved_emp_length
    if approved_default_status:
        combined_payload["cb_person_default_on_file"] = approved_default_status
    if approved_income and application.person_income < approved_income:
        combined_payload["person_income"] = approved_income
        combined_payload["loan_percent_income"] = (
            combined_payload["loan_amnt"] / approved_income
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

    prob_approved = _score_application(application)
    risk_factors = _derive_improvement_suggestions(application, prob_approved)
    decision = _make_decision(prob_approved)


    return PredictionResponse(
        decision=decision,
        probability_approved=round(prob_approved, 4),
        risk_factors=risk_factors,
    )
<<<<<<< Updated upstream
=======


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
            "Model-tested improvement suggestions: "
            f"{', '.join(message.prediction.risk_factors) if message.prediction.risk_factors else 'None'}"
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

    # Fallback mock: if an application was provided, include model-tested improvements
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
            f"Suggested improvements: {', '.join(message.prediction.risk_factors) if message.prediction else 'None'}"
            if message.prediction
            else "No model prediction supplied."
        )
        mock = (
            f"I reviewed the provided application. Summary: {_format_application_summary(app_obj)}. "
            f"Model context: {prediction_text} "
            f"Current estimated approval probability: {probability * 100:.1f}%. "
            f"The most useful model-tested next steps are: {improvement_text}. "
            "These are counterfactual estimates from the trained model, so treat them as guidance rather than a lender guarantee."
        )
        return ChatResponse(response=mock)

    return ChatResponse(response=_mock_chat_response(message.user_message))
>>>>>>> Stashed changes
