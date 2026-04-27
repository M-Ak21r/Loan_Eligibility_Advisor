"""
train_model.py
--------------
Phase 1: ML Engine

Generates a synthetic loan dataset, trains a RandomForestClassifier, prints
evaluation metrics, and serializes the model and scaler to disk.

Usage:
    python train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------
N = 5_000

annual_income = rng.normal(loc=65_000, scale=25_000, size=N).clip(15_000, 300_000)
credit_score = rng.normal(loc=650, scale=80, size=N).clip(300, 850)
debt_to_income_ratio = rng.uniform(0.05, 0.60, size=N)
employment_length_years = rng.exponential(scale=5, size=N).clip(0, 30)
loan_amount_requested = rng.normal(loc=20_000, scale=10_000, size=N).clip(1_000, 100_000)

# Deterministic approval rule so the dataset is learnable
approval_score = (
    (annual_income / 100_000) * 0.3
    + (credit_score / 850) * 0.35
    + (1 - debt_to_income_ratio) * 0.2
    + (employment_length_years / 30) * 0.1
    + (1 - loan_amount_requested / 100_000) * 0.05
)
noise = rng.normal(0, 0.05, size=N)
loan_approved = (approval_score + noise >= 0.55).astype(int)

df = pd.DataFrame(
    {
        "annual_income": annual_income,
        "credit_score": credit_score,
        "debt_to_income_ratio": debt_to_income_ratio,
        "employment_length_years": employment_length_years,
        "loan_amount_requested": loan_amount_requested,
        "loan_approved": loan_approved,
    }
)

print(f"Dataset shape: {df.shape}")
print(f"Approval rate : {df['loan_approved'].mean():.2%}\n")

# ---------------------------------------------------------------------------
# 2. Feature / target split and train-test split
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "annual_income",
    "credit_score",
    "debt_to_income_ratio",
    "employment_length_years",
    "loan_amount_requested",
]

X = df[FEATURE_COLS].values
y = df["loan_approved"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# ---------------------------------------------------------------------------
# 3. Feature scaling
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)
clf.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------
accuracy = clf.score(X_test_scaled, y_test)
print(f"Test Accuracy : {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test_scaled)))

# ---------------------------------------------------------------------------
# 6. Serialization
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

joblib.dump(clf, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"Model  saved → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")
