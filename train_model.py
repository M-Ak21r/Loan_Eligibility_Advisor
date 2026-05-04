"""
train_model.py
--------------
Phase 1: ML Engine

Ingests credit_data.csv with pandas, performs lightweight EDA,
trains a RandomForestClassifier, prints evaluation metrics, and serializes
the fitted artifacts for backend ingestion.

Usage:
    python train_model.py
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MODEL_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = (
    MODEL_DIR / "credit_risk_dataset.csv",
)
MODEL_PATH = MODEL_DIR / "rf_model.pkl"

PREFERRED_TARGET_COLUMNS = (
    "loan_approved",
    "approved",
    "loan_status",
    "status",
    "target",
    "label",
    "y",
)

APPROVAL_TARGET_COLUMNS = {
    "loan_approved",
    "approved",
    "approval_status",
    "is_approved",
}

DEFAULT_RISK_TARGET_COLUMNS = {
    "loan_status",
    "default",
    "defaulted",
    "charged_off",
    "bad_loan",
    "risk",
}


def _resolve_target_column(df: pd.DataFrame) -> str:
    for column in PREFERRED_TARGET_COLUMNS:
        if column in df.columns:
            return column
    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature column and one target column.")
    return df.columns[-1]


def _resolve_positive_label(y: pd.Series):
    unique_values = list(pd.Series(y).dropna().unique())
    if len(unique_values) != 2:
        return None

    preferred_labels = (
        1,
        True,
        "1",
        "true",
        "yes",
        "y",
        "approved",
        "approve",
        "positive",
    )
    for candidate in preferred_labels:
        for value in unique_values:
            if str(value).strip().lower() == str(candidate).strip().lower():
                return value

    if pd.api.types.is_numeric_dtype(pd.Series(unique_values)):
        return max(unique_values)

    return sorted((str(value) for value in unique_values))[-1]


def _resolve_approval_label(target_column: str, y: pd.Series):
    positive_label = _resolve_positive_label(y)
    unique_values = list(pd.Series(y).dropna().unique())
    normalized_target = target_column.strip().lower()

    if len(unique_values) != 2:
        return positive_label

    if normalized_target in DEFAULT_RISK_TARGET_COLUMNS:
        for value in unique_values:
            if str(value).strip().lower() in {"0", "false", "no", "n", "good", "paid"}:
                return value

    if normalized_target in APPROVAL_TARGET_COLUMNS:
        return positive_label

    approval_words = {"approved", "approve", "yes", "y", "true", "good", "paid"}
    for value in unique_values:
        if str(value).strip().lower() in approval_words:
            return value

    return positive_label

def _build_approval_profile(
    X_raw: pd.DataFrame,
    y: pd.Series,
    approval_label,
) -> dict[str, dict[str, Any]]:
    approved_rows = X_raw.loc[y == approval_label]
    if approved_rows.empty:
        return {"numeric_medians": {}, "categorical_modes": {}}

    numeric_medians = approved_rows.select_dtypes(include=["number"]).median().to_dict()
    categorical_modes = {}
    categorical_columns = approved_rows.select_dtypes(
        include=["object", "str", "category", "bool"],
    ).columns

    for column in categorical_columns:
        mode = approved_rows[column].mode(dropna=True)
        if not mode.empty:
            categorical_modes[column] = mode.iloc[0]

    return {
        "numeric_medians": numeric_medians,
        "categorical_modes": categorical_modes,
    }
DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        "No supported dataset file was found. Place credit_data.csv or credit_risk_dataset.csv "
        f"in {MODEL_DIR} and rerun train_model.py."
    )

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset: {DATA_PATH.name}")
print(f"Initial shape : {df.shape}")

df = df.dropna().copy()
print(f"After dropna   : {df.shape}")

target_column = _resolve_target_column(df)
print(f"Target column  : {target_column}")

y = df[target_column]
X_raw = df.drop(columns=[target_column])

categorical_columns = X_raw.select_dtypes(
    include=["object", "str", "category", "bool"],
).columns.tolist()
X_encoded = pd.get_dummies(X_raw, columns=categorical_columns)
feature_columns = X_encoded.columns.tolist()

print(f"Feature count  : {len(feature_columns)}")
if categorical_columns:
    print(f"Categorical cols: {', '.join(categorical_columns)}")
else:
    print("Categorical cols: none")

stratify_target = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=stratify_target,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_leaf=5,
    random_state=RANDOM_SEED,
    n_jobs=1,
)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy : {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

approval_label = _resolve_approval_label(target_column, y)
approval_profile = _build_approval_profile(X_raw, y, approval_label)
artifact_bundle = {
    "model": clf,
    "scaler": scaler,
    "feature_columns": feature_columns,
    "categorical_columns": categorical_columns,
    "target_column": target_column,
    "approval_label": approval_label,
    "approval_profile": approval_profile,
}

joblib.dump(artifact_bundle, MODEL_PATH)

print(f"Artifacts saved -> {MODEL_PATH}")
