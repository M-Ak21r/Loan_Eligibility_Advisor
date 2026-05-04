import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Config
# -----------------------------
RANDOM_SEED = 42

# try different n_estimators
n_estimators_list = [10, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]

# Keep these aligned with train_model.py so the comparison matches the app model.
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 5

# -----------------------------
# Load Data (same logic)
# -----------------------------
df = pd.read_csv("credit_risk_dataset.csv")
df = df.dropna()

# target detection (simple version)
target_col = "loan_status"

X = df.drop(columns=[target_col])
y = df[target_col]

# encode categorical
X = pd.get_dummies(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Train loop
# -----------------------------
accuracies = []

for n in n_estimators_list:
    model = RandomForestClassifier(
        n_estimators=n,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"n_estimators={n} : accuracy={acc:.4f}")

    if n == 300:
        print("\nClassification report for n_estimators=300:")
        print(classification_report(y_test, y_pred, digits=4))

# -----------------------------
# Plot graph
# -----------------------------
plt.figure()

plt.plot(n_estimators_list, accuracies, marker='o')

plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.title("Random Forest Performance vs n_estimators")

plt.grid()

plt.show()
