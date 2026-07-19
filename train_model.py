import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned claims from SQLite — the model learns from the same
# data the analytics endpoints serve.
conn = sqlite3.connect("claims.db")
df = pd.read_sql("SELECT provider_name, diagnosis_code, claim_amount, claim_month, status FROM claims", conn)
conn.close()

# Label: 1 = denied, 0 = anything else. This is what the model must predict.
df["denied"] = (df["status"] == "denied").astype(int)

# Features: the clues. get_dummies converts text columns into 0/1 columns
# because models can only do math on numbers.
X = pd.get_dummies(df[["provider_name", "diagnosis_code", "claim_amount", "claim_month"]])
y = df["denied"]

# Hold back 20% as unseen exam questions. random_state makes the split
# reproducible - same split every run.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("training rows:", len(X_train))
print("test rows:", len(X_test))
print("denial rate:", round(y.mean() * 100, 1), "%")
print("feature columns:", list(X.columns))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train: the model studies 400 claims and their outcomes.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Exam: predict on the 100 claims it has never seen.
predictions = model.predict(X_test)

print("accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

from sklearn.ensemble import RandomForestClassifier

# Forest of 200 decision trees, each voting; trees split on thresholds
# ("amount > X?") so step-shaped patterns suit them well.
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
forest_predictions = forest.predict(X_test)

print("forest accuracy:", accuracy_score(y_test, forest_predictions))
print(classification_report(y_test, forest_predictions))

# The forest's raw probabilities for class 1 (denied), one per test claim.
probs = forest.predict_proba(X_test)[:, 1]

# Business rule: flag anything with >= 35% denial risk for human review.
flagged = (probs >= 0.35).astype(int)

print("--- threshold 0.35 (flag for review) ---")
print(classification_report(y_test, flagged))

import joblib

# Persist model + training columns together: at serving time, incoming
# claims must be encoded into EXACTLY these columns, in this order.
joblib.dump((forest, list(X.columns)), "denial_model.joblib")
print("saved denial_model.joblib")