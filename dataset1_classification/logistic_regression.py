
# dataset1_classification/logistic_regression.py
# TEAM MEMBER: Christian
# TASK: Classification — predict Dead (1) vs Alive (0)


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_dataset1
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, confusion_matrix
)
import pandas as pd

# Load preprocessed Dataset 1
X_train, X_test, y_train, y_test = load_dataset1()

# LOGISTIC REGRESSION
# Predicts probability of death — outputs 0 (Alive) or 1 (Dead)
# Uses negative log likelihood loss internally

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds, zero_division=0)
rec = recall_score(y_test, preds, zero_division=0)

cm = confusion_matrix(y_test, preds)
TN, FP, FN, TP = cm.ravel()

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"Correct Alive: {TN} | False Alarms: {FP}")
print(f"Missed Dead:   {FN} | Correct Dead: {TP}")

# FEATURE IMPORTANCE
# Positive coefficient = pushes prediction toward Dead
# Negative coefficient = pushes prediction toward Alive

importance = pd.Series(lr.coef_[0], index=X_train.columns)
importance = importance.abs().sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE (Coefficients)")
print("==============================")
print(importance.head(10))

# TODO: Add a comment — which features surprised you most?
