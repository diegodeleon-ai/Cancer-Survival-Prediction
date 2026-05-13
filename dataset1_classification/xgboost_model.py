# =============================================================
# dataset1_classification/xgboost_model.py
# TEAM MEMBER: Mario
# DATASET: breast_cancer_survival.csv
# TASK: Classification — predict Dead (1) vs Alive (0)
#
# HOW TO RUN:
#   python dataset1_classification/xgboost_model.py
#
# INSTALL IF NEEDED:
#   pip install xgboost
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_dataset1
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, confusion_matrix
)
import pandas as pd

# Load preprocessed Dataset 1
X_train, X_test, y_train, y_test = load_dataset1()

# =============================================================
# XGBOOST CLASSIFIER
# Gradient boosting builds trees sequentially, where each tree
# tries to correct the errors from the previous trees.
#
# eval_metric='logloss' connects to the professor's note about
# negative log likelihood / log loss.
#
# scale_pos_weight helps with class imbalance because there are
# more Alive patients than Dead patients.
# =============================================================
alive_count = (y_train == 0).sum()
dead_count = (y_train == 1).sum()
scale_pos_weight = alive_count / dead_count

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

# =============================================================
# THRESHOLD ADJUSTMENT
# Default classification threshold is 0.50.
#
# Lower threshold = higher recall, more false alarms.
# Higher threshold = fewer false alarms, lower recall.
#
# Since this is a medical classification task, recall matters
# because missing a Dead/high-risk patient is worse than creating
# a false alarm.
# =============================================================
dead_probabilities = xgb.predict_proba(X_test)[:, 1]

thresholds = [0.50, 0.40, 0.30]
results = []

for threshold in thresholds:
    preds = (dead_probabilities >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)

    cm = confusion_matrix(y_test, preds)
    TN, FP, FN, TP = cm.ravel()

    results.append({
        "Threshold": threshold,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Correct Alive": TN,
        "False Alarms": FP,
        "Missed Dead": FN,
        "Correct Dead": TP
    })

    print("\n==============================")
    print(f"XGBOOST CLASSIFIER — THRESHOLD {threshold}")
    print("==============================")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Correct Alive: {TN} | False Alarms: {FP}")
    print(f"Missed Dead:   {FN} | Correct Dead: {TP}")

# =============================================================
# MODEL COMPARISON
# This table shows how changing the threshold affects performance.
# =============================================================
summary = pd.DataFrame(results)

print("\n==============================")
print("THRESHOLD COMPARISON")
print("==============================")
print(summary)

# Choose the best threshold based on recall first, then precision.
# Recall matters most here because missing Dead/high-risk patients
# is the biggest concern.
best_result = summary.sort_values(
    by=["Recall", "Precision"],
    ascending=False
).iloc[0]

print("\n==============================")
print("BEST THRESHOLD BASED ON RECALL")
print("==============================")
print(best_result)

# =============================================================
# FEATURE IMPORTANCE
# Higher value = feature was more important for XGBoost predictions.
# =============================================================
importance = pd.Series(xgb.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(importance.head(10))

# =============================================================
# RESULTS SUMMARY
# XGBoost is a nonlinear model that uses multiple decision trees.
# This makes it more flexible than Logistic Regression.
#
# For this medical classification problem, recall is very important
# because missing a Dead/high-risk patient is worse than creating
# a false alarm.
# =============================================================
print("\n==============================")
print("CONCLUSION")
print("==============================")
print(
    "XGBoost was used to predict whether a patient is Alive or Dead. "
    "The model uses logloss as its evaluation metric and scale_pos_weight "
    "to help with class imbalance. Different prediction thresholds were tested "
    "because the default threshold of 0.50 missed too many Dead/high-risk patients. "
    "Lowering the threshold improved recall, but it also increased false alarms. "
    "The 0.30 threshold gave the best recall, while the 0.50 threshold gave better "
    "overall accuracy. For this medical classification task, recall is especially "
    "important because missing a high-risk patient is worse than creating a false alarm. "
    "This model should still be compared with Logistic Regression and SVM before choosing "
    "the final classification model."
)