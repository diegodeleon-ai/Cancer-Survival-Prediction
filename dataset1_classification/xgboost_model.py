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
# Gradient boosting — builds trees sequentially, each one
# correcting the errors of the previous one
# eval_metric='logloss' is negative log likelihood (prof asked for this)
# TODO: Run the model and record Accuracy, Precision, Recall
# TODO: Try changing n_estimators to 50 or 200 — does it change results?
# TODO: Try adding scale_pos_weight=4 to handle class imbalance
#       (ratio of alive/dead = ~255/66 ≈ 4)
# =============================================================
xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)

cm = confusion_matrix(y_test, preds)
TN, FP, FN, TP = cm.ravel()

print("\n==============================")
print("XGBOOST CLASSIFIER")
print("==============================")
print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
print(f"Precision: {precision_score(y_test, preds, zero_division=0):.3f}")
print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.3f}")
print(f"\nConfusion Matrix:\n{cm}")
print(f"Correct Alive: {TN} | False Alarms: {FP}")
print(f"Missed Dead:   {FN} | Correct Dead: {TP}")

# =============================================================
# FEATURE IMPORTANCE
# TODO: Compare these results with logistic_regression.py
#       Do both models agree on which features matter most?
# =============================================================
importance = pd.Series(xgb.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(importance.head(10))

# TODO: Add a comment — how does XGBoost compare to SVM and Logistic Regression?
