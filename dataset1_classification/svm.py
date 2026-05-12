# =============================================================
# dataset1_classification/svm.py
# TEAM MEMBER: Diego
# DATASET: breast_cancer_survival.csv
# TASK: Classification — predict Dead (1) vs Alive (0)
#
# 
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_dataset1
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, confusion_matrix
)

# Load preprocessed Dataset 1
X_train, X_test, y_train, y_test = load_dataset1()

def evaluate(name, y_test, preds):
    cm = confusion_matrix(y_test, preds)
    TN, FP, FN, TP = cm.ravel()
    print(f"\n{'='*30}")
    print(name)
    print(f"{'='*30}")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.3f}")
    print(f"Precision: {precision_score(y_test, preds, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_test, preds, zero_division=0):.3f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Correct Alive: {TN} | False Alarms: {FP}")
    print(f"Missed Dead:   {FN} | Correct Dead: {TP}")

# =============================================================
# SVM WITHOUT KERNEL (Linear)
# A straight line separates alive vs dead patients
# =============================================================
svm_linear = SVC(kernel='linear', C=1, class_weight = 'balanced', random_state=42)
svm_linear.fit(X_train, y_train)
preds_linear = svm_linear.predict(X_test)
evaluate("SVM WITHOUT KERNEL (Linear)", y_test, preds_linear)

# =============================================================
# SVM WITH KERNEL (RBF)
# Uses a curved boundary — usually better on real-world data
# 
# =============================================================
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale', class_weight = 'balanced', random_state=42)
svm_rbf.fit(X_train, y_train)
preds_rbf = svm_rbf.predict(X_test)
evaluate("SVM WITH KERNEL (RBF)", y_test, preds_rbf)

# # RESULTS SUMMARY:
# Linear SVM: Recall=0.692, catches 9/13 dead patients, 28 false alarms
# RBF SVM:    Recall=0.846, catches 11/13 dead patients, 36 false alarms
# Conclusion: RBF outperforms linear for detecting high-risk patients
# Tradeoff: higher recall comes with more false alarms
# In clinical settings, higher recall is preferred — missing a dying
# patient is worse than a false alarm
