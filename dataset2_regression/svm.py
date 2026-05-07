# =============================================================
# dataset2_regression/svm.py
# TEAM MEMBER: Mario
# DATASET: mendeley_cancer_survival.csv
# TASK: Regression — predict survival time in days
#
# HOW TO RUN:
#   python dataset2_regression/svm.py
#
# NOTE: StandardScaler is already applied in preprocessing.py
#       Do not scale the data again here
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_dataset2
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load preprocessed Dataset 2 (already scaled)
X_train, X_test, y_train, y_test = load_dataset2()

def evaluate(name, y_test, preds):
    print(f"\n{'='*30}")
    print(name)
    print(f"{'='*30}")
    print(f"R2 Score: {r2_score(y_test, preds):.3f}")
    print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, preds)):.1f} days")
    print("R2 closer to 1.0 = better | RMSE = avg days off")

# =============================================================
# SVR WITHOUT KERNEL (Linear)
# Predicts survival days with a straight line
# TODO: Run and record R2 and RMSE as your baseline
# TODO: Try changing C=1 to C=0.1 or C=10 — does R2 improve?
# =============================================================
svr_linear = SVR(kernel='linear', C=1)
svr_linear.fit(X_train, y_train)
preds_linear = svr_linear.predict(X_test)
evaluate("SVR WITHOUT KERNEL (Linear)", y_test, preds_linear)

# =============================================================
# SVR WITH KERNEL (RBF)
# Curved boundary — handles nonlinear relationships better
# TODO: Compare R2 here vs linear above — which is higher?
# TODO: Try changing C= and gamma= values to tune performance
# =============================================================
svr_rbf = SVR(kernel='rbf', C=1, gamma='scale')
svr_rbf.fit(X_train, y_train)
preds_rbf = svr_rbf.predict(X_test)
evaluate("SVR WITH KERNEL (RBF)", y_test, preds_rbf)

# TODO: Add a comment here — which kernel worked better and why?
print("\nDone! Higher R2 = better. Compare with xgboost_model.py results.")