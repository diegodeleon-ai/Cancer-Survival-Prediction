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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Load preprocessed Dataset 2
X_train, X_test, y_train, y_test = load_dataset2()

def evaluate(name, y_test, preds):
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\n{'='*30}")
    print(name)
    print(f"{'='*30}")
    print(f"R2 Score: {r2:.3f}")
    print(f"MAE:      {mae:.1f} days")
    print(f"RMSE:     {rmse:.1f} days")
    print("R2 closer to 1.0 = better")
    print("MAE/RMSE = how many days off the model is on average")

    return r2, mae, rmse

# =============================================================
# SVR LINEAR
# Predicts survival days using a straight-line relationship.
# This is the simpler SVR model and works as a baseline.
# =============================================================
svr_linear = SVR(kernel='linear', C=1)
svr_linear.fit(X_train, y_train)
preds_linear = svr_linear.predict(X_test)
linear_r2, linear_mae, linear_rmse = evaluate("SVR LINEAR", y_test, preds_linear)

# =============================================================
# SVR WITH RBF KERNEL
# Uses a nonlinear curve to predict survival days.
# This can perform better if the relationship between the features
# and survival time is not linear.
# =============================================================
svr_rbf = SVR(kernel='rbf', C=1, gamma='scale')
svr_rbf.fit(X_train, y_train)
preds_rbf = svr_rbf.predict(X_test)
rbf_r2, rbf_mae, rbf_rmse = evaluate("SVR WITH KERNEL (RBF)", y_test, preds_rbf)

# =============================================================
# RESULTS SUMMARY
# For Dataset 2 regression, the best model should have:
#   - higher R2 score
#   - lower MAE
#   - lower RMSE
# =============================================================
print("\n==============================")
print("CONCLUSION")
print("==============================")
print(
    "SVR was used to estimate survival time in days. The Linear SVR and RBF SVR "
    "had very similar results in this run. Both models should be compared with "
    "Linear Regression and XGBoost using R2, MAE, and RMSE before choosing the "
    "best regression model."
)