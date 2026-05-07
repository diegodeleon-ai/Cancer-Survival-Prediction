# =============================================================
# dataset2_regression/xgboost_model.py
# TEAM MEMBER: Diego
# DATASET: mendeley_cancer_survival.csv
# TASK: Regression — predict survival time in days
#
# HOW TO RUN:
#   python dataset2_regression/xgboost_model.py
#
# INSTALL IF NEEDED:
#   pip install xgboost
#
# NOTE: StandardScaler is already applied in preprocessing.py
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_dataset2
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

# Load preprocessed Dataset 2 (already scaled)
X_train, X_test, y_train, y_test = load_dataset2()

# =============================================================
# XGBOOST REGRESSOR
# Predicts survival time in days
# TODO: Run and record R2 and RMSE
# TODO: Try changing n_estimators to 50 or 200 — does R2 improve?
# TODO: Try adding learning_rate=0.01 or 0.3 — what changes?
# =============================================================
xgb = XGBRegressor(
    n_estimators=50,    #fewer trees to prevent overfitting on small dataset
    max_depth = 2,      #shallower trees make model simpler
    learning_rate = 0.1,  #default lr
    random_state=42,
    eval_metric='rmse'
)
xgb.fit(X_train, y_train)
preds = xgb.predict(X_test)

print("\n==============================")
print("XGBOOST REGRESSOR")
print("==============================")
print(f"R2 Score: {r2_score(y_test, preds):.3f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, preds)):.1f} days")
print("R2 closer to 1.0 = better | RMSE = avg days off")

# =============================================================
# FEATURE IMPORTANCE
# TODO: Compare with Linear Regression coefficients above
#       Do both models agree on which features matter most?
# =============================================================
feature_names = ['AGE OF PATIENTS', 'AGE AT MENARACHE', 'BREASTFEED',
                 'CONTRACEPT', 'DETECTION', 'NEOADJUVANT']
importance = pd.Series(xgb.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(importance)

# TODO: Add a comment — how does XGBoost compare to SVR and Linear Regression?