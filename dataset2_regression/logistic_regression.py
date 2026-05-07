# =============================================================
# dataset2_regression/logistic_regression.py
# TEAM MEMBER: Christian
# DATASET: mendeley_cancer_survival.csv
# TASK: Regression — predict survival time in days
#
# HOW TO RUN:
#   python dataset2_regression/logistic_regression.py
#
# NOTE: StandardScaler is already applied in preprocessing.py
# NOTE: For regression we use Linear Regression (not Logistic)
#       Logistic Regression is only for classification tasks
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import load_dataset2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

# Load preprocessed Dataset 2 (already scaled)
X_train, X_test, y_train, y_test = load_dataset2()

# =============================================================
# LINEAR REGRESSION
# Predicts exact survival time in days
# TODO: Run and record R2 and RMSE
# TODO: Compare results with svm.py and xgboost_model.py
#       Which model predicts survival time most accurately?
# =============================================================
lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")
print(f"R2 Score: {r2_score(y_test, preds):.3f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, preds)):.1f} days")
print("R2 closer to 1.0 = better | RMSE = avg days off")

# =============================================================
# FEATURE IMPORTANCE
# TODO: Do these features make clinical sense?
# TODO: Compare with feature importance from Dataset 1 models
#       Are the same features important across both datasets?
# =============================================================
feature_names = ['AGE OF PATIENTS', 'AGE AT MENARACHE', 'BREASTFEED',
                 'CONTRACEPT', 'DETECTION', 'NEOADJUVANT']
importance = pd.Series(lr.coef_, index=feature_names)
importance = importance.abs().sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE (Coefficients)")
print("==============================")
print(importance)

# TODO: Add a comment — which feature surprised you most?