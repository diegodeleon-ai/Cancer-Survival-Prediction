# =============================================================
# dataset2_regression/xgboost_model.py
# TEAM MEMBER: Diego
# DATASET: mendeley_cancer_survival.csv
# TASK: Regression — predict survival time in days
# 
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
# =============================================================
xgb = XGBRegressor(
    n_estimators=50,    #fewer trees to prevent overfitting on small dataset
    max_depth = 2,      #shallower trees make model simpler
    learning_rate = 0.05,  #smaller lr to increase performance
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
# 
# =============================================================
feature_names = ['AGE OF PATIENTS', 'AGE AT MENARACHE', 'BREASTFEED',
                 'CONTRACEPT', 'DETECTION', 'NEOADJUVANT']
importance = pd.Series(xgb.feature_importances_, index=feature_names)
importance = importance.sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE")
print("==============================")
print(importance)

# 
