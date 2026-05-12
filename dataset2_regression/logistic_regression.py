# dataset2_regression/logistic_regression.py
# TEAM_MEMBER: Christian 
# TASK: Regression — predict survival time in days

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

# LINEAR REGRESSION
# Predicts exact survival time in days

lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)

#Calculate Metrics
r2 = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\n==============================")
print("LINEAR REGRESSION")
print("==============================")
print(f"R2 Score: {r2:.3f}")
print(f"RMSE:     {rmse:.1f} days")
print("R2 closer to 1.0 = better | RMSE = avg days off")

# FEATURE IMPORTANCE

feature_names = ['AGE OF PATIENTS', 'AGE AT MENARACHE', 'BREASTFEED',
                 'CONTRACEPT', 'DETECTION', 'NEOADJUVANT']
importance = pd.Series(lr.coef_, index=feature_names)
importance = importance.abs().sort_values(ascending=False)

print("\n==============================")
print("FEATURE IMPORTANCE (Coefficients)")
print("==============================")
print(importance)

