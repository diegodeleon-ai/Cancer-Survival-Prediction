# =============================================================
# exploration.py
# Initial data exploration and baseline model
# This was our first look at the data before the project
# structure was set up. Not the main model — see:
#   dataset1_classification/ for classification models
#   dataset2_regression/ for regression models
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =============================================================
# LOAD DATA
# =============================================================
df = pd.read_csv('data/breast_cancer_survival.csv')
df2 = pd.read_csv('data/mendeley_cancer_survival.csv')

print("=== DATASET 1 (Breast Cancer) ===")
print(df.head())
print(df.columns.tolist())
print(df.shape)
print(df['Patient_Status'].value_counts())

print("\n=== DATASET 2 (Mendeley) ===")
print(df2.shape)
print(df2.columns.tolist())
print(df2.head())

# =============================================================
# PREPROCESSING — DATASET 1
# =============================================================
df['Date_of_Surgery'] = pd.to_datetime(df['Date_of_Surgery'], format='%d-%b-%y')
df['Date_of_Last_Visit'] = pd.to_datetime(df['Date_of_Last_Visit'], format='%d-%b-%y')
df['Survival_Days'] = (df['Date_of_Last_Visit'] - df['Date_of_Surgery']).dt.days
df['Event'] = df['Patient_Status'] == 'Dead'
df = df.dropna(subset=['Survival_Days'])
print(f"\nPatients remaining: {len(df)}")

# Encode categorical columns
df['Gender'] = df['Gender'].map({'FEMALE': 0, 'MALE': 1})
df['Tumour_Stage'] = df['Tumour_Stage'].map({'I': 1, 'II': 2, 'III': 3})
df['ER status'] = df['ER status'].map({'Negative': 0, 'Positive': 1})
df['PR status'] = df['PR status'].map({'Positive': 1, 'Negative': 0})
df['HER2 status'] = df['HER2 status'].map({'Positive': 1, 'Negative': 0})
df = pd.get_dummies(df, columns=['Histology', 'Surgery_type'])

# =============================================================
# BASELINE MODEL — Random Forest Regressor
# NOTE: This is a naive baseline — it does not handle censoring
# The proper models are in dataset1_classification/
# and dataset2_regression/
# =============================================================
feature_cols = ['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3', 'Protein4',
                'Tumour_Stage', 'ER status', 'PR status', 'HER2 status',
                'Histology_Infiltrating Ductal Carcinoma',
                'Histology_Infiltrating Lobular Carcinoma',
                'Histology_Mucinous Carcinoma',
                'Surgery_type_Lumpectomy',
                'Surgery_type_Modified Radical Mastectomy',
                'Surgery_type_Other',
                'Surgery_type_Simple Mastectomy']

X = df[feature_cols]
y = df['Survival_Days']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training patients: {len(X_train)}")
print(f"Test patients:     {len(X_test)}")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = np.mean(np.abs(y_test - predictions))
print(f"\nBaseline MAE: {mae:.1f} days (~{mae/30:.1f} months off on average)")
print("Note: MAE used here for exploration only — see proper models for better metrics")