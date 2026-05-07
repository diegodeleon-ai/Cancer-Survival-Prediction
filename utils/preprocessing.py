# =============================================================
# utils/preprocessing.py
# Shared preprocessing functions for both datasets
# Used by all team members — do not modify without discussing
#
# TODO (Diego): Verify Tumour_Stage unique values match mapping
# TODO (All): If you add a new feature, add it to feature_cols
#             and update both load functions accordingly
# =============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================
# DATASET 1 — Breast Cancer (breast_cancer_survival.csv)
# Used for: Classification
# =============================================================
def load_dataset1(path='data/breast_cancer_survival.csv'):
    df = pd.read_csv(path)

    # Calculate survival time
    df['Date_of_Surgery'] = pd.to_datetime(df['Date_of_Surgery'], format='%d-%b-%y')
    df['Date_of_Last_Visit'] = pd.to_datetime(df['Date_of_Last_Visit'], format='%d-%b-%y')
    df['Survival_Days'] = (df['Date_of_Last_Visit'] - df['Date_of_Surgery']).dt.days

    # Event indicator: 1 = died, 0 = alive/censored
    df['Event'] = (df['Patient_Status'] == 'Dead').astype(int)

    # Drop missing survival times
    df = df.dropna(subset=['Survival_Days'])

    # Encode categorical columns
    df['Gender'] = df['Gender'].map({'FEMALE': 0, 'MALE': 1})
    df['Tumour_Stage'] = df['Tumour_Stage'].map({'I': 1, 'II': 2, 'III': 3})
    df['ER status'] = df['ER status'].map({'Negative': 0, 'Positive': 1})
    df['PR status'] = df['PR status'].map({'Positive': 1, 'Negative': 0})
    df['HER2 status'] = df['HER2 status'].map({'Positive': 1, 'Negative': 0})

    # One-hot encode Histology and Surgery_type
    df = pd.get_dummies(df, columns=['Histology', 'Surgery_type'])

    feature_cols = [
        'Age', 'Gender', 'Protein1', 'Protein2', 'Protein3', 'Protein4',
        'Tumour_Stage', 'ER status', 'PR status', 'HER2 status',
        'Histology_Infiltrating Ductal Carcinoma',
        'Histology_Infiltrating Lobular Carcinoma',
        'Histology_Mucinous Carcinoma',
        'Surgery_type_Lumpectomy',
        'Surgery_type_Modified Radical Mastectomy',
        'Surgery_type_Other',
        'Surgery_type_Simple Mastectomy'
    ]

    X = df[feature_cols]
    y = df['Event']  # Classification target: 0 = Alive, 1 = Dead

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# =============================================================
# DATASET 2 — Mendeley Cancer Survival (mendeley_cancer_survival.csv)
# Used for: Regression
# StandardScaler applied here so all models use the same scaled data
# =============================================================
def load_dataset2(path='data/mendeley_cancer_survival.csv'):
    df = pd.read_csv(path)

    feature_cols = [
        'AGE OF PATIENTS', 'AGE AT MENARACHE', 'BREASTFEED',
        'CONTRACEPT', 'DETECTION', 'NEOADJUVANT'
    ]

    X = df[feature_cols]
    y = df['TIME(In Days)']  # Regression target: survival time in days

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # StandardScaler normalizes all features to same scale
    # Required for SVM — without this, larger numbers dominate
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
