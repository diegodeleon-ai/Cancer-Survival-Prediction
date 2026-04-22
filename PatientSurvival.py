#Main Topic: Cancer Survival Prediction 
# * Create a machine learning model that can predict the survival time of 
#   cancer patients based on various features such as age, type of cancer, cancer stage,
#   year diagnosed, etc.
# * Gives the user a prediction/ estimate on their time to live.
# * It can allow doctors or other health-care professionals to get an insight
#   into the time frame a patient has to live and how to act accordingly. 
# * Data will be sourced from publicly available sources such as the National 
#   Cancer Institute, the U.S. National Institutes of Health and the CDC.

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Load the training data
train_data = pd.read_csv('data/breast_cancer_survival.csv')
df = pd.read_csv('data/breast_cancer_survival.csv')


#Display first 5 rows
print(train_data.head())

#See all columns and patient status
print(df.columns.tolist())
print(df.shape)
print(df['Patient_Status'].value_counts())

# Calculate survival time in days
df['Date_of_Surgery'] = pd.to_datetime(df['Date_of_Surgery'], format='%d-%b-%y')
df['Date_of_Last_Visit'] = pd.to_datetime(df['Date_of_Last_Visit'], format='%d-%b-%y')
df['Survival_Days'] = (df['Date_of_Last_Visit'] - df['Date_of_Surgery']).dt.days

# Create the event indicator (True = died, False = still alive/censored)
df['Event'] = df['Patient_Status'] == 'Dead'

# Drop rows with missing survival time
df = df.dropna(subset=['Survival_Days'])
print(f"Patients remaining: {len(df)}")

#Encode categorical values with numbers
df['Gender'] = df['Gender'].map({'FEMALE': 0, 'MALE': 1})
df['Tumour_Stage'] = df['Tumour_Stage'].map({'STAGE I': 1, 'STAGE II': 2, 'STAGE III': 3})
df['ER status'] = df['ER status'].map({'Negative': 0, 'Positive': 1})
df['PR status'] = df['PR status'].map({'Positive': 1, 'Negative': 0})
df['HER2 status'] = df['HER2 status'].map({'Positive': 1, 'Negative': 0})

# One-hot encode Histology and Surgery_type
df = pd.get_dummies(df, columns=['Histology', 'Surgery_type'])

print(df.columns.tolist())
print(df.head())

# Define features
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

# Split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training patients: {len(X_train)}")
print(f"Test patients: {len(X_test)}")


