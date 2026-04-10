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

print(df[['Survival_Days', 'Event']].head(10))
print(df['Event'].value_counts())
