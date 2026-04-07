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


