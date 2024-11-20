import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import learning_curve
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the datasets
df_accidents = pd.read_csv('C:/Users/umarb/OneDrive/Documents/BS in CSE/3rd Year/1st Semester/CSE445/fatalaccidentdata.csv')
df_casualties = pd.read_csv('C:/Users/umarb/OneDrive/Documents/BS in CSE/3rd Year/1st Semester/CSE445/fatalcasualtydata.csv')

# Dropping rows with missing values in critical columns
df_accidents = df_accidents.dropna(subset=['Fatal_Accident_Index'])
df_casualties = df_casualties.dropna(subset=['Fatal_Accident_Index'])

# Convert age to numeric
df_casualties['Fatal_Casualty_Age'] = pd.to_numeric(df_casualties['Fatal_Casualty_Age'], errors='coerce')
df_casualties.dropna(subset=['Fatal_Casualty_Age'], inplace=True)
df_casualties['Fatal_Casualty_Age'] = df_casualties['Fatal_Casualty_Age'].astype(int)

#Merging the two datasets
df = pd.merge(df_accidents, df_casualties, on='Fatal_Accident_Index', how='inner')

# One-hot encode the `Month_of_Accident` column
month_encoded = pd.get_dummies(df['Month_of_Accident'], prefix='Month')

# Concatenate the encoded month columns with the original DataFrame
df = pd.concat([df, month_encoded], axis=1)

# Create a binary sex column
df['Fatal_Casualty_Sex_Binary'] = df['Fatal_Casualty_Sex'].map({'Female': 0, 'Male': 1})
df = df[df['Fatal_Casualty_Sex'] != 'Not Reported']

# Select features and target variable
X = df[['Month_of_Accident', 'Hour_of_Accident', 'Longitude', 'Latitude', 'Pedestrian_Casualties', 'Pedal_Cycles', 'Motor_Cycles', 'Cars', 'Buses_or_Coaches', 'Vans', 'HGVs', 'Other_Vehicles', 'Total_Vehicles_Involved', 'Fatal_Casualties', 'Serious_Casualties', 'Slight_Casualties', 'Total_Number_of_Casualties', 'Fatal_Casualty_Sex_Binary', 'Fatal_Casualty_Age']]
y = df['Fatal_Casualty_Type']

# One-hot encode the `Month_of_Accident` column for X
X = pd.get_dummies(X, columns=['Month_of_Accident'])

# Impute missing values BEFORE applying SMOTE
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Impute missing values in X
