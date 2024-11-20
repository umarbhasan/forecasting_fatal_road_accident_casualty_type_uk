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