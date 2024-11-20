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

# Apply SMOTE to balance the classes in the dataset
smote = SMOTE(random_state=40, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)  # Use X_imputed

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=40, stratify=y_resampled)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check class distribution
print("Class distribution in training set after SMOTE:")
print(pd.Series(y_train).value_counts())

# Initialize Logistic Regression classifier
lr_model = LogisticRegression(random_state=40, class_weight='balanced', max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions for LR
lr_predictions = lr_model.predict(X_test_scaled)

# Print accuracy and classification report for LR
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions, zero_division=0))

# Initialize and train KNN classifier
knn_model = KNeighborsClassifier()  # You can adjust n_neighbors
knn_model.fit(X_train_scaled, y_train)

# Predictions for KNN
knn_predictions = knn_model.predict(X_test_scaled)

# Print accuracy and classification report for KNN
print("k-NN Accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions, zero_division=0))

# Initialize and train Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=40, class_weight='balanced')
dt_model.fit(X_train_scaled, y_train)

#Predictions for DT
dt_predictions = dt_model.predict(X_test_scaled)

# Print accuracy and classification report for DT
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions, zero_division=0))

# Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test) 

# Now use y_train_encoded and y_test_encoded in bias_variance_decomp

# --- Logistic Regression ---
print("Logistic Regression:")
lr_avg_expected_loss, lr_avg_bias, lr_avg_var = bias_variance_decomp(
    lr_model, 
    X_train_scaled, 
    y_train_encoded, 
    X_test_scaled, 
    y_test_encoded, 
    loss='0-1_loss', 
    num_rounds=100, 
    random_seed=40
)
print(f'Average expected loss: {lr_avg_expected_loss:.4f}')
print(f'Average bias: {lr_avg_bias:.4f}')
print(f'Average variance: {lr_avg_var:.4f}\n')

# --- KNN ---
print("k-NN:")
knn_avg_expected_loss, knn_avg_bias, knn_avg_var = bias_variance_decomp(
    knn_model,
    X_train_scaled, 
    y_train_encoded, 
    X_test_scaled, 
    y_test_encoded, 
    loss='0-1_loss',
    num_rounds=100,
    random_seed=40
)
print(f'Average expected loss: {knn_avg_expected_loss:.4f}')
print(f'Average bias: {knn_avg_bias:.4f}')
print(f'Average variance: {knn_avg_var:.4f}\n')

# --- Decision Tree ---
print("Decision Tree:")
dt_avg_expected_loss, dt_avg_bias, dt_avg_var = bias_variance_decomp(
    dt_model,
    X_train_scaled, 
    y_train_encoded, 
    X_test_scaled, 
    y_test_encoded, 
    loss='0-1_loss',
    num_rounds=100,
    random_seed=40
)
print(f'Average expected loss: {dt_avg_expected_loss:.4f}')
print(f'Average bias: {dt_avg_bias:.4f}')
print(f'Average variance: {dt_avg_var:.4f}\n')

# Perform cross-validation for Logistic Regression
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=10)
print("Logistic Regression CV Score:", lr_cv_scores.mean())

# Perform cross-validation for KNN
knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=10)
print("k-NN CV Score:", knn_cv_scores.mean())

# Perform cross-validation for Decision Tree
dt_cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=10)
print("Decision Tree CV Score:", dt_cv_scores.mean())

# Confusion matrix of LR
lr_cm = confusion_matrix(y_test, lr_predictions)

# Plot confusion matrix for LR
plt.figure(figsize=(15, 15))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=lr_model.classes_, yticklabels=lr_model.classes_)
plt.title('LR Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of kNN
knn_cm = confusion_matrix(y_test, knn_predictions)

# Plot confusion matrix for KNN
plt.figure(figsize=(15, 15))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title('k-NN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of DT
dt_cm = confusion_matrix(y_test, dt_predictions)

# Plot confusion matrix for DT
plt.figure(figsize=(15, 15))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
plt.title('DT Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')