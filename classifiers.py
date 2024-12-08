import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from mlxtend.evaluate import bias_variance_decomp
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize  # Used to convert multi-class labels to binary format for ROC curve analysis
from sklearn.multiclass import OneVsRestClassifier  # Strategy for multi-class classification by treating each class as a separate binary classification problem
from sklearn.metrics import roc_curve, auc  # Functions for calculating ROC curve and area under the curve (AUC)
from sklearn.impute import SimpleImputer # Used to impute NaN values with the mean
from sklearn.inspection import permutation_importance # Import the permutation_importance function for kNN and SVM

# Load the datasets
df_accidents = pd.read_csv('C:/Users/umarb/OneDrive/Documents/BS in CSE/3rd Year/1st Semester/CSE445/fatalaccidentdata.csv')
df_casualties = pd.read_csv('C:/Users/umarb/OneDrive/Documents/BS in CSE/3rd Year/1st Semester/CSE445/fatalcasualtydata.csv')

# Dropping rows with missing values in Index column
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

df.dropna(subset=['Fatal_Casualty_Sex_Binary'], inplace=True)

# Select features and target variable
X = df[['Month_of_Accident', 'Hour_of_Accident', 'Longitude', 'Latitude', 'Pedestrian_Casualties', 'Pedal_Cycles', 'Motor_Cycles', 'Cars', 'Buses_or_Coaches', 'Vans', 'HGVs', 'Other_Vehicles', 'Total_Vehicles_Involved', 'Fatal_Casualties', 'Serious_Casualties', 'Slight_Casualties', 'Total_Number_of_Casualties', 'Fatal_Casualty_Sex_Binary', 'Fatal_Casualty_Age']]
y = df['Fatal_Casualty_Type']

# One-hot encode the `Month_of_Accident` column for X
X = pd.get_dummies(X, columns=['Month_of_Accident'])

# Impute missing values BEFORE applying SMOTE
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Impute missing values in X

# Apply SMOTE to balance the classes in the dataset
#smote = SMOTE(random_state=40, k_neighbors=1)
#X_resampled, y_resampled = smote.fit_resample(X_imputed, y)  # Use X_imputed

# Split data into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, random_state=40, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Correlation Matrix
# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Columns with Fatal Casualty Type')
plt.show()

# Check class distribution
print("Class distribution in dataset before SMOTE:")
print(pd.Series(y).value_counts())

# Check class distribution
print("Class distribution in training set before SMOTE:")
print(pd.Series(y_train).value_counts())

# Check class distribution
print("Class distribution in testing set before SMOTE:")
print(pd.Series(y_test).value_counts())

# Get the frequency of each casualty type
casualty_counts = df['Fatal_Casualty_Type'].value_counts()

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title('Target Class Distribution before SMOTE')
plt.xlabel('Fatal Casualty Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Initialize Logistic Regression classifier
lr_model = LogisticRegression(random_state=40, class_weight='balanced', max_iter=1000, solver='newton-cg')
lr_model.fit(X_train_scaled, y_train)

# Predictions for LR
lr_predictions = lr_model.predict(X_test_scaled)

# Initialize and train KNN classifier
knn_model = KNeighborsClassifier(3)  # You can adjust n_neighbors
knn_model.fit(X_train_scaled, y_train)

# Predictions for KNN
knn_predictions = knn_model.predict(X_test_scaled)

# Initialize and train Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=40, class_weight='balanced')
dt_model.fit(X_train_scaled, y_train)

#Predictions for DT
dt_predictions = dt_model.predict(X_test_scaled)

# Initialize and train Random Forest classifiers
rf_model = RandomForestClassifier(random_state=40, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)

#Predictions for RF
rf_predictions = rf_model.predict(X_test_scaled)

# Initialize SVM classifier with an RBF kernel
svm_model = SVC(kernel='rbf', random_state=40, class_weight='balanced') 
svm_model.fit(X_train_scaled, y_train)

# Predictions for SVM
svm_predictions = svm_model.predict(X_test_scaled)

# Feature Importance

# --- Logistic Regression ---
# Note: Feature importance for Logistic Regression is different
# We'll use coefficients as a proxy for importance
lr_importances = np.abs(lr_model.coef_[0])  # Take absolute values of coefficients
lr_indices = np.argsort(lr_importances)[::-1]
plt.figure(figsize=(10, 8))
plt.title("Feature Importance (Logistic Regression)")
plt.bar(range(X_train_scaled.shape[1]), lr_importances[lr_indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), X.columns[lr_indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- k-NN ---
# Note: k-NN doesn't have a direct feature importance measure like some tree-based models.
# We can use permutation importance as a workaround to estimate feature importance.

# Calculate permutation importance for the k-NN model
# `permutation_importance`:  This function randomly shuffles the values of each feature 
#                            and measures how much the model's performance decreases.
#                            A feature is considered important if shuffling its values leads to a significant drop in performance.
# `knn_model`: The trained k-NN model.
# `X_train_scaled`: The scaled training data.
# `y_train`: The training labels.
# `n_repeats=10`: The number of times the permutation is repeated for each feature.
# `random_state=40`: A seed for the random number generator to ensure reproducibility.
knn_result = permutation_importance(knn_model, X_train_scaled, y_train, n_repeats=10, random_state=40)

# Get the mean importance scores from the permutation importance result
knn_importances = knn_result.importances_mean  

# Get indices that would sort the importance scores in descending order
knn_indices = np.argsort(knn_importances)[::-1]  

# Plotting
plt.figure(figsize=(10, 8))  # Set the figure size for the plot
plt.title("Feature Importance (k-NN)")  # Set the title of the plot

# Create a bar plot of feature importances
# `range(X_train_scaled.shape[1])`:  Provides the x-axis positions for the bars (number of features).
# `knn_importances[knn_indices]`: The feature importance values sorted in descending order.
# `align="center"`: Aligns the bars to the center of their x-axis positions.
plt.bar(range(X_train_scaled.shape[1]), knn_importances[knn_indices], align="center")  

# Set x-axis labels with feature names (rotated for better readability)
plt.xticks(range(X_train_scaled.shape[1]), X.columns[knn_indices], rotation=45, ha='right')  

plt.tight_layout()  # Adjust the layout to prevent labels from overlapping
plt.show()  # Display the plot

# --- Decision Tree ---
dt_importances = dt_model.feature_importances_
dt_indices = np.argsort(dt_importances)[::-1]
plt.figure(figsize=(10, 8))
plt.title("Feature Importance (Decision Tree)")
plt.bar(range(X_train_scaled.shape[1]), dt_importances[dt_indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), X.columns[dt_indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# --- Random Forest ---
rf_importances = rf_model.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]
plt.figure(figsize=(10, 8))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X_train_scaled.shape[1]), rf_importances[rf_indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), X.columns[rf_indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Calculate permutation feature importance
result = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=10, random_state=40)
svm_importances = result.importances_mean

# Sort feature importances
svm_indices = np.argsort(svm_importances)[::-1]

# Plotting
plt.figure(figsize=(10, 8))
plt.title("Feature Importance (SVM with RBF Kernel)")
plt.bar(range(X_test_scaled.shape[1]), svm_importances[svm_indices], align="center")
plt.xticks(range(X_test_scaled.shape[1]), X.columns[svm_indices], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Print accuracy and classification report for LR
print("Logistic Regression Classifier Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions, zero_division=0))

# Print accuracy and classification report for KNN
print("k-NN Classifier Accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions, zero_division=0))

# Print accuracy and classification report for DT
print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions, zero_division=0))

# Print accuracy and classification report for RF
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions, zero_division=0))

# Print accuracy and classification report for SVM (RBF)
print("SVM (RBF) Classifier Accuracy:", accuracy_score(y_test, svm_predictions))
print(classification_report(y_test, svm_predictions, zero_division=0))