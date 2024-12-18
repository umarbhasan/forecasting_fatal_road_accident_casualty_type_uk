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

models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest', 'SVM']
accuracies = [accuracy_score(y_test, lr_predictions), 
              accuracy_score(y_test, knn_predictions), 
              accuracy_score(y_test, dt_predictions),
              accuracy_score(y_test, rf_predictions),
              accuracy_score(y_test, svm_predictions)]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies)
plt.title('Model Accuracy Comparison Before SMOTE')
plt.ylabel('Accuracy')
plt.ylim([0, 1])  # Set y-axis limits for better comparison
plt.show()

#--Confusion Matrices before SMOTE--
# Confusion matrix of LR
lr_cm = confusion_matrix(y_test, lr_predictions)

# Plot confusion matrix for LR
plt.figure(figsize=(12, 8))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=lr_model.classes_, yticklabels=lr_model.classes_)
plt.title('Logistic Regression Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of kNN
knn_cm = confusion_matrix(y_test, knn_predictions)

# Plot confusion matrix for KNN
plt.figure(figsize=(12, 8))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title('k-NN Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of DT
dt_cm = confusion_matrix(y_test, dt_predictions)

# Plot confusion matrix for DT
plt.figure(figsize=(12, 8))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
plt.title('Decision Tree Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of RF
rf_cm = confusion_matrix(y_test, rf_predictions)

# Plot confusion matrix for RF
plt.figure(figsize=(12, 8))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Random Forest Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix for SVM (RBF)
svm_cm = confusion_matrix(y_test, svm_predictions)

# Plot confusion matrix for SVM (RBF)
plt.figure(figsize=(12, 8))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_) 
plt.title('SVM (RBF) Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# Perform cross-validation for Logistic Regression
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=10)
print("Logistic Regression Classifier CV Score:", lr_cv_scores.mean())
# Perform cross-validation for KNN
knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=10)
print("k-NN Classifier CV Score:", knn_cv_scores.mean())
# Perform cross-validation for Decision Tree
dt_cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=10)
print("Decision Tree Classifier CV Score:", dt_cv_scores.mean())
# Perform cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=10)
print("Random Forest Classifier CV Score:", rf_cv_scores.mean())
# Perform cross-validation for SVM (RBF)
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=10) 
print("SVM (RBF) Classifier CV Score:", svm_cv_scores.mean())

# Create a list of model names and their corresponding CV scores
models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest', 'SVM (RBF)']
cv_scores = [lr_cv_scores, knn_cv_scores, dt_cv_scores, rf_cv_scores, svm_cv_scores]

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores, tick_labels=models)
plt.title('Cross-Validation Scores Comparison Before SMOTE')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#----------AFTER SMOTE------------
# Apply SMOTE to balance the classes in the dataset
smote = SMOTE(random_state=40, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)  # Use X_imputed

# Split data into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=40, stratify=y_resampled)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check class distribution
print("Class distribution in dataset after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Check class distribution
print("Class distribution in training set after SMOTE:")
print(pd.Series(y_train).value_counts())

# Check class distribution
print("Class distribution in testing set after SMOTE:")
print(pd.Series(y_test).value_counts())

# Plotting
plt.figure(figsize=(10, 6))
sns.countplot(x=y_resampled)
plt.title('Target Class Distribution after SMOTE')
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

models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest', 'SVM']
accuracies = [accuracy_score(y_test, lr_predictions), 
              accuracy_score(y_test, knn_predictions), 
              accuracy_score(y_test, dt_predictions),
              accuracy_score(y_test, rf_predictions),
              accuracy_score(y_test, svm_predictions)]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies)
plt.title('Model Accuracy Comparison After SMOTE')
plt.ylabel('Accuracy')
plt.ylim([0, 1])  # Set y-axis limits for better comparison
plt.show()

# Categories (same for all models)
categories = ['Bus_Driver', 'Bus_Passenger', 'Car_Driver', 'Car_Passenger', 'HGV_Driver', 
              'HGV_Passenger', 'Motor_Cycle_Passenger', 'Motor_Cycle_Rider', 'Other_Vehicle_Occupant',
              'Pedal_Cyclist', 'Pedestrian', 'Van_Driver', 'Van_Passenger']

# F1-scores from the classification reports
lr_scores = [1.00, 0.99, 0.66, 0.69, 0.94, 0.99, 0.94, 0.94, 0.99, 1.00, 0.99, 0.88, 0.86]
knn_scores = [1.00, 1.00, 0.72, 0.75, 0.99, 1.00, 0.98, 0.98, 0.99, 1.00, 1.00, 1.00, 1.00]
dt_scores = [1.00, 1.00, 0.73, 0.76, 0.98, 1.00, 0.98, 0.98, 0.99, 1.00, 1.00, 0.98, 0.99]
rf_scores = [1.00, 1.00, 0.77, 0.78, 0.99, 1.00, 0.99, 0.98, 0.99, 1.00, 1.00, 0.99, 1.00]
svm_scores = [1.00, 1.00, 0.77, 0.78, 0.98, 1.00, 0.98, 0.98, 0.99, 1.00, 1.00, 0.99, 1.00] # Added SVM scores

# Set up the plot
bar_width = 0.15  # Reduced bar width to accommodate 5 models
index = np.arange(len(categories))

fig, ax = plt.subplots(figsize=(16, 8)) 

# Create bars for each model (including SVM)
rects1 = ax.bar(index, lr_scores, bar_width, color='blue', label='LR')
rects2 = ax.bar(index + bar_width, knn_scores, bar_width, color='orange', label='K-NN')
rects3 = ax.bar(index + 2 * bar_width, dt_scores, bar_width, color='green', label='DT')
rects4 = ax.bar(index + 3 * bar_width, rf_scores, bar_width, color='red', label='RF')
rects5 = ax.bar(index + 4 * bar_width, svm_scores, bar_width, color='purple', label='SVM') # Added SVM bars

# Set chart labels and title
ax.set_xlabel('Fatal Casualty Type')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Scores for Different Fatal Casualty Types After SMOTE')
ax.set_xticks(index + 2 * bar_width)  # Adjusted x-ticks for 5 models
ax.set_xticklabels(categories, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.show()

# Confusion matrix of LR
lr_cm = confusion_matrix(y_test, lr_predictions)

# Plot confusion matrix for LR
plt.figure(figsize=(12, 8))
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=lr_model.classes_, yticklabels=lr_model.classes_)
plt.title('Logistic Regression Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of kNN
knn_cm = confusion_matrix(y_test, knn_predictions)

# Plot confusion matrix for KNN
plt.figure(figsize=(12, 8))
sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.title('k-NN Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of DT
dt_cm = confusion_matrix(y_test, dt_predictions)

# Plot confusion matrix for DT
plt.figure(figsize=(12, 8))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
plt.title('Decision Tree Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix of RF
rf_cm = confusion_matrix(y_test, rf_predictions)

# Plot confusion matrix for RF
plt.figure(figsize=(12, 8))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Random Forest Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

# Confusion matrix for SVM (RBF)
svm_cm = confusion_matrix(y_test, svm_predictions)

# Plot confusion matrix for SVM (RBF)
plt.figure(figsize=(12, 8))
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_) 
plt.title('SVM (RBF) Classifier Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()

# Data for the bar plot
classifiers = ['Logistic Regression', 'kNN', 'Decision Tree', 'Random Forest', 'SVM']
ratios = [10.67, 21.02, 20.39, 24.85, 23.98]

# Create the bar plot
plt.figure(figsize=(10, 6)) 
plt.bar(classifiers, ratios)

# Add labels and title
plt.xlabel('Classifier')
plt.ylabel('Ratio of Correct to Incorrect Classifications')
plt.title('Classifier Performance based on Confusion Matrices')

# Show the plot
plt.show()

# Perform cross-validation for Logistic Regression
lr_cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=10)
print("Logistic Regression Classifier CV Score:", lr_cv_scores.mean())
# Perform cross-validation for KNN
knn_cv_scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=10)
print("k-NN Classifier CV Score:", knn_cv_scores.mean())
# Perform cross-validation for Decision Tree
dt_cv_scores = cross_val_score(dt_model, X_train_scaled, y_train, cv=10)
print("Decision Tree Classifier CV Score:", dt_cv_scores.mean())
# Perform cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=10)
print("Random Forest Classifier CV Score:", rf_cv_scores.mean())
# Perform cross-validation for SVM (RBF)
svm_cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=10) 
print("SVM (RBF) Classifier CV Score:", svm_cv_scores.mean())

# Create a list of model names and their corresponding CV scores
models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest', 'SVM (RBF)']
cv_scores = [lr_cv_scores, knn_cv_scores, dt_cv_scores, rf_cv_scores, svm_cv_scores]

# Create the box plot
plt.figure(figsize=(10, 6))
plt.boxplot(cv_scores, tick_labels=models)
plt.title('Cross-Validation Scores Comparison After SMOTE')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Data Preprocessing ---

# Binarize the target variable (y_test)
# This converts the multi-class labels into a binary format (one-hot encoding) required for ROC curve analysis
# `classes=np.unique(y_test)` ensures that the binarization considers all unique classes in the target variable
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))  
n_classes = y_test_bin.shape[1]  # Get the number of classes

# --- Model Preparation ---

# Create OneVsRestClassifier for each model
# OneVsRestClassifier is a strategy that fits one classifier per class. 
# For each classifier, the class is fitted against all the other classes.
lr_ovr = OneVsRestClassifier(lr_model)  # Logistic Regression
knn_ovr = OneVsRestClassifier(knn_model)  # k-Nearest Neighbors
dt_ovr = OneVsRestClassifier(dt_model)  # Decision Tree
rf_ovr = OneVsRestClassifier(rf_model)  # Random Forest
# Add SVM (RBF) to OneVsRestClassifier
svm_ovr = OneVsRestClassifier(SVC(kernel='rbf', random_state=40, class_weight='balanced', probability=True))

# --- Model Training ---

# Fit the classifiers on the scaled training data (X_train_scaled) and training labels (y_train)
lr_ovr.fit(X_train_scaled, y_train)  
knn_ovr.fit(X_train_scaled, y_train)
dt_ovr.fit(X_train_scaled, y_train)
rf_ovr.fit(X_train_scaled, y_train)
svm_ovr.fit(X_train_scaled, y_train)

# --- Prediction ---

# Get predicted probabilities for each class for the scaled test data (X_test_scaled)
lr_probs = lr_ovr.predict_proba(X_test_scaled)
knn_probs = knn_ovr.predict_proba(X_test_scaled)
dt_probs = dt_ovr.predict_proba(X_test_scaled)
rf_probs = rf_ovr.predict_proba(X_test_scaled)
svm_probs = svm_ovr.predict_proba(X_test_scaled)

# --- ROC Curve Plotting ---

# Store models and their predicted probabilities in a dictionary for easy iteration
models = {
    "Logistic Regression": lr_probs,
    "k-NN": knn_probs,
    "Decision Tree": dt_probs,
    "Random Forest": rf_probs,
    "SVM (RBF)": svm_probs  # Add SVM
}

plt.figure(figsize=(10, 8))  # Set the figure size for the plot

# Iterate through each model and its predicted probabilities
for model_name, model_probs in models.items():
    # Check for NaN values in predicted probabilities and impute with the mean if found
    if np.isnan(model_probs).any():
        print(f"Warning: NaN values found in {model_name} predictions. Imputing with mean.")
        imputer = SimpleImputer(strategy='mean')
        model_probs = imputer.fit_transform(model_probs)

    # Compute ROC curve and ROC area for each class
    fpr = dict()  # Dictionary to store false positive rates for each class
    tpr = dict()  # Dictionary to store true positive rates for each class
    roc_auc = dict()  # Dictionary to store AUC values for each class

    for i in range(n_classes):
        # Calculate ROC curve (fpr, tpr) and AUC (roc_auc) for each class
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    # Micro-averaging aggregates the contributions of all classes to compute the average metric
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), model_probs.ravel())  # Ravel flattens the arrays
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot the micro-average ROC curve for the current model
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{model_name} (micro-average AUC = {roc_auc["micro"]:0.2f})')

# Plot the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])  # Set x-axis limits
plt.ylim([0.0, 1.05])  # Set y-axis limits
plt.xlabel('False Positive Rate')  # Set x-axis label
plt.ylabel('True Positive Rate')  # Set y-axis label
plt.title('ROC Curve Comparison (Micro-Average) After SMOTE')  # Set plot title
plt.legend(loc="lower right")  # Place the legend in the lower right corner
plt.show()  # Show the plot

# --- Logistic Regression ---

# Predict probabilities for the test set using the trained Logistic Regression model
lr_probs = lr_model.predict_proba(X_test_scaled)  

# Calculate the ROC AUC score for Logistic Regression
# `roc_auc_score`:  Calculates the area under the receiver operating characteristic curve (ROC AUC).
# `multi_class='ovr'`: Specifies the 'one-vs-rest' strategy for handling multi-class ROC AUC calculation.
lr_auc = roc_auc_score(y_test, lr_probs, multi_class='ovr')  

# Print the ROC AUC score for Logistic Regression (formatted to 4 decimal places)
print(f"Logistic Regression ROC AUC: {lr_auc:.4f}")  

# --- k-NN ---

# Predict probabilities for the test set using the trained k-NN model
knn_probs = knn_model.predict_proba(X_test_scaled)  

# Calculate the ROC AUC score for k-NN
knn_auc = roc_auc_score(y_test, knn_probs, multi_class='ovr')  

# Print the ROC AUC score for k-NN
print(f"k-NN ROC AUC: {knn_auc:.4f}")  

# --- Decision Tree ---

# Predict probabilities for the test set using the trained Decision Tree model
dt_probs = dt_model.predict_proba(X_test_scaled)  

# Calculate the ROC AUC score for Decision Tree
dt_auc = roc_auc_score(y_test, dt_probs, multi_class='ovr')  

# Print the ROC AUC score for Decision Tree
print(f"Decision Tree ROC AUC: {dt_auc:.4f}")  

# --- Random Forest ---

# Predict probabilities for the test set using the trained Random Forest model
rf_probs = rf_model.predict_proba(X_test_scaled)  

# Calculate the ROC AUC score for Random Forest
rf_auc = roc_auc_score(y_test, rf_probs, multi_class='ovr')  

# Print the ROC AUC score for Random Forest
print(f"Random Forest ROC AUC: {rf_auc:.4f}")

# ... your existing code ...

# --- SVM (RBF) ---

# Ensure your SVM model is initialized with probability=True
svm_model = SVC(kernel='rbf', random_state=40, class_weight='balanced', probability=True) 

# Fit the SVM model to your training data (this was missing)
svm_model.fit(X_train_scaled, y_train)  

# Predict probabilities for the test set using the trained SVM (RBF) model
svm_probs = svm_model.predict_proba(X_test_scaled)  

# ... (rest of your existing code) ...

# ... (rest of your existing code) ...
# Calculate the ROC AUC score for SVM (RBF)
svm_auc = roc_auc_score(y_test, svm_probs, multi_class='ovr')

# Print the ROC AUC score for SVM (RBF)
print(f"SVM (RBF) ROC AUC: {svm_auc:.4f}")

models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest', 'SVM']
auc_scores = [lr_auc, knn_auc, dt_auc, rf_auc, svm_auc]

plt.figure(figsize=(8, 6))
plt.bar(models, auc_scores)
plt.title('ROC-AUC Comparison After SMOTE')
plt.ylabel('ROC-AUC')
plt.ylim([0, 1])  # Set y-axis limits for better comparison
plt.show()

# Encode the target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# --- Logistic Regression ---
print("Logistic Regression:")

# Perform bias-variance decomposition for the Logistic Regression model
# `bias_variance_decomp`: This function estimates the bias and variance of a model.
# `lr_model`: The trained Logistic Regression model.
# `X_train_scaled`: The scaled training data.
# `y_train_encoded`: The encoded training labels.
# `X_test_scaled`: The scaled test data.
# `y_test_encoded`: The encoded test labels.
# `loss='0-1_loss'`: Specifies the 0-1 loss function (classification error) for the calculation.
# `num_rounds=100`:  The number of times the model will be re-trained and evaluated to estimate bias and variance. 
#                    Higher values generally give more stable estimates but increase computation time.
# `random_seed=40`:  A seed for the random number generator to ensure reproducibility of the results.
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

# Print the results of the bias-variance decomposition
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

# --- RF ---
print("RF:")
rf_avg_expected_loss, rf_avg_bias, rf_avg_var = bias_variance_decomp(
    rf_model, 
    X_train_scaled, 
    y_train_encoded, 
    X_test_scaled, 
    y_test_encoded, 
    loss='0-1_loss', 
    num_rounds=100, 
    random_seed=40
)
print(f'Average expected loss: {rf_avg_expected_loss:.4f}')
print(f'Average bias: {rf_avg_bias:.4f}')
print(f'Average variance: {rf_avg_var:.4f}\n')

# --- SVM (RBF) --- 
print("SVM (RBF):")

# Perform bias-variance decomposition for SVM (RBF) with fewer rounds
svm_avg_expected_loss, svm_avg_bias, svm_avg_var = bias_variance_decomp(
    svm_model, 
    X_train_scaled, 
    y_train_encoded,  
    X_test_scaled, 
    y_test_encoded, 
    loss='0-1_loss', 
    num_rounds=100, 
    random_seed=40
)

print(f'Average expected loss: {svm_avg_expected_loss:.4f}')
print(f'Average bias: {svm_avg_bias:.4f}')
print(f'Average variance: {svm_avg_var:.4f}\n')

loss_values = [0.0878, 0.0542, 0.0504, 0.0373, 0.0449]
bias_values = [0.0857, 0.0451, 0.0390, 0.0368, 0.0408]
variance_values = [0.0133, 0.0289, 0.0320, 0.0111, 0.0137]

# You can use a grouped bar chart or separate bar charts for bias and variance

# Example for grouped bar chart:
bar_width = 0.30
index = np.arange(len(models))

plt.figure(figsize=(10, 6))
plt.bar(index, loss_values, bar_width, label='Average Expected Loss')
plt.bar(index+ bar_width, bias_values, bar_width, label='Average Bias')
plt.bar(index + bar_width + bar_width, variance_values, bar_width, label='Average Variance')
plt.xticks(index + bar_width + bar_width / 50, models)
plt.title('Average Expected Loss, Average Bias, and Average Variance Comparison After SMOTE')
plt.ylabel('Error')
plt.legend()
plt.show()