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

# Get the target class distribution
casualty_counts = df['Fatal_Casualty_Type'].value_counts()

# Plotting before SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title('Target Class Distribution before SMOTE')
plt.xlabel('Fatal Casualty Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plotting after SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y_resampled)
plt.title('Target Class Distribution after SMOTE')
plt.xlabel('Fatal Casualty Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Correlation Matrix
# Select only numeric columns for correlation analysis
numeric_df = df.select_dtypes(include=[np.number])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Columns with Fatal Casualty Type')
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

# --- k-NN ---
# Note: k-NN doesn't have a direct feature importance measure
# We can use permutation importance as a workaround
from sklearn.inspection import permutation_importance
knn_result = permutation_importance(knn_model, X_train_scaled, y_train, n_repeats=10, random_state=40)
knn_importances = knn_result.importances_mean
knn_indices = np.argsort(knn_importances)[::-1]
plt.figure(figsize=(10, 8))
plt.title("Feature Importance (k-NN)")
plt.bar(range(X_train_scaled.shape[1]), knn_importances[knn_indices], align="center")
plt.xticks(range(X_train_scaled.shape[1]), X.columns[knn_indices], rotation=45, ha='right')
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

models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest']
accuracies = [accuracy_score(y_test, lr_predictions), 
              accuracy_score(y_test, knn_predictions), 
              accuracy_score(y_test, dt_predictions),
              accuracy_score(y_test, rf_predictions)]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim([0, 1])  # Set y-axis limits for better comparison
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

plt.tight_layout()
plt.show()

# Bar plot of Correct:Incorrect ratios of the Confusion Matrices
classifiers = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest']
ratios = [10.67, 21.02, 20.39, 24.85]

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

# Bar plot of Cross-Validation Scores
cv_scores = [lr_cv_scores.mean(), knn_cv_scores.mean(), dt_cv_scores.mean(), rf_cv_scores.mean()]

plt.figure(figsize=(8, 6))
plt.bar(models, cv_scores)
plt.title('Cross-Validation Score Comparison')
plt.ylabel('CV Score')
plt.ylim([0, 1])  # Set y-axis limits for better comparison
plt.show()

#---ROC Curves---
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

# Binarize the target variable
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Create OneVsRestClassifier for each model
lr_ovr = OneVsRestClassifier(lr_model)
knn_ovr = OneVsRestClassifier(knn_model)
dt_ovr = OneVsRestClassifier(dt_model)
rf_ovr = OneVsRestClassifier(rf_model)

# Fit the classifiers
lr_ovr.fit(X_train_scaled, y_train)
knn_ovr.fit(X_train_scaled, y_train)
dt_ovr.fit(X_train_scaled, y_train)
rf_ovr.fit(X_train_scaled, y_train)

# Get predicted probabilities for each class
lr_probs = lr_ovr.predict_proba(X_test_scaled)
knn_probs = knn_ovr.predict_proba(X_test_scaled)
dt_probs = dt_ovr.predict_proba(X_test_scaled)
rf_probs = rf_ovr.predict_proba(X_test_scaled)

# --- Plot ROC curves for each model ---

models = {
    "Logistic Regression": lr_probs,
    "k-NN": knn_probs,
    "Decision Tree": dt_probs,
    "Random Forest": rf_probs
}

plt.figure(figsize=(10, 8))

for model_name, model_probs in models.items():
    # Check for NaN values and impute if necessary
    if np.isnan(model_probs).any():
        print(f"Warning: NaN values found in {model_name} predictions. Imputing with mean.")
        imputer = SimpleImputer(strategy='mean')
        model_probs = imputer.fit_transform(model_probs)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate ROC metrics for micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), model_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot micro-average ROC curve
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'{model_name} (micro-average AUC = {roc_auc["micro"]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison (Micro-Average)')
plt.legend(loc="lower right")
plt.show()

# --- ROC AUC Scores ---
# --- Logistic Regression ---
lr_probs = lr_model.predict_proba(X_test_scaled)
lr_auc = roc_auc_score(y_test, lr_probs, multi_class='ovr')  # Use 'ovr' for multi-class
print(f"Logistic Regression ROC AUC: {lr_auc:.4f}")

# --- k-NN ---
knn_probs = knn_model.predict_proba(X_test_scaled)
knn_auc = roc_auc_score(y_test, knn_probs, multi_class='ovr')
print(f"k-NN ROC AUC: {knn_auc:.4f}")

# --- Decision Tree ---
dt_probs = dt_model.predict_proba(X_test_scaled)
dt_auc = roc_auc_score(y_test, dt_probs, multi_class='ovr')
print(f"Decision Tree ROC AUC: {dt_auc:.4f}")

# --- Random Forest ---
rf_probs = rf_model.predict_proba(X_test_scaled)
rf_auc = roc_auc_score(y_test, rf_probs, multi_class='ovr')
print(f"Random Forest ROC AUC: {rf_auc:.4f}")

# Bar plot of ROC-AUC scores
models = ['Logistic Regression', 'k-NN', 'Decision Tree', 'Random Forest']
auc_scores = [lr_auc, knn_auc, dt_auc, rf_auc]

plt.figure(figsize=(8, 6))
plt.bar(models, auc_scores)
plt.title('ROC-AUC Comparison')
plt.ylabel('ROC-AUC')
plt.ylim([0, 1])  # Set y-axis limits for better comparison
plt.show()

#---Bias-Variance Decomposition---
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

plt.tight_layout()
plt.show()