# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("WineQT.csv")

# Display first few rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Convert wine quality into binary classification (Good: 7-10, Bad: 3-6)
df['quality'] = df['quality'].apply(lambda x: 'Good' if x >= 7 else 'Bad')

# Encode labels (Bad = 0, Good = 1)
label_encoder = LabelEncoder()
df['quality'] = label_encoder.fit_transform(df['quality'])

# Split data into features and target
X = df.drop(columns=['quality'])
y = df['quality']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM model
svm_model = SVC(probability=True, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Model Evaluation
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final Model Accuracy after Hyperparameter Tuning: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - SVM for Wine Quality Classification")
plt.show()

# SHAP Explanation Fix
def model_predict(X):
    return best_model.predict_proba(X)

# Use KernelExplainer for SHAP
explainer = shap.KernelExplainer(model_predict, X_train[:100])  # Subset for efficiency
shap_values = explainer.shap_values(X_test[:100])

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test[:100], feature_names=df.drop(columns=['quality']).columns)
