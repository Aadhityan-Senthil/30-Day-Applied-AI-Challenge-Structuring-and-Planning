import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Exploratory Data Analysis (EDA)
def basic_eda(data):
    print("\nFirst 5 rows:\n", data.head())
    print("\nDataset Info:\n", data.info())
    print("\nSummary Statistics:\n", data.describe())
    print("\nMissing Values:\n", data.isnull().sum())

basic_eda(df)

# Handle missing values
df.dropna(inplace=True)

# Apply K-Means Clustering to generate 'Segment' column
features = df[['Age', 'Annual Income', 'Spending Score']]
kmeans = KMeans(n_clusters=4, random_state=42)
df['Segmentation'] = kmeans.fit_predict(features)

# Define Features and Target
X = df[['Age', 'Annual Income', 'Spending Score']]
y = df['Segmentation']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 4, 6, 8]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Model Selection
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}\n")

# Cross-Validation Score
cv_score = cross_val_score(best_model, X_train, y_train, cv=5).mean()
print(f"Cross-Validation Accuracy: {cv_score:.4f}\n")

# Model Training
best_model.fit(X_train, y_train)

# Predictions
y_pred = best_model.predict(X_test)

#Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Feature Importance
feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': best_model.feature_importances_})
feat_importance = feat_importance.sort_values(by="Importance", ascending=False)
print("\n🔹 Feature Importance:\n", feat_importance)

# Visualize Decision Tree
plt.figure(figsize=(12,8))
plot_tree(best_model, feature_names=X.columns, class_names=[str(i) for i in y.unique()], filled=True)
plt.title("Decision Tree Visualization")
plt.show()
