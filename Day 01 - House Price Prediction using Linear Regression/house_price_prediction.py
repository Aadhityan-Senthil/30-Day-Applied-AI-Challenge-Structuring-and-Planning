# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV

# Load Dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Price'] = data.target  

# Step 1: Feature Selection using Correlation Analysis
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Select top 5 features based on correlation
selected_features = ["MedInc", "AveRooms", "HouseAge", "AveBedrms", "Latitude"]
X = df[selected_features]
y = df['Price']

# Step 2: Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Adding Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Step 4: Train Model with Cross-Validation
model = LinearRegression()
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Average R² Score: {cv_scores.mean():.4f}")

# Step 5: Hyperparameter Tuning (GridSearchCV)
param_grid = {'fit_intercept': [True, False]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Train Best Model
best_model.fit(X_train, y_train)

# Make Predictions
y_pred = best_model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Performance Metrics
print(f"Best Model Params: {grid_search.best_params_}")
print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, R² Score: {r2:.4f}")

# Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.show()

#Model coeffiecent and Intercept
#Model coeffiecent and Intercepts
feature_names = poly.get_feature_names_out(selected_features)
print("Model Coefficients:")
for feature, coef in zip(feature_names, best_model.coef_):
    print(f"{feature}: {coef:.4f}")

print(f"\nIntercept: {best_model.intercept_:.4f}")