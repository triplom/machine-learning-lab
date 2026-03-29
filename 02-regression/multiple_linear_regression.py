"""
Multiple Linear Regression
==========================
Dataset: datasets/50_Startups.csv
Features: R&D Spend, Administration, Marketing Spend, State
Target: Profit

Steps:
  1. Encode categorical column (State)
  2. Avoid dummy variable trap (drop='first')
  3. Fit linear model
  4. Evaluate and inspect coefficients
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("datasets/50_Startups.csv")
print(df.head())
print()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode State column (index 3)
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), [3])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Compare predictions vs actuals
comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred.round(2)})
print("Predictions vs Actuals:")
print(comparison.to_string(index=False))
print()

# Evaluate
print(f"R² (test): {r2_score(y_test, y_pred):.4f}")
print(f"RMSE (test): {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}")
