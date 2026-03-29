"""
Simple Linear Regression
========================
Dataset: datasets/Salary_Data.csv
Features: YearsExperience -> Salary

Goal: Fit y = b0 + b1*x and visualise the regression line.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("datasets/Salary_Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(f"Intercept (b0): {model.intercept_:.2f}")
print(f"Coefficient (b1): {model.coef_[0]:.2f}")
print(f"R² (test): {r2_score(y_test, y_pred):.4f}")
print(f"RMSE (test): {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# Visualise training set
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color="red", label="Actual")
plt.plot(X_train, model.predict(X_train), color="blue", label="Predicted")
plt.title("Training Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()

# Visualise test set
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color="red", label="Actual")
plt.plot(X_train, model.predict(X_train), color="blue", label="Regression line")
plt.title("Test Set")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("simple_linear_regression.png", dpi=100)
plt.show()
print("Plot saved: simple_linear_regression.png")
