"""
Polynomial Regression
=====================
Dataset: datasets/Position_Salaries.csv
Features: Level (1-10) -> Salary

Why: salary vs level is non-linear — polynomial terms improve the fit.

Compare: Linear vs Polynomial regression visually.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("datasets/Position_Salaries.csv")
print(df)
X = df.iloc[:, 1:2].values  # Level (keep as 2D)
y = df.iloc[:, -1].values  # Salary

# --- Linear regression (baseline) ---
lin_model = LinearRegression()
lin_model.fit(X, y)

# --- Polynomial regression (degree 4) ---
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Visualise
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color="red")
plt.plot(X, lin_model.predict(X), color="blue")
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")

plt.subplot(1, 2, 2)
plt.scatter(X, y, color="red")
plt.plot(X_grid, poly_model.predict(poly.transform(X_grid)), color="blue")
plt.title("Polynomial Regression (degree=4)")
plt.xlabel("Level")
plt.ylabel("Salary")

plt.tight_layout()
plt.savefig("polynomial_regression.png", dpi=100)
plt.show()

# Predict salary for level 6.5
level = np.array([[6.5]])
print(f"\nPrediction for level 6.5:")
print(f"  Linear:     ${lin_model.predict(level)[0]:,.2f}")
print(f"  Polynomial: ${poly_model.predict(poly.transform(level))[0]:,.2f}")

# R² on training data
print(f"\nR² (linear):     {r2_score(y, lin_model.predict(X)):.4f}")
print(f"R² (polynomial): {r2_score(y, poly_model.predict(X_poly)):.4f}")
