"""
Support Vector Regression (SVR)
================================
Dataset: datasets/Position_Salaries.csv
Features: Level -> Salary

Note: SVR is sensitive to feature scale — ALWAYS apply StandardScaler.
      Also, y must be scaled (SVR uses it internally in kernel calculations).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("datasets/Position_Salaries.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values.reshape(-1, 1)  # reshape for scaler

# Feature scaling (required for SVR)
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y).ravel()

# Train SVR (RBF kernel)
model = SVR(kernel="rbf")
model.fit(X_scaled, y_scaled)

# Predict for level 6.5
level = np.array([[6.5]])
level_scaled = sc_X.transform(level)
pred_scaled = model.predict(level_scaled).reshape(-1, 1)
pred_salary = sc_y.inverse_transform(pred_scaled)
print(f"SVR prediction for level 6.5: ${pred_salary[0][0]:,.2f}")

# Visualise
X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
X_grid_scaled = sc_X.transform(X_grid)
y_grid_pred = sc_y.inverse_transform(model.predict(X_grid_scaled).reshape(-1, 1))

plt.scatter(X, sc_y.inverse_transform(y), color="red", label="Actual")
plt.plot(X_grid, y_grid_pred, color="blue", label="SVR (RBF)")
plt.title("SVR — Position vs Salary")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("svr.png", dpi=100)
plt.show()
print("Plot saved: svr.png")
