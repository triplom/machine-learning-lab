"""
Random Forest Regression
=========================
Dataset: datasets/Position_Salaries.csv
Features: Level -> Salary

Random Forest = ensemble of decision trees. Averages predictions for lower variance.
n_estimators controls the number of trees.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("datasets/Position_Salaries.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

# Train model
model = RandomForestRegressor(n_estimators=300, random_state=0)
model.fit(X, y)

# Predict for level 6.5
level = np.array([[6.5]])
print(f"Random Forest prediction for level 6.5: ${model.predict(level)[0]:,.2f}")

# Feature importance
print(f"Feature importance: {model.feature_importances_}")

# Visualise (fine grid)
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)

plt.scatter(X, y, color="red", label="Actual")
plt.plot(X_grid, model.predict(X_grid), color="blue", label="RF (n=300)")
plt.title("Random Forest Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("random_forest_regression.png", dpi=100)
plt.show()
print("Plot saved: random_forest_regression.png")
