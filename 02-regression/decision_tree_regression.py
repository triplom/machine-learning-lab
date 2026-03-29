"""
Decision Tree Regression
=========================
Dataset: datasets/Position_Salaries.csv
Features: Level -> Salary

Note: Decision trees split at discrete thresholds — plot with fine grid to see steps.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("datasets/Position_Salaries.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1].values

# Train model
model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)

# Predict for level 6.5
level = np.array([[6.5]])
print(f"Decision Tree prediction for level 6.5: ${model.predict(level)[0]:,.2f}")

# Visualise (fine grid reveals staircase pattern)
X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)

plt.scatter(X, y, color="red", label="Actual")
plt.plot(X_grid, model.predict(X_grid), color="blue", label="Decision Tree")
plt.title("Decision Tree Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("decision_tree_regression.png", dpi=100)
plt.show()
print("Plot saved: decision_tree_regression.png")
