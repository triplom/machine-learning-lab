"""
Grid Search — Hyperparameter Tuning
=====================================
Dataset: Social_Network_Ads.csv
Model: Kernel SVM

GridSearchCV exhaustively searches all combinations of hyperparameters
using cross-validation to find the best configuration.

Tip: use RandomizedSearchCV for large hyperparameter spaces.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.stats import loguniform

# Load data
df = pd.read_csv("../03-classification/datasets/Social_Network_Ads.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

sc = StandardScaler()
X = sc.fit_transform(X)

# --- Grid Search ---
param_grid = [
    {"C": [0.1, 1, 10, 100], "kernel": ["linear"]},
    {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf"],
        "gamma": [0.01, 0.1, 1, "scale", "auto"],
    },
]

grid_search = GridSearchCV(
    SVC(random_state=0), param_grid, cv=10, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_search.fit(X, y)

print("=== Grid Search Results ===")
print(f"Best params:    {grid_search.best_params_}")
print(f"Best CV score:  {grid_search.best_score_:.4f}")

# Top 5 results
results = pd.DataFrame(grid_search.cv_results_)
top5 = results.nlargest(5, "mean_test_score")[
    ["params", "mean_test_score", "std_test_score"]
]
print("\nTop 5 parameter combinations:")
print(top5.to_string(index=False))

# --- Randomized Search (faster for large spaces) ---
print("\n=== Randomized Search ===")
param_dist = {
    "C": loguniform(0.01, 100),
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
}
rand_search = RandomizedSearchCV(
    SVC(random_state=0),
    param_dist,
    n_iter=20,
    cv=10,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
rand_search.fit(X, y)

print(f"Best params:   {rand_search.best_params_}")
print(f"Best CV score: {rand_search.best_score_:.4f}")
