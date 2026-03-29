"""
k-Fold Cross Validation
========================
Dataset: Social_Network_Ads.csv
Model: Kernel SVM (RBF)

A single train/test split gives a biased accuracy estimate.
k-Fold CV splits the data k times and averages results.

Reports: mean accuracy and standard deviation across folds.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("../03-classification/datasets/Social_Network_Ads.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Use a Pipeline to avoid data leakage during CV
pipeline = Pipeline(
    [("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", random_state=0))]
)

# 10-Fold Stratified CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("10-Fold Cross Validation — Kernel SVM (RBF)")
print(f"Fold accuracies: {scores.round(4)}")
print(f"Mean accuracy:   {scores.mean():.4f}")
print(f"Std deviation:   {scores.std():.4f}")
print(
    f"95% CI:          [{scores.mean() - 2 * scores.std():.4f}, "
    f"{scores.mean() + 2 * scores.std():.4f}]"
)

# Compare with single split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
pipeline.fit(X_train, y_train)
single_acc = pipeline.score(X_test, y_test)
print(f"\nSingle split accuracy: {single_acc:.4f}")
print(f"CV mean accuracy:      {scores.mean():.4f}")
print("\nCV gives a more reliable estimate by averaging over {cv.n_splits} folds.")
