"""
XGBoost — Extreme Gradient Boosting
=====================================
Install: pip install xgboost

Dataset: Breast Cancer (sklearn built-in) — binary classification
         30 features, 569 samples

XGBoost builds trees sequentially; each tree corrects the errors of the previous.
Usually outperforms Random Forest on tabular data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, plot_importance

# Load data
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Classes: {cancer.target_names}")
print()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- XGBoost ---
model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)

# Early stopping on eval set
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# k-Fold CV
scores = cross_val_score(
    XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
    X,
    y,
    cv=10,
    scoring="accuracy",
)
print(f"\n10-Fold CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Feature importance
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=15, importance_type="gain")
plt.title("XGBoost — Top 15 Feature Importances (gain)")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=100)
plt.show()
print("Feature importance plot saved: xgboost_feature_importance.png")

# Learning curve
results = model.evals_result()
plt.figure(figsize=(8, 4))
plt.plot(results["validation_0"]["logloss"], label="Test log-loss")
plt.xlabel("Estimators")
plt.ylabel("Log Loss")
plt.title("XGBoost Learning Curve")
plt.legend()
plt.tight_layout()
plt.savefig("xgboost_learning_curve.png", dpi=100)
plt.show()
print("Learning curve saved: xgboost_learning_curve.png")
