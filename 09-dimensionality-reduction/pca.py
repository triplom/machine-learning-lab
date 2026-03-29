"""
Principal Component Analysis (PCA)
====================================
Dataset: Wine dataset (sklearn built-in, 13 features, 3 classes)
Goal: Reduce to 2 components, visualise, then classify.

Steps:
  1. Scale features
  2. Apply PCA (reduce to 2 components)
  3. Plot explained variance (scree plot)
  4. Train Logistic Regression on reduced features
  5. Visualise decision boundary
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
class_names = wine.target_names

print(f"Original shape: {X.shape} ({X.shape[1]} features)")

# Split + scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# --- PCA: explained variance ---
pca_full = PCA()
pca_full.fit(X_train_sc)
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_var) + 1), explained_var, color="steelblue")
plt.title("Explained Variance per Component")
plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, marker="o")
plt.axhline(0.95, color="red", linestyle="--", label="95% threshold")
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.legend()
plt.tight_layout()
plt.savefig("pca_scree_plot.png", dpi=100)
plt.show()
print("Scree plot saved: pca_scree_plot.png")

components_for_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"Components needed for 95% variance: {components_for_95}")

# --- Reduce to 2 components for visualisation ---
pca2 = PCA(n_components=2)
X_train_2d = pca2.fit_transform(X_train_sc)
X_test_2d = pca2.transform(X_test_sc)

print(f"\nReduced shape: {X_train_2d.shape}")
print(f"Variance captured: {pca2.explained_variance_ratio_.sum():.4f}")

# Classify on 2D projection
model = LogisticRegression(random_state=0)
model.fit(X_train_2d, y_train)
y_pred = model.predict(X_test_2d)

print(f"\nAccuracy on 2 PCA components: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualise 2D decision boundary
colors = ["red", "green", "blue"]
markers = ["o", "s", "^"]
plt.figure(figsize=(8, 6))
for i, (c, m) in enumerate(zip(colors, markers)):
    plt.scatter(
        X_train_2d[y_train == i, 0],
        X_train_2d[y_train == i, 1],
        c=c,
        marker=m,
        label=class_names[i],
    )
plt.title("PCA — Wine Dataset (2 Components)")
plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.2%} variance)")
plt.legend()
plt.tight_layout()
plt.savefig("pca_2d_projection.png", dpi=100)
plt.show()
print("2D projection saved: pca_2d_projection.png")

# Top contributing features per component
loadings = pd.DataFrame(pca2.components_.T, columns=["PC1", "PC2"], index=feature_names)
print("\nTop feature loadings per component:")
print(loadings.abs().sort_values("PC1", ascending=False).round(4))
