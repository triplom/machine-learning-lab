"""
Kernel PCA
===========
Dataset: Synthetic non-linear (make_moons / make_circles)

Standard PCA fails on non-linearly separable data.
Kernel PCA maps data to higher dimensions using a kernel trick, then applies PCA.

Kernels: rbf (Gaussian), poly, sigmoid, cosine
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Dataset 1: make_moons ---
X, y = make_moons(n_samples=400, noise=0.1, random_state=42)
sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standard PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lr_pca = LogisticRegression()
lr_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, lr_pca.predict(X_test_pca))

# Kernel PCA (RBF)
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=15)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)

lr_kpca = LogisticRegression()
lr_kpca.fit(X_train_kpca, y_train)
acc_kpca = accuracy_score(y_test, lr_kpca.predict(X_test_kpca))

print("=== make_moons dataset ===")
print(f"PCA accuracy:        {acc_pca:.4f}")
print(f"Kernel PCA accuracy: {acc_kpca:.4f}")

# Visualise
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", alpha=0.7)
axes[0].set_title("Original Data")

X_train_pca_full = pca.fit_transform(X)
axes[1].scatter(
    X_train_pca_full[:, 0],
    X_train_pca_full[:, 1],
    c=y,
    cmap="bwr",
    edgecolors="k",
    alpha=0.7,
)
axes[1].set_title(f"PCA (acc={acc_pca:.2f})")

X_kpca_full = kpca.fit_transform(X)
axes[2].scatter(
    X_kpca_full[:, 0], X_kpca_full[:, 1], c=y, cmap="bwr", edgecolors="k", alpha=0.7
)
axes[2].set_title(f"Kernel PCA RBF (acc={acc_kpca:.2f})")

plt.suptitle("PCA vs Kernel PCA — make_moons")
plt.tight_layout()
plt.savefig("kernel_pca_moons.png", dpi=100)
plt.show()
print("Plot saved: kernel_pca_moons.png")
