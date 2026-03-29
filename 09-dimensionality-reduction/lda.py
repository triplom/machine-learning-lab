"""
Linear Discriminant Analysis (LDA)
=====================================
Dataset: Wine dataset (sklearn built-in)

LDA is a supervised dimensionality reduction method.
It maximises the ratio of between-class to within-class variance.
Max components = n_classes - 1 (so 2 for Wine's 3 classes).

Compare: LDA vs PCA — LDA often gives better class separation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()
X = wine.data
y = wine.target
class_names = wine.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# --- LDA: reduce to 2 discriminant components ---
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_sc, y_train)
X_test_lda = lda.transform(X_test_sc)

print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
print(f"Total variance captured: {lda.explained_variance_ratio_.sum():.4f}")

# Classify on LDA components
model = LogisticRegression(random_state=0)
model.fit(X_train_lda, y_train)
y_pred = model.predict(X_test_lda)

print(f"\nAccuracy on 2 LDA components: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Visualise
colors = ["red", "green", "blue"]
markers = ["o", "s", "^"]
plt.figure(figsize=(8, 6))
for i, (c, m) in enumerate(zip(colors, markers)):
    plt.scatter(
        X_train_lda[y_train == i, 0],
        X_train_lda[y_train == i, 1],
        c=c,
        marker=m,
        label=class_names[i],
    )
plt.title("LDA — Wine Dataset (2 Components)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.tight_layout()
plt.savefig("lda_2d_projection.png", dpi=100)
plt.show()
print("2D LDA projection saved: lda_2d_projection.png")

# Note: compare vs PCA
print("\nKey difference: LDA uses class labels -> better class separation.")
print("PCA is unsupervised -> maximises variance, ignores class structure.")
