"""
Hierarchical Clustering
========================
Dataset: datasets/Mall_Customers.csv
Features: AnnualIncome, SpendingScore

Steps:
  1. Build dendrogram (Ward linkage)
  2. Choose number of clusters from dendrogram
  3. Fit AgglomerativeClustering and visualise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Load data
df = pd.read_csv("datasets/Mall_Customers.csv")
X = df.iloc[:, [3, 4]].values

# --- Dendrogram ---
plt.figure(figsize=(12, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method="ward"))
plt.title("Dendrogram — Mall Customers")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance (Ward)")
plt.tight_layout()
plt.savefig("hierarchical_dendrogram.png", dpi=100)
plt.show()
print("Dendrogram saved: hierarchical_dendrogram.png")
print("(Look for the largest vertical gap to determine optimal k)")

# --- Fit AgglomerativeClustering (k=5) ---
model = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
y_pred = model.fit_predict(X)

# Visualise clusters
colors = ["red", "blue", "green", "cyan", "magenta"]
plt.figure(figsize=(8, 6))
for i in range(5):
    plt.scatter(
        X[y_pred == i, 0],
        X[y_pred == i, 1],
        s=50,
        c=colors[i],
        label=f"Cluster {i + 1}",
    )
plt.title("Hierarchical Clustering (k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig("hierarchical_clusters.png", dpi=100)
plt.show()
print("Cluster plot saved: hierarchical_clusters.png")

# Cluster profile
df["Cluster"] = y_pred
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())
