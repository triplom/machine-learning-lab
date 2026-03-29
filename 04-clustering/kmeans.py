"""
K-Means Clustering
===================
Dataset: datasets/Mall_Customers.csv
Features: AnnualIncome, SpendingScore

Steps:
  1. Find optimal k using the Elbow Method (WCSS)
  2. Fit K-Means with optimal k
  3. Visualise clusters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv("datasets/Mall_Customers.csv")
X = df.iloc[:, [3, 4]].values  # AnnualIncome, SpendingScore

# --- Elbow Method: find optimal k ---
wcss = []
for k in range(1, 11):
    km = KMeans(
        n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=42
    )
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS (inertia)")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=100)
plt.show()
print("Elbow plot saved: kmeans_elbow.png")

# Silhouette scores
print("\nSilhouette scores:")
for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"  k={k}: {score:.4f}")

# --- Fit with k=5 (elbow typically here for this dataset) ---
optimal_k = 5
model = KMeans(
    n_clusters=optimal_k, init="k-means++", max_iter=300, n_init=10, random_state=42
)
y_pred = model.fit_predict(X)

# Visualise clusters
colors = ["red", "blue", "green", "cyan", "magenta"]
plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    plt.scatter(
        X[y_pred == i, 0],
        X[y_pred == i, 1],
        s=50,
        c=colors[i],
        label=f"Cluster {i + 1}",
    )
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    s=200,
    c="yellow",
    marker="*",
    edgecolors="black",
    label="Centroids",
)
plt.title("K-Means Clustering (k=5)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig("kmeans_clusters.png", dpi=100)
plt.show()
print("Cluster plot saved: kmeans_clusters.png")

# Cluster profile
df["Cluster"] = y_pred
print("\nCluster sizes:")
print(df["Cluster"].value_counts().sort_index())
print("\nCluster means (Income, Spending):")
print(
    df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]]
    .mean()
    .round(1)
)
