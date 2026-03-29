# Clustering Cheatsheet

## Quick Import

```python
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
```

## K-Means

```python
# Find optimal k with elbow method
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)
# Plot WCSS vs k, look for the "elbow"

# Fit final model
model = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42)
labels = model.fit_predict(X)
centroids = model.cluster_centers_
```

### Key parameters

| Parameter | Effect |
|-----------|--------|
| `n_clusters` | Number of clusters (k) |
| `init="k-means++"` | Smart centroid initialisation (avoids bad local minima) |
| `n_init` | Number of random initialisations (default 10) |
| `max_iter` | Max EM iterations per init |

## Hierarchical Clustering

```python
# Dendrogram (find optimal k visually)
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

dend = sch.dendrogram(sch.linkage(X, method="ward"))
plt.show()
# Look for the longest vertical line without a horizontal bar crossing it

# Fit
model = AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
labels = model.fit_predict(X)
```

### Linkage methods

| Method | Merges based on |
|--------|----------------|
| `ward` | Minimise within-cluster variance (most common) |
| `complete` | Maximum distance between clusters |
| `average` | Average distance |
| `single` | Minimum distance (can produce long chains) |

## Silhouette Score

```python
score = silhouette_score(X, labels)
# Range: [-1, 1] — higher is better
# 1: perfect clusters, 0: overlapping, -1: wrong cluster
```

## Notes

- K-Means requires specifying k upfront; hierarchical does not
- K-Means is sensitive to outliers (use `k-means++` init)
- Always scale features before clustering
- K-Means is faster; hierarchical is better for small datasets with unknown k
