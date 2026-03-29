# Clustering

**Unsupervised learning** — find natural groupings in data without labels.

## Algorithms

| Algorithm | Key Idea | File |
|-----------|----------|------|
| K-Means | Assign points to nearest centroid, repeat | kmeans.py |
| Hierarchical | Build dendrogram; cut at desired number of clusters | hierarchical.py |

## Dataset

`datasets/Mall_Customers.csv` — CustomerID, Genre, Age, AnnualIncome, SpendingScore

## Labs

```bash
cd 04-clustering
python kmeans.py
python hierarchical.py
```

## Key Concepts

- **Elbow method** — plot WCSS vs k to find optimal number of clusters
- **Dendrogram** — tree diagram to visualise hierarchical merging
- **Silhouette score** — measures cluster cohesion and separation

## Cross-reference

- [python_learning/11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning)
- [cheatsheets/clustering.md](../cheatsheets/clustering.md)
