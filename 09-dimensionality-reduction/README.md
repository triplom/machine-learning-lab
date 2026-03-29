# Dimensionality Reduction

Reduce the number of features while preserving as much information as possible. Useful before visualisation and to combat the curse of dimensionality.

## Algorithms

| Algorithm | Supervised? | Key Idea | File |
|-----------|------------|----------|------|
| PCA | No | Maximise variance in projected space | pca.py |
| LDA | Yes | Maximise class separability | lda.py |
| Kernel PCA | No | PCA in kernel-transformed space (non-linear) | kernel_pca.py |

## Lab

```bash
cd 09-dimensionality-reduction
python pca.py
python lda.py
python kernel_pca.py
```

## Key Concepts

- **Explained variance ratio** — how much variance each component captures
- **Scree plot** — visualise explained variance per component
- **LDA** — uses class labels; better than PCA when classes are well-separated

## Cross-reference

- [python_learning/11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning) — PCA in pipeline
