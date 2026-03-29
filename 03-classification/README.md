# Classification

Predict a **categorical** output (class label) from input features. Seven algorithms covered.

## Algorithms

| Algorithm | Best For | File |
|-----------|----------|------|
| Logistic Regression | Linearly separable, probabilistic output | logistic_regression.py |
| K-Nearest Neighbors (KNN) | Simple, non-parametric | knn.py |
| Support Vector Machine (SVM) | Linear margin maximization | svm.py |
| Kernel SVM | Non-linear boundaries | kernel_svm.py |
| Naive Bayes | Text classification, fast | naive_bayes.py |
| Decision Tree | Interpretable, non-linear | decision_tree_classification.py |
| Random Forest | High accuracy ensemble | random_forest_classification.py |

## Dataset

`datasets/Social_Network_Ads.csv` — Age, EstimatedSalary, Purchased (0/1)

## Labs

```bash
cd 03-classification
python logistic_regression.py
python knn.py
python svm.py
python kernel_svm.py
python naive_bayes.py
python decision_tree_classification.py
python random_forest_classification.py
```

## Key Metrics

- **Accuracy** — correct predictions / total
- **Confusion Matrix** — TP, TN, FP, FN
- **Precision / Recall / F1** — useful when classes are imbalanced

## Cross-reference

- [python_learning/11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning)
- [cheatsheets/classification.md](../cheatsheets/classification.md)
