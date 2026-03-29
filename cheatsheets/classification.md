# Classification Cheatsheet

## Quick Import

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
    classification_report, roc_auc_score)
```

## Algorithm Selection

| Algorithm | Strengths | Weaknesses |
|-----------|-----------|------------|
| Logistic Regression | Fast, probabilistic output | Assumes linear boundary |
| KNN | Simple, non-parametric | Slow at predict time, sensitive to scale |
| SVM (linear) | Effective high-dim | Slow on large datasets |
| Kernel SVM | Non-linear boundary | Slow on large datasets |
| Naive Bayes | Very fast, great for text | Feature independence assumption |
| Decision Tree | Interpretable | Overfits |
| Random Forest | High accuracy | Less interpretable |

## Common Code Pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## Evaluation

```python
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# For binary classification
proba = model.predict_proba(X_test)[:, 1]
print(roc_auc_score(y_test, proba))
```

### Confusion Matrix Layout

```
                Predicted 0    Predicted 1
Actual 0    [  TN          FP  ]
Actual 1    [  FN          TP  ]
```

- **Precision** = TP / (TP + FP) — of all predicted positives, how many were correct
- **Recall** = TP / (TP + FN) — of all actual positives, how many were caught
- **F1** = 2 * P * R / (P + R) — harmonic mean

## Which metric to use?

| Scenario | Metric |
|----------|--------|
| Balanced classes | Accuracy |
| Imbalanced, FP costly (spam filter) | Precision |
| Imbalanced, FN costly (cancer detection) | Recall |
| Imbalanced, balance both | F1 |
| Ranking / probability | ROC-AUC |

## Notes

- Always scale features for KNN, SVM, Logistic Regression
- Decision Tree / Random Forest do NOT need scaling
- Use `stratify=y` in `train_test_split` for imbalanced datasets
