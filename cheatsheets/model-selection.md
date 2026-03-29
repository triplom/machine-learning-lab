# Model Selection Cheatsheet

## Cross Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

print(f"Mean: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Use Pipeline to avoid data leakage:**
```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC())])
scores = cross_val_score(pipe, X, y, cv=10)
```

## Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
gs = GridSearchCV(SVC(), param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gs.fit(X, y)

print(gs.best_params_)
print(gs.best_score_)
best_model = gs.best_estimator_
```

## Randomized Search (faster)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

param_dist = {"C": loguniform(0.01, 100), "gamma": [0.001, 0.01, "scale"]}
rs = RandomizedSearchCV(SVC(), param_dist, n_iter=20, cv=10, random_state=42)
rs.fit(X, y)
```

## XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=50)
```

### XGBoost key params

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | 100 | Number of trees |
| `max_depth` | 6 | Tree depth (lower = less overfit) |
| `learning_rate` | 0.3 | Shrinkage (lower = better generalisation) |
| `subsample` | 1.0 | Row sampling per tree |
| `colsample_bytree` | 1.0 | Feature sampling per tree |
| `reg_lambda` | 1 | L2 regularisation |

## Model Comparison

```python
from sklearn.model_selection import cross_val_score

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM (RBF)": SVC(kernel="rbf"),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=10)
    print(f"{name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Bias-Variance Tradeoff

| | High Bias (underfitting) | High Variance (overfitting) |
|---|---|---|
| Train acc | Low | High |
| Test acc | Low | Low |
| Fix | More complex model | Regularise, more data, dropout |

## Notes

- Always use `StratifiedKFold` for classification (preserves class ratio)
- Use `Pipeline` to prevent data leakage in CV
- Use `n_jobs=-1` to parallelise Grid Search
- Compare models by CV mean ± std, not single test split
