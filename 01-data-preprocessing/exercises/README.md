# Exercises: Data Preprocessing

## Exercise 1 — End-to-End Preprocessing

Load `datasets/Data.csv`. Apply the following pipeline:
1. Impute missing values in numeric columns using the **median** (not mean)
2. OneHot-encode the `Country` column
3. Label-encode `Purchased`
4. Scale with `MinMaxScaler` instead of `StandardScaler`
5. Split: 80% train / 20% test

**Expected:** print shapes of X_train, X_test, y_train, y_test.

---

## Exercise 2 — StandardScaler vs MinMaxScaler

Using the iris dataset (`sklearn.datasets.load_iris`):
1. Apply `StandardScaler` → compute mean and std of scaled X
2. Apply `MinMaxScaler` → compute min and max of scaled X
3. Print results and explain the difference

---

## Exercise 3 — Full ColumnTransformer Pipeline

Create a `Pipeline` with:
- `ColumnTransformer`: OneHotEncoder on categorical cols, passthrough on numeric
- `SimpleImputer` for missing values (add some NaN manually)
- `StandardScaler` on all numeric features

Use `sklearn.pipeline.Pipeline` to chain all steps.

---

## Exercise 4 — Spotting Data Leakage

Given this code snippet:

```python
sc = StandardScaler()
X_scaled = sc.fit_transform(X)          # fit on all data
X_train, X_test = train_test_split(X_scaled, ...)
```

**Question:** What is wrong with this approach? How does it cause data leakage?  
**Fix it.**
