# Scikit-learn Preprocessing Cheatsheet

## Import

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
    LabelEncoder, OneHotEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```

## Missing Values

```python
imp = SimpleImputer(strategy="mean")   # mean | median | most_frequent | constant
X[:, 1:3] = imp.fit_transform(X[:, 1:3])
```

## Categorical Encoding

```python
# Binary / ordinal target
le = LabelEncoder()
y = le.fit_transform(y)

# Nominal feature (one-hot)
ct = ColumnTransformer([
    ("ohe", OneHotEncoder(drop="first"), [0])   # drop='first' avoids dummy trap
], remainder="passthrough")
X = np.array(ct.fit_transform(X))
```

## Feature Scaling

```python
sc = StandardScaler()              # mean=0, std=1 (use for SVM, KNN, neural nets)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)      # NEVER fit on test — use training stats

mm = MinMaxScaler()                # scale to [0, 1]
```

## Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 80/20 split
    random_state=42,
    stratify=y           # preserves class proportions
)
```

## Pipeline (prevents data leakage)

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression()),
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `sc.fit_transform(X)` then split | Split first, then `fit_transform(X_train)`, `transform(X_test)` |
| `Imputer` (legacy) | `SimpleImputer` |
| `OneHotEncoder(categorical_features=[0])` | `ColumnTransformer` |
| `sklearn.cross_validation` | `sklearn.model_selection` |
