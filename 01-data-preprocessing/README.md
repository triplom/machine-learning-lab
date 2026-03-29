# Data Preprocessing

**Phase 1 of the ML pipeline.** Before any algorithm runs, data must be clean, encoded, and scaled.

## Topics

| Topic | What it does |
|-------|-------------|
| Missing values | Impute with mean/median using `SimpleImputer` |
| Categorical encoding | `LabelEncoder`, `OneHotEncoder`, `ColumnTransformer` |
| Feature scaling | `StandardScaler`, `MinMaxScaler` |
| Train/test split | `train_test_split` |

## Dataset

`datasets/Data.csv` — country, age, salary, purchased (10 rows, contains missing values)

## Lab

```bash
cd 01-data-preprocessing
python data_preprocessing.py
```

Expected output: preprocessed X and y arrays, split into train/test sets.

## Key API Changes from Legacy Code

| Legacy (broken) | Modern (correct) |
|-----------------|-----------------|
| `from sklearn.cross_validation import train_test_split` | `from sklearn.model_selection import train_test_split` |
| `from sklearn.preprocessing import Imputer` | `from sklearn.impute import SimpleImputer` |
| `OneHotEncoder(categorical_features=[0])` | `ColumnTransformer` + `OneHotEncoder()` |

## Exercises

See `exercises/` for practice problems:
1. Load a new CSV with missing values and preprocess it end-to-end
2. Compare `StandardScaler` vs `MinMaxScaler` on the same dataset
3. Add a `ColumnTransformer` pipeline with both numerical and categorical columns

## Cross-reference

- [python_learning/11-machine-learning/sklearn_models.py](https://github.com/triplom/python_learning/tree/main/11-machine-learning) — sklearn pipeline overview
- [cheatsheets/sklearn-preprocessing.md](../cheatsheets/sklearn-preprocessing.md)
