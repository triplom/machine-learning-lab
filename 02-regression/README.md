# Regression

Predict a **continuous** output variable from input features. Six algorithms covered.

## Algorithms

| Algorithm | Use When | File |
|-----------|----------|------|
| Simple Linear Regression | 1 feature, linear relationship | simple_linear_regression.py |
| Multiple Linear Regression | Multiple features, linear relationship | multiple_linear_regression.py |
| Polynomial Regression | Non-linear relationship | polynomial_regression.py |
| Support Vector Regression | Non-linear, robust to outliers | svr.py |
| Decision Tree Regression | Non-linear, interpretable splits | decision_tree_regression.py |
| Random Forest Regression | Non-linear, high accuracy via ensemble | random_forest_regression.py |

## Datasets

| File | Columns | Used By |
|------|---------|---------|
| Salary_Data.csv | YearsExperience, Salary | Simple Linear |
| 50_Startups.csv | R&D Spend, Admin, Marketing, State, Profit | Multiple Linear |
| Position_Salaries.csv | Position, Level, Salary | Polynomial, SVR, DT, RF |

## Labs

```bash
cd 02-regression
python simple_linear_regression.py
python multiple_linear_regression.py
python polynomial_regression.py
python svr.py
python decision_tree_regression.py
python random_forest_regression.py
```

## Key Metrics

- **R² score** — proportion of variance explained (1.0 = perfect)
- **RMSE** — root mean squared error (lower = better)

## Cross-reference

- [python_learning/11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning)
- [cheatsheets/regression.md](../cheatsheets/regression.md)
