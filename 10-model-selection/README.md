# Model Selection & Boosting

Evaluate and tune models systematically to find the best configuration.

## Topics

| Topic | File |
|-------|------|
| k-Fold Cross Validation | cross_validation.py |
| Grid Search (hyperparameter tuning) | grid_search.py |
| XGBoost | xgboost_lab.py |

## Key Concepts

| Technique | Purpose |
|-----------|---------|
| k-Fold CV | Reduces bias from single train/test split; gives mean ± std accuracy |
| Grid Search | Exhaustively search hyperparameter combinations |
| Randomized Search | Sample random subset of hyperparameters (faster) |
| XGBoost | Gradient boosting — state-of-the-art on tabular data |

## Installation

```bash
pip install xgboost
```

## Labs

```bash
cd 10-model-selection
python cross_validation.py
python grid_search.py
python xgboost_lab.py
```

## Cross-reference

- [cheatsheets/model-selection.md](../cheatsheets/model-selection.md)
- [python_learning/11-machine-learning](https://github.com/triplom/python_learning/tree/main/11-machine-learning)
