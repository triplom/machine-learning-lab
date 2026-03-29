"""
Data Preprocessing Lab
======================
Modern scikit-learn API (sklearn >= 0.24)

Dataset: datasets/Data.csv
Columns: Country, Age, Salary, Purchased

Steps:
  1. Load data
  2. Handle missing values (SimpleImputer)
  3. Encode categorical features (ColumnTransformer + OneHotEncoder)
  4. Encode target variable (LabelEncoder)
  5. Feature scaling (StandardScaler)
  6. Train/test split
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------------
df = pd.read_csv("datasets/Data.csv")
print("Raw data:")
print(df)
print()

X = df.iloc[:, :-1].values  # Country, Age, Salary
y = df.iloc[:, -1].values  # Purchased (Yes/No)

# ------------------------------------------------------------------
# 2. Handle missing values in numeric columns (Age, Salary)
# ------------------------------------------------------------------
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

print("After imputation (numeric columns filled with mean):")
print(X)
print()

# ------------------------------------------------------------------
# 3. Encode categorical feature: Country (column 0)
# ------------------------------------------------------------------
ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(), [0])  # OneHot on column 0 (Country)
    ],
    remainder="passthrough",  # keep Age, Salary as-is
)
X = np.array(ct.fit_transform(X))

print("After OneHotEncoding Country:")
print(X)
print()

# ------------------------------------------------------------------
# 4. Encode target variable: Purchased (Yes -> 1, No -> 0)
# ------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(y)
print("Encoded target (y):", y)
print()

# ------------------------------------------------------------------
# 5. Train/test split
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ------------------------------------------------------------------
# 6. Feature scaling
# ------------------------------------------------------------------
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # use training stats on test set — important!

print("X_train (scaled):")
print(X_train)
print()
print("X_test (scaled):")
print(X_test)
print()
print("y_train:", y_train)
print("y_test: ", y_test)
