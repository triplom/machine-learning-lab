"""
K-Nearest Neighbors (KNN) Classifier
=======================================
Dataset: datasets/Social_Network_Ads.csv
Features: Age, EstimatedSalary -> Purchased

Hyperparameter: n_neighbors (k=5 default). Try k=3,5,7,11 and compare.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("datasets/Social_Network_Ads.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Try different k values
for k in [3, 5, 7, 11]:
    model = KNeighborsClassifier(n_neighbors=k, metric="minkowski", p=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"k={k:2d} -> Accuracy: {acc:.4f}")

# Final model with k=5
model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nBest model (k=5):")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
