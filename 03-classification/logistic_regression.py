"""
Logistic Regression Classifier
================================
Dataset: datasets/Social_Network_Ads.csv
Features: Age, EstimatedSalary -> Purchased (0/1)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load data
df = pd.read_csv("datasets/Social_Network_Ads.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split + Scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train
model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


def plot_decision_boundary(X, y, model, title, filename):
    X1_min, X1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X2_min, X2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    XX1, XX2 = np.meshgrid(
        np.arange(X1_min, X1_max, 0.01), np.arange(X2_min, X2_max, 0.01)
    )
    Z = model.predict(np.array([XX1.ravel(), XX2.ravel()]).T).reshape(XX1.shape)
    plt.figure()
    plt.contourf(XX1, XX2, Z, alpha=0.75, cmap=ListedColormap(("salmon", "dodgerblue")))
    plt.scatter(
        X[:, 0], X[:, 1], c=y, cmap=ListedColormap(("red", "blue")), edgecolors="k"
    )
    plt.title(title)
    plt.xlabel("Age (scaled)")
    plt.ylabel("Salary (scaled)")
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.show()
    print(f"Plot saved: {filename}")


plot_decision_boundary(
    X_train,
    y_train,
    model,
    "Logistic Regression — Training Set",
    "logistic_regression_train.png",
)
plot_decision_boundary(
    X_test,
    y_test,
    model,
    "Logistic Regression — Test Set",
    "logistic_regression_test.png",
)
