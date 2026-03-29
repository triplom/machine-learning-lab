"""
Decision Tree Classifier
=========================
Dataset: datasets/Social_Network_Ads.csv

Interpretable splits using entropy / gini impurity.
Prone to overfitting — compare with Random Forest.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
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

model = DecisionTreeClassifier(criterion="entropy", random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Decision Tree (criterion=entropy)")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nTree depth: {model.get_depth()}")
print("\nTree structure (text):")
print(export_text(model, feature_names=["Age", "Salary"]))
