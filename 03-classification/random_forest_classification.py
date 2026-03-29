"""
Random Forest Classifier
=========================
Dataset: datasets/Social_Network_Ads.csv

Ensemble of decision trees — reduces variance via bagging.
n_estimators: more trees = more stable predictions (with diminishing returns).
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

# Compare n_estimators
for n in [10, 50, 100, 300]:
    model = RandomForestClassifier(n_estimators=n, criterion="entropy", random_state=0)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"n_estimators={n:3d} -> Accuracy: {acc:.4f}")

# Final model
model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nFinal model (n=100):")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(
    f"\nFeature importances: {dict(zip(['Age', 'Salary'], model.feature_importances_.round(4)))}"
)
