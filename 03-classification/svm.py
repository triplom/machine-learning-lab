"""
Support Vector Machine (SVM) — Linear Kernel
=============================================
Dataset: datasets/Social_Network_Ads.csv
Features: Age, EstimatedSalary -> Purchased

Linear SVM finds a hyperplane that maximises the margin between classes.
Use kernel='rbf' (see kernel_svm.py) for non-linear boundaries.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

model = SVC(kernel="linear", random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("SVM (linear kernel)")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nNumber of support vectors: {model.n_support_}")
