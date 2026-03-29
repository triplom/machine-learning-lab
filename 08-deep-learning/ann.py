"""
Artificial Neural Network (ANN) — Customer Churn Prediction
=============================================================
Dataset: Synthetic bank customer churn (similar to classic ML A-Z dataset).
Task: Predict if a customer will leave the bank (Exited = 0/1).

Architecture:
  Input -> Dense(64, relu) -> Dropout(0.3) ->
           Dense(32, relu) -> Dropout(0.3) ->
           Dense(1, sigmoid)

Cross-reference: python_learning/12-deep-learning/keras_mnist.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

# ------------------------------------------------------------------
# 1. Synthetic dataset — bank customer churn
# ------------------------------------------------------------------
np.random.seed(42)
n = 1000

df = pd.DataFrame(
    {
        "CreditScore": np.random.randint(350, 850, n),
        "Geography": np.random.choice(["France", "Spain", "Germany"], n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Age": np.random.randint(18, 70, n),
        "Tenure": np.random.randint(0, 10, n),
        "Balance": np.random.uniform(0, 250000, n).round(2),
        "NumOfProducts": np.random.randint(1, 5, n),
        "HasCrCard": np.random.randint(0, 2, n),
        "IsActiveMember": np.random.randint(0, 2, n),
        "EstimatedSalary": np.random.uniform(10000, 200000, n).round(2),
    }
)
# Exited: older, German, inactive members more likely to churn
df["Exited"] = (
    (df["Age"] > 45).astype(int)
    + (df["Geography"] == "Germany").astype(int)
    + (df["IsActiveMember"] == 0).astype(int)
    + (np.random.rand(n) < 0.15).astype(int)
).clip(0, 1)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Exited'].mean():.2%}")
print()

X = df.drop("Exited", axis=1).values
y = df["Exited"].values

# ------------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------------
# Encode Gender (col 2)
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# OneHot Geography (col 1)
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(drop="first"), [1])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X), dtype=float)

# Split + scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ------------------------------------------------------------------
# 3. Build ANN
# ------------------------------------------------------------------
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# ------------------------------------------------------------------
# 4. Train
# ------------------------------------------------------------------
history = model.fit(
    X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1
)

# ------------------------------------------------------------------
# 5. Evaluate
# ------------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Loss:     {loss:.4f}")

y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------------------------------------------
# 6. Plot training curves
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig("ann_training_curves.png", dpi=100)
plt.show()
print("Plot saved: ann_training_curves.png")

# ------------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------------
model.save("ann_churn_model.keras")
print("Model saved: ann_churn_model.keras")

# Predict single customer
sample = X_test[0:1]
pred = model.predict(sample, verbose=0)[0][0]
print(
    f"\nSample prediction — Churn probability: {pred:.4f} -> {'CHURN' if pred >= 0.5 else 'STAY'}"
)
