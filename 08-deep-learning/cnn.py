"""
Convolutional Neural Network (CNN) — Image Classification
==========================================================
Dataset: CIFAR-10 (downloaded automatically via tf.keras.datasets)
Task: Classify 32x32 colour images into 10 categories.
Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Architecture:
  Conv2D(32) -> MaxPool -> BatchNorm ->
  Conv2D(64) -> MaxPool -> BatchNorm ->
  Conv2D(128) -> MaxPool -> Dropout ->
  Flatten -> Dense(256) -> Dropout -> Dense(10, softmax)

Cross-reference: python_learning/12-deep-learning/keras_mnist.py
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

print(f"TensorFlow version: {tf.__version__}")

# Class labels
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# ------------------------------------------------------------------
# 1. Load and preprocess CIFAR-10
# ------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

# Normalise pixel values to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")
print(f"Classes: {CLASS_NAMES}")

# Visualise sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(CLASS_NAMES[y_train[i]])
    ax.axis("off")
plt.suptitle("CIFAR-10 Sample Images")
plt.tight_layout()
plt.savefig("cifar10_samples.png", dpi=100)
plt.show()
print("Sample images saved: cifar10_samples.png")

# ------------------------------------------------------------------
# 2. Build CNN
# ------------------------------------------------------------------
model = tf.keras.Sequential(
    [
        # Block 1
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
        ),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        # Classifier head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ------------------------------------------------------------------
# 3. Data augmentation
# ------------------------------------------------------------------
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

# ------------------------------------------------------------------
# 4. Train
# ------------------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
]

# Use augmented batches
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .batch(64)
    .map(lambda x, y: (data_augmentation(x, training=True), y))
    .prefetch(tf.data.AUTOTUNE)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=callbacks,
    verbose=1,
)

# ------------------------------------------------------------------
# 5. Evaluate
# ------------------------------------------------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Loss:     {loss:.4f}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# ------------------------------------------------------------------
# 6. Plot training curves
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history["accuracy"], label="Train")
ax1.plot(history.history["val_accuracy"], label="Val")
ax1.set_title("Accuracy per Epoch")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.plot(history.history["loss"], label="Train")
ax2.plot(history.history["val_loss"], label="Val")
ax2.set_title("Loss per Epoch")
ax2.set_xlabel("Epoch")
ax2.legend()

plt.tight_layout()
plt.savefig("cnn_training_curves.png", dpi=100)
plt.show()
print("Plot saved: cnn_training_curves.png")

# ------------------------------------------------------------------
# 7. Save model
# ------------------------------------------------------------------
model.save("cnn_cifar10_model.keras")
print("Model saved: cnn_cifar10_model.keras")

# ------------------------------------------------------------------
# 8. Transfer Learning hint
# ------------------------------------------------------------------
print("\n--- Transfer Learning Example (not run here) ---")
print("""
# Load pretrained VGG16 (ImageNet weights), freeze base, add new head:

base = tf.keras.applications.VGG16(weights='imagenet', include_top=False,
                                    input_shape=(32, 32, 3))
base.trainable = False

model = tf.keras.Sequential([
    base,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
""")
