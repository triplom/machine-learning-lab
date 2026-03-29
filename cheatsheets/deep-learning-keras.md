# Deep Learning / Keras Cheatsheet

## Quick Import

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## Build a Model

### Sequential API
```python
model = tf.keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid"),   # binary output
])
```

### Functional API (multi-input/output)
```python
inputs = keras.Input(shape=(n_features,))
x = layers.Dense(64, activation="relu")(inputs)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
```

## Compile

```python
model.compile(
    optimizer="adam",                          # or tf.keras.optimizers.Adam(lr=0.001)
    loss="binary_crossentropy",                # binary classification
    # loss="sparse_categorical_crossentropy",  # multi-class (integer labels)
    # loss="categorical_crossentropy",         # multi-class (one-hot labels)
    # loss="mse",                              # regression
    metrics=["accuracy"]
)
```

## Train

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,        # or validation_data=(X_val, y_val)
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]
)
```

## Evaluate & Predict

```python
loss, acc = model.evaluate(X_test, y_test)
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)     # binary threshold
y_pred = np.argmax(model.predict(X_test), axis=1)  # multi-class
```

## Activation Functions

| Activation | Use |
|-----------|-----|
| `relu` | Hidden layers (standard) |
| `sigmoid` | Binary output (0–1 probability) |
| `softmax` | Multi-class output (probabilities sum to 1) |
| `tanh` | Sometimes better than relu in RNNs |
| `leaky_relu` | Avoids dying ReLU problem |

## Common Layer Types

| Layer | Purpose |
|-------|---------|
| `Dense(n, activation)` | Fully connected |
| `Dropout(rate)` | Regularisation |
| `BatchNormalization()` | Stabilise training |
| `Conv2D(filters, kernel)` | Spatial feature extraction |
| `MaxPooling2D(pool)` | Downsampling |
| `Flatten()` | 2D → 1D for Dense layers |
| `LSTM(units)` | Sequential data |
| `Embedding(vocab, dim)` | Text embeddings |

## Save / Load

```python
model.save("model.keras")                   # SavedModel format
model = tf.keras.models.load_model("model.keras")

# Save weights only
model.save_weights("weights.h5")
model.load_weights("weights.h5")
```

## Transfer Learning

```python
base = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False   # freeze

model = tf.keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
```

## Cross-reference

- [08-deep-learning/ann.py](../08-deep-learning/ann.py)
- [08-deep-learning/cnn.py](../08-deep-learning/cnn.py)
- [python_learning/12-deep-learning](https://github.com/triplom/python_learning/tree/main/12-deep-learning)
