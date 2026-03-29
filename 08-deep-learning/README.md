# Deep Learning

Build and train neural networks with **TensorFlow 2 / Keras**.

## Topics

| Topic | File |
|-------|------|
| Artificial Neural Network (ANN) — binary classification | ann.py |
| Convolutional Neural Network (CNN) — image classification | cnn.py |

## Installation

```bash
pip install tensorflow
```

## Labs

```bash
cd 08-deep-learning
python ann.py      # ANN on Churn dataset
python cnn.py      # CNN on CIFAR-10 (downloads automatically)
```

## Key Concepts

### ANN
- Layers: `Dense` (fully connected)
- Activations: `relu` (hidden), `sigmoid` (binary output), `softmax` (multi-class)
- Loss: `binary_crossentropy` for binary, `categorical_crossentropy` for multi-class
- Optimiser: `Adam`

### CNN
- Conv layers: `Conv2D` + `MaxPooling2D`
- Regularisation: `Dropout`, `BatchNormalization`
- Transfer learning: load pretrained weights (VGG16, ResNet50)

## Cross-reference

- [python_learning/12-deep-learning](https://github.com/triplom/python_learning/tree/main/12-deep-learning) — Keras MNIST basics
- [cheatsheets/deep-learning-keras.md](../cheatsheets/deep-learning-keras.md)
