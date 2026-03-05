# src/ann/activations.py
# Activation functions for the MLP.
# Each activation class implements:
#   - forward(z)  : applies the activation to pre-activation z
#   - backward(grad): propagates gradient through the activation

import numpy as np


class Sigmoid:
    """
    Sigmoid activation: f(z) = 1 / (1 + exp(-z))
    Output range: (0, 1)
    Known issue: causes vanishing gradients in deep networks
    because derivative max is only 0.25.
    """

    def forward(self, z):
        # Clip z to avoid overflow in exp
        self.out = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, grad):
        # d/dz sigmoid(z) = sigmoid(z) * (1 - sigmoid(z))
        return grad * self.out * (1 - self.out)


class Tanh:
    """
    Tanh activation: f(z) = tanh(z)
    Output range: (-1, 1)
    Better than sigmoid for hidden layers as it is zero-centered.
    """

    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad):
        # d/dz tanh(z) = 1 - tanh(z)^2
        return grad * (1 - self.out ** 2)


class ReLU:
    """
    Rectified Linear Unit: f(z) = max(0, z)
    Output range: [0, inf)
    Default choice for hidden layers — avoids vanishing gradients.
    Risk: "dead neurons" if learning rate is too high.
    """

    def forward(self, z):
        self.z = z  # cache for backward
        return np.maximum(0, z)

    def backward(self, grad):
        # Gradient is 1 where z > 0, else 0
        return grad * (self.z > 0).astype(float)


class Identity:
    """
    Identity activation: f(z) = z
    Used for the output layer — loss function handles softmax internally.
    """

    def forward(self, z):
        return z

    def backward(self, grad):
        return grad


def get_activation(name: str):
    """
    Factory function — returns an activation object by name.

    Args:
        name: One of 'sigmoid', 'tanh', 'relu', 'identity'
    Returns:
        Activation object with forward() and backward() methods
    """
    activations = {
        "sigmoid":  Sigmoid,
        "tanh":     Tanh,
        "relu":     ReLU,
        "identity": Identity,
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from: {list(activations.keys())}"
        )
    return activations[name]()
