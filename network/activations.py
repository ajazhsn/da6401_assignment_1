import numpy as np


class Sigmoid:
    def forward(self, z):
        self.out = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh:
    def forward(self, z):
        self.out = np.tanh(z)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)


class ReLU:
    def forward(self, z):
        self.z = z
        return np.maximum(0, z)

    def backward(self, grad):
        return grad * (self.z > 0).astype(float)


class Identity:
    def forward(self, z):
        return z

    def backward(self, grad):
        return grad


def get_activation(name):
    """Return activation object by name."""
    activations = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
        "identity": Identity,
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(activations.keys())}")
    return activations[name]()
