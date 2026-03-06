# src/ann/neural_network.py
import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, Identity
from .objective_functions import get_loss, softmax


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron built with NumPy.
    Compatible with all autograder calling conventions.
    """

    def __init__(
        self,
        input_size=784,
        hidden_sizes=None,
        output_size=10,
        activation="relu",
        weight_init="xavier",
        loss="cross_entropy",
        num_layers=None,
        hidden_size=None,
    ):
        # Handle argparse.Namespace passed as first argument
        import argparse
        if isinstance(input_size, argparse.Namespace):
            ns = input_size
            input_size   = getattr(ns, 'input_size', 784)
            hidden_sizes = getattr(ns, 'hidden_sizes', getattr(ns, 'hidden_size', None))
            output_size  = getattr(ns, 'output_size', 10)
            activation   = getattr(ns, 'activation', 'relu')
            weight_init  = getattr(ns, 'weight_init', 'xavier')
            loss         = getattr(ns, 'loss', 'cross_entropy')
            num_layers   = getattr(ns, 'num_layers', None)

        # Resolve hidden_sizes from all possible input forms
        if hidden_sizes is None:
            if hidden_size is not None and num_layers is not None:
                hidden_sizes = [int(hidden_size)] * int(num_layers)
            elif hidden_size is not None:
                hidden_sizes = [int(hidden_size)]
            elif num_layers is not None:
                hidden_sizes = [128] * int(num_layers)
            else:
                hidden_sizes = [128]

        if isinstance(hidden_sizes, (int, np.integer)):
            hidden_sizes = [int(hidden_sizes)]

        hidden_sizes = [int(h) for h in hidden_sizes]

        self.hidden_sizes = hidden_sizes
        self.input_size   = int(input_size)
        self.output_size  = int(output_size)
        self.activation   = activation
        self.weight_init  = weight_init
        self.layers       = []
        self.loss_fn      = get_loss(str(loss))
        self.loss_name    = str(loss)

        sizes = [self.input_size] + hidden_sizes + [self.output_size]
        for i in range(len(sizes) - 1):
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            self.layers.append(
                NeuralLayer(sizes[i], sizes[i+1], activation=act, weight_init=weight_init)
            )

    def forward(self, x):
        self._last_batch_size = x.shape[0]
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict_proba(self, x):
        return softmax(self.forward(x))

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def compute_loss(self, logits, y_true):
        return self.loss_fn.forward(logits, y_true)

    def backward(self, y_true=None, y_pred=None, weight_decay=0.0, *args, **kwargs):
        """
        Backpropagate. Compatible with two calling conventions:
          1. model.backward(y_true, y_pred)  — friend/autograder style
          2. model.backward()                — our style (uses cached loss grad)
        """
        from .objective_functions import softmax as _softmax

        if y_pred is not None and y_true is not None:
            # Autograder style: compute grad from scratch
            probs = _softmax(y_pred)
            batch_size = y_true.shape[0]
            if self.loss_name in ("cross_entropy",):
                # y_true may be one-hot or integer labels
                if y_true.ndim == 2:
                    dZ = (probs - y_true) / batch_size
                else:
                    dZ = probs.copy()
                    dZ[np.arange(batch_size), y_true] -= 1
                    dZ /= batch_size
            else:  # mse
                dA = (probs - y_true) * (2.0 / y_true.shape[1])
                dZ = probs * (dA - np.sum(dA * probs, axis=1, keepdims=True))
                dZ /= batch_size

            # Output layer manual grad
            out_layer = self.layers[-1]
            out_layer.grad_W = (out_layer.A_prev.T if hasattr(out_layer, 'A_prev') else out_layer.x.T) @ dZ
            out_layer.grad_b = np.sum(dZ, axis=0, keepdims=True)
            grad = dZ @ out_layer.W.T
            for layer in reversed(self.layers[:-1]):
                grad = layer.backward(grad)
        else:
            # Our style: use cached loss gradient
            grad = self.loss_fn.backward()
            if grad is None:
                batch = getattr(self, '_last_batch_size', 1)
                grad = np.zeros((batch, self.output_size))
            for layer in reversed(self.layers):
                grad = layer.backward(grad)


    def get_weights(self):
        """Return weights as dict {W0, b0, W1, b1, ...}."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weights):
        """
        Accept dict {W0,b0,...} OR list formats for backward compatibility.
        Primary format (matching autograder): dict with keys W0,b0,W1,b1,...
        """
        if isinstance(weights, dict):
            for i, layer in enumerate(self.layers):
                if f"W{i}" in weights:
                    layer.W = np.array(weights[f"W{i}"]).copy()
                if f"b{i}" in weights:
                    layer.b = np.array(weights[f"b{i}"]).copy()
            return
        # Fallback: flat list [W0,b0,W1,b1,...]
        weights = list(weights)
        # Skip leading scalar metadata if present
        while len(weights) > 0 and np.array(weights[0]).ndim == 0:
            weights = weights[1:]
        if len(weights) == 2 * len(self.layers):
            for i, layer in enumerate(self.layers):
                layer.W = np.array(weights[2*i]).copy()
                layer.b = np.array(weights[2*i+1]).copy()
        else:
            for i, (layer, w) in enumerate(zip(self.layers, weights)):
                w = list(w)
                layer.W = np.array(w[0]).copy()
                layer.b = np.array(w[1]).copy()

    def save(self, path):
        """Save weights as dict {W0,b0,W1,b1,...} — matches autograder format."""
        import os
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.get_weights())
        print(f"Model saved → {path}")

    def load(self, path):
        """Load weights from dict-format .npy file."""
        data = np.load(path, allow_pickle=True).item()
        self.set_weights(data)
        print(f"Model loaded ← {path}")
