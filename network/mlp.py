import numpy as np
from .layers import DenseLayer
from .activations import get_activation, Identity
from .losses import get_loss, softmax


class MLP:
    """
    Configurable Multi-Layer Perceptron built with NumPy.

    Architecture:
        Input  →  [Hidden Layer × num_hidden_layers]  →  Output (softmax)
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        activation: str = "relu",
        weight_init: str = "xavier",
        loss: str = "cross_entropy",
    ):
        """
        Args:
            input_size:    Dimension of the flattened input.
            hidden_sizes:  List of neuron counts for each hidden layer.
            output_size:   Number of output classes.
            activation:    Activation function name for hidden layers.
            weight_init:   Weight initialisation strategy ('random' | 'xavier').
            loss:          Loss function name ('cross_entropy' | 'mean_squared_error').
        """
        self.layers = []
        self.loss_fn = get_loss(loss)

        sizes = [input_size] + list(hidden_sizes) + [output_size]

        for i in range(len(sizes) - 1):
            # Last layer uses Identity (loss handles softmax internally)
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            self.layers.append(
                DenseLayer(sizes[i], sizes[i + 1], activation=act, weight_init=weight_init)
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input of shape (batch, input_size).
        Returns:
            Logits of shape (batch, output_size).
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out  # raw logits

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        return softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(x), axis=1)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def compute_loss(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_fn.forward(logits, y_true)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self):
        """
        Backpropagate from loss through all layers.
        Gradients are stored in each layer's .grad_W and .grad_b attributes.
        """
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def get_weights(self) -> list:
        """Return list of (W, b) tuples for all layers."""
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights: list):
        """Load weights from list of (W, b) tuples."""
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = W.copy()
            layer.b = b.copy()

    def save(self, path: str):
        """Save all layer weights to a .npy file."""
        weights = self.get_weights()
        np.save(path, np.array(weights, dtype=object), allow_pickle=True)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load layer weights from a .npy file."""
        weights = np.load(path, allow_pickle=True)
        self.set_weights(list(weights))
        print(f"Model loaded from {path}")
