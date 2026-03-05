# src/ann/neural_network.py
# Defines the NeuralNetwork (MLP) class.
# Stacks NeuralLayer objects and orchestrates forward/backward passes.
# Also handles model serialisation (save/load as .npy).

import numpy as np
from .neural_layer import NeuralLayer
from .activations import get_activation, Identity
from .objective_functions import get_loss, softmax


class NeuralNetwork:
    """
    Configurable Multi-Layer Perceptron (MLP) built entirely with NumPy.

    Architecture:
        Input → [Hidden Layers with activation] → Output (Identity, softmax in loss)

    Usage:
        model = NeuralNetwork(784, [128, 128], 10, activation='relu')
        logits = model.forward(x_batch)
        loss   = model.compute_loss(logits, y_batch)
        model.backward()
        optimizer.step(model.layers)
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
            input_size:   Number of input features (e.g. 784 for 28x28 images).
            hidden_sizes: List of neuron counts per hidden layer (e.g. [128, 128]).
            output_size:  Number of output classes (e.g. 10 for Fashion-MNIST).
            activation:   Activation for hidden layers: 'sigmoid' | 'tanh' | 'relu'.
            weight_init:  Weight init strategy: 'xavier' | 'random'.
            loss:         Loss function: 'cross_entropy' | 'mean_squared_error'.
        """
        self.layers  = []
        self.loss_fn = get_loss(loss)

        # Build layer stack: input → hidden... → output
        sizes = [input_size] + list(hidden_sizes) + [output_size]

        for i in range(len(sizes) - 1):
            # Output layer uses Identity activation — loss handles softmax
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            self.layers.append(
                NeuralLayer(
                    input_size=sizes[i],
                    output_size=sizes[i + 1],
                    activation=act,
                    weight_init=weight_init,
                )
            )

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Run input through all layers sequentially.

        Args:
            x: Input array of shape (batch_size, input_size)
        Returns:
            Raw logits of shape (batch_size, output_size)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out   # raw logits — softmax applied inside loss

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return softmax probabilities over classes."""
        return softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted class index for each sample."""
        return np.argmax(self.predict_proba(x), axis=1)

    # ------------------------------------------------------------------
    # Loss Computation
    # ------------------------------------------------------------------
    def compute_loss(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute scalar loss and cache state for backward pass.

        Args:
            logits: Raw network output (batch, output_size)
            y_true: Integer class labels (batch,)
        Returns:
            Scalar loss value
        """
        return self.loss_fn.forward(logits, y_true)

    # ------------------------------------------------------------------
    # Backward Pass
    # ------------------------------------------------------------------
    def backward(self):
        """
        Backpropagate gradients from loss through all layers in reverse.
        After this call, each layer's .grad_W and .grad_b are populated
        and ready for the optimizer to use.
        """
        grad = self.loss_fn.backward()          # gradient w.r.t. output logits
        for layer in reversed(self.layers):
            grad = layer.backward(grad)         # propagate backward layer by layer

    # ------------------------------------------------------------------
    # Model Serialisation
    # ------------------------------------------------------------------
    def get_weights(self) -> list:
        """Return all layer weights as list of (W, b) tuples."""
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights: list):
        """Load weights from list of (W, b) tuples."""
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = W.copy()
            layer.b = b.copy()

    def save(self, path: str):
        """
        Serialize all layer weights to a .npy file.
        Uses allow_pickle=True to store list of arrays.
        """
        weights = self.get_weights()
        np.save(path, np.array(weights, dtype=object), allow_pickle=True)
        print(f"Model saved → {path}")

    def load(self, path: str):
        """Load layer weights from a previously saved .npy file."""
        weights = np.load(path, allow_pickle=True)
        self.set_weights(list(weights))
        print(f"Model loaded ← {path}")
