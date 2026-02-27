import numpy as np


class DenseLayer:
    """
    Fully connected (dense) layer.
    Stores weights, biases, and their gradients after backward pass.
    """

    def __init__(self, input_size: int, output_size: int, activation, weight_init: str = "xavier"):
        """
        Args:
            input_size:   Number of input features.
            output_size:  Number of neurons in this layer.
            activation:   Activation object (must have forward/backward methods).
            weight_init:  'random' or 'xavier'.
        """
        self.activation = activation
        self._init_weights(input_size, output_size, weight_init)

        # Gradients (populated after backward())
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        # Optimizer state placeholders (used by stateful optimizers)
        self.optimizer_state = {}

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self, fan_in: int, fan_out: int, method: str):
        if method == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.W = np.random.uniform(-limit, limit, (fan_in, fan_out))
        elif method == "random":
            self.W = np.random.randn(fan_in, fan_out) * 0.01
        else:
            raise ValueError(f"Unknown weight init '{method}'. Choose 'random' or 'xavier'.")
        self.b = np.zeros((1, fan_out))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input of shape (batch, input_size).
        Returns:
            Activated output of shape (batch, output_size).
        """
        self.x = x                          # cache for backward
        self.z = x @ self.W + self.b        # pre-activation
        return self.activation.forward(self.z)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients and propagate upstream gradient.

        Args:
            grad_output: Gradient w.r.t. layer output, shape (batch, output_size).
        Returns:
            Gradient w.r.t. layer input, shape (batch, input_size).
        """
        batch_size = self.x.shape[0]

        # Gradient through activation
        grad_z = self.activation.backward(grad_output)   # (batch, output_size)

        # Parameter gradients (averaged over batch)
        self.grad_W = self.x.T @ grad_z / batch_size     # (input_size, output_size)
        self.grad_b = grad_z.mean(axis=0, keepdims=True) # (1, output_size)

        # Propagate gradient to previous layer
        return grad_z @ self.W.T                          # (batch, input_size)
