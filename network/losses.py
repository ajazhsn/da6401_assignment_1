import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class CrossEntropyLoss:
    """
    Softmax + Cross-Entropy combined for numerical stability.
    Expects raw logits as input.
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Args:
            logits: (batch, num_classes) raw network output.
            y_true: (batch,) integer class labels.
        Returns:
            Scalar mean loss.
        """
        self.probs = softmax(logits)           # (batch, C)
        self.y_true = y_true
        batch_size = logits.shape[0]
        log_likelihood = -np.log(self.probs[np.arange(batch_size), y_true] + 1e-9)
        return log_likelihood.mean()

    def backward(self) -> np.ndarray:
        """Returns gradient of loss w.r.t. logits."""
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.y_true] -= 1
        return grad / batch_size              # (batch, C)


class MSELoss:
    """
    Mean Squared Error loss with one-hot encoding of targets.
    Expects raw logits; applies softmax internally for gradient computation.
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Args:
            logits: (batch, num_classes) raw network output.
            y_true: (batch,) integer class labels.
        Returns:
            Scalar mean loss.
        """
        self.probs = softmax(logits)
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]

        # One-hot encode targets
        self.one_hot = np.zeros((batch_size, num_classes))
        self.one_hot[np.arange(batch_size), y_true] = 1

        diff = self.probs - self.one_hot
        return np.mean(diff ** 2)

    def backward(self) -> np.ndarray:
        """Returns gradient of MSE loss w.r.t. logits (through softmax)."""
        batch_size = self.probs.shape[0]
        diff = self.probs - self.one_hot                          # (batch, C)

        # Gradient of MSE through softmax: dL/dz_i = sum_j(dL/dp_j * dp_j/dz_i)
        # dp_j/dz_i = p_i*(delta_ij - p_j) => combined gradient below
        grad = np.zeros_like(self.probs)
        for i in range(batch_size):
            p = self.probs[i]                                     # (C,)
            d = diff[i]                                           # (C,)
            # Jacobian of softmax
            jacobian = np.diag(p) - np.outer(p, p)               # (C, C)
            grad[i] = (2.0 / self.probs.shape[1]) * jacobian @ d

        return grad / batch_size


def get_loss(name: str):
    """Return loss object by name."""
    losses = {
        "cross_entropy": CrossEntropyLoss,
        "mean_squared_error": MSELoss,
        "mse": MSELoss,
    }
    name = name.lower()
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(losses.keys())}")
    return losses[name]()
