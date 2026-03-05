# src/ann/objective_functions.py
# Loss functions for multi-class classification.
# Both losses expect raw logits from the network output layer
# and apply softmax internally for numerical stability.
#
# Each loss class implements:
#   - forward(logits, y_true) : computes scalar loss
#   - backward()              : returns gradient w.r.t. logits

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    Subtracts max per sample to prevent exp overflow.

    Args:
        z: Raw logits of shape (batch, num_classes)
    Returns:
        Probability distribution of shape (batch, num_classes)
    """
    z_shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


class CrossEntropyLoss:
    """
    Softmax + Cross-Entropy loss (combined for stability).

    CE loss = -log(p_correct_class)
    This is the natural loss for multi-class classification —
    derived from maximum likelihood estimation over a categorical distribution.
    Gradient w.r.t. logits is simply: (softmax_probs - one_hot_labels) / batch
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Args:
            logits: (batch, num_classes) raw network output
            y_true: (batch,) integer class labels
        Returns:
            Scalar mean cross-entropy loss
        """
        self.probs  = softmax(logits)    # (batch, C)
        self.y_true = y_true
        batch_size  = logits.shape[0]

        # Log probability of the correct class for each sample
        log_likelihood = -np.log(
            self.probs[np.arange(batch_size), y_true] + 1e-9
        )
        return log_likelihood.mean()

    def backward(self) -> np.ndarray:
        """
        Gradient of CE loss w.r.t. logits.
        Elegant result: (softmax_probs - one_hot) / batch_size
        """
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.y_true] -= 1
        return grad / batch_size          # (batch, C)


class MSELoss:
    """
    Mean Squared Error loss for classification.

    Applies softmax to get probabilities, then computes MSE
    against one-hot encoded targets.

    Note: MSE is suboptimal for classification — gradients are small
    early in training, causing slow convergence compared to CrossEntropy.
    """

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Args:
            logits: (batch, num_classes) raw network output
            y_true: (batch,) integer class labels
        Returns:
            Scalar mean squared error loss
        """
        self.probs     = softmax(logits)
        batch_size     = logits.shape[0]
        num_classes    = logits.shape[1]

        # Build one-hot target matrix
        self.one_hot = np.zeros((batch_size, num_classes))
        self.one_hot[np.arange(batch_size), y_true] = 1

        diff = self.probs - self.one_hot
        return np.mean(diff ** 2)

    def backward(self) -> np.ndarray:
        """
        Gradient of MSE w.r.t. logits — passed through the softmax Jacobian.
        More expensive than CrossEntropy gradient due to the Jacobian computation.
        """
        batch_size  = self.probs.shape[0]
        num_classes = self.probs.shape[1]
        diff        = self.probs - self.one_hot   # (batch, C)

        grad = np.zeros_like(self.probs)
        for i in range(batch_size):
            p = self.probs[i]                     # (C,)
            d = diff[i]                           # (C,)
            # Jacobian of softmax: diag(p) - p * p^T
            jacobian = np.diag(p) - np.outer(p, p)
            grad[i]  = (2.0 / num_classes) * jacobian @ d

        return grad / batch_size


def get_loss(name: str):
    """
    Factory function — returns a loss object by name.

    Args:
        name: 'cross_entropy' or 'mean_squared_error' (or 'mse')
    Returns:
        Loss object with forward() and backward() methods
    """
    losses = {
        "cross_entropy":      CrossEntropyLoss,
        "mean_squared_error": MSELoss,
        "mse":                MSELoss,
    }
    name = name.lower()
    if name not in losses:
        raise ValueError(
            f"Unknown loss '{name}'. Choose from: {list(losses.keys())}"
        )
    return losses[name]()
