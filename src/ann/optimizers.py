# src/ann/optimizers.py
# All six optimizers required by the assignment.
# Each optimizer implements step(layers) which updates W and b
# for every layer using stored gradients (grad_W, grad_b).
#
# Optimizers: SGD, MomentumSGD, NAG, RMSProp, Adam, Nadam

import numpy as np


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, lr: float, weight_decay: float = 0.0):
        """
        Args:
            lr:           Learning rate.
            weight_decay: L2 regularization coefficient (lambda).
        """
        self.lr           = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent.
    W = W - lr * (grad_W + weight_decay * W)

    Simple but slow — uses same lr for all parameters,
    no momentum to escape local minima.
    """

    def step(self, layers):
        for layer in layers:
            # Apply L2 regularization to weight gradient
            layer.grad_W += self.weight_decay * layer.W
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


class MomentumSGD(Optimizer):
    """
    SGD with Momentum.
    Accumulates a velocity vector in directions of persistent gradient.
    Helps escape saddle points and accelerates convergence.

    v = beta * v + grad
    W = W - lr * v
    """

    def __init__(self, lr, weight_decay=0.0, beta=0.9):
        super().__init__(lr, weight_decay)
        self.beta = beta   # momentum coefficient (typically 0.9)

    def step(self, layers):
        for layer in layers:
            state = layer.optimizer_state

            # Initialise velocity on first call
            if "v_W" not in state:
                state["v_W"] = np.zeros_like(layer.W)
                state["v_b"] = np.zeros_like(layer.b)

            layer.grad_W += self.weight_decay * layer.W

            # Update velocity and apply
            state["v_W"] = self.beta * state["v_W"] + layer.grad_W
            state["v_b"] = self.beta * state["v_b"] + layer.grad_b

            layer.W -= self.lr * state["v_W"]
            layer.b -= self.lr * state["v_b"]


class NAG(Optimizer):
    """
    Nesterov Accelerated Gradient.
    Looks ahead before computing gradient — more accurate direction.
    Slightly better convergence than standard momentum.

    v_new = beta * v + grad
    W = W - lr * ((1 + beta) * v_new - beta * v_old)
    """

    def __init__(self, lr, weight_decay=0.0, beta=0.9):
        super().__init__(lr, weight_decay)
        self.beta = beta

    def step(self, layers):
        for layer in layers:
            state = layer.optimizer_state

            if "v_W" not in state:
                state["v_W"] = np.zeros_like(layer.W)
                state["v_b"] = np.zeros_like(layer.b)

            layer.grad_W += self.weight_decay * layer.W

            # Save old velocity before update
            v_W_prev = state["v_W"].copy()
            v_b_prev = state["v_b"].copy()

            state["v_W"] = self.beta * state["v_W"] + layer.grad_W
            state["v_b"] = self.beta * state["v_b"] + layer.grad_b

            # Nesterov update: look-ahead correction
            layer.W -= self.lr * (
                (1 + self.beta) * state["v_W"] - self.beta * v_W_prev
            )
            layer.b -= self.lr * (
                (1 + self.beta) * state["v_b"] - self.beta * v_b_prev
            )


class RMSProp(Optimizer):
    """
    RMSProp — Root Mean Square Propagation.
    Adapts learning rate per parameter by dividing by
    exponential moving average of squared gradients.
    Good for non-stationary objectives.

    s = beta * s + (1 - beta) * grad^2
    W = W - lr * grad / (sqrt(s) + eps)
    """

    def __init__(self, lr, weight_decay=0.0, beta=0.9, eps=1e-8):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self.eps  = eps

    def step(self, layers):
        for layer in layers:
            state = layer.optimizer_state

            if "s_W" not in state:
                state["s_W"] = np.zeros_like(layer.W)
                state["s_b"] = np.zeros_like(layer.b)

            layer.grad_W += self.weight_decay * layer.W

            # Update exponential moving average of squared gradients
            state["s_W"] = (
                self.beta * state["s_W"] + (1 - self.beta) * layer.grad_W ** 2
            )
            state["s_b"] = (
                self.beta * state["s_b"] + (1 - self.beta) * layer.grad_b ** 2
            )

            layer.W -= self.lr * layer.grad_W / (np.sqrt(state["s_W"]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(state["s_b"]) + self.eps)


class Adam(Optimizer):
    """
    Adam — Adaptive Moment Estimation.
    Combines momentum (1st moment) and RMSProp (2nd moment).
    Bias-corrected to handle zero-initialization of moments.

    m = beta1 * m + (1 - beta1) * grad          # 1st moment (mean)
    v = beta2 * v + (1 - beta2) * grad^2        # 2nd moment (variance)
    m_hat = m / (1 - beta1^t)                   # bias correction
    v_hat = v / (1 - beta2^t)
    W = W - lr * m_hat / (sqrt(v_hat) + eps)

    Default betas (0.9, 0.999) work well for most problems.
    Most popular optimizer for deep learning.
    """

    def __init__(self, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0     # time step for bias correction

    def step(self, layers):
        self.t += 1
        for layer in layers:
            state = layer.optimizer_state

            if "m_W" not in state:
                state["m_W"] = np.zeros_like(layer.W)
                state["v_W"] = np.zeros_like(layer.W)
                state["m_b"] = np.zeros_like(layer.b)
                state["v_b"] = np.zeros_like(layer.b)

            layer.grad_W += self.weight_decay * layer.W

            # Update biased moments
            state["m_W"] = self.beta1 * state["m_W"] + (1 - self.beta1) * layer.grad_W
            state["v_W"] = self.beta2 * state["v_W"] + (1 - self.beta2) * layer.grad_W ** 2
            state["m_b"] = self.beta1 * state["m_b"] + (1 - self.beta1) * layer.grad_b
            state["v_b"] = self.beta2 * state["v_b"] + (1 - self.beta2) * layer.grad_b ** 2

            # Bias correction
            m_W_hat = state["m_W"] / (1 - self.beta1 ** self.t)
            v_W_hat = state["v_W"] / (1 - self.beta2 ** self.t)
            m_b_hat = state["m_b"] / (1 - self.beta1 ** self.t)
            v_b_hat = state["v_b"] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


class Nadam(Optimizer):
    """
    Nadam — Nesterov Adam.
    Adam with Nesterov momentum correction.
    Uses look-ahead gradient for the 1st moment update,
    giving slightly faster convergence than standard Adam.
    """

    def __init__(self, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0

    def step(self, layers):
        self.t += 1
        for layer in layers:
            state = layer.optimizer_state

            if "m_W" not in state:
                state["m_W"] = np.zeros_like(layer.W)
                state["v_W"] = np.zeros_like(layer.W)
                state["m_b"] = np.zeros_like(layer.b)
                state["v_b"] = np.zeros_like(layer.b)

            layer.grad_W += self.weight_decay * layer.W

            state["m_W"] = self.beta1 * state["m_W"] + (1 - self.beta1) * layer.grad_W
            state["v_W"] = self.beta2 * state["v_W"] + (1 - self.beta2) * layer.grad_W ** 2
            state["m_b"] = self.beta1 * state["m_b"] + (1 - self.beta1) * layer.grad_b
            state["v_b"] = self.beta2 * state["v_b"] + (1 - self.beta2) * layer.grad_b ** 2

            m_W_hat = state["m_W"] / (1 - self.beta1 ** self.t)
            v_W_hat = state["v_W"] / (1 - self.beta2 ** self.t)
            m_b_hat = state["m_b"] / (1 - self.beta1 ** self.t)
            v_b_hat = state["v_b"] / (1 - self.beta2 ** self.t)

            # Nesterov correction: look-ahead on 1st moment
            m_W_nesterov = (
                self.beta1 * m_W_hat
                + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.t)
            )
            m_b_nesterov = (
                self.beta1 * m_b_hat
                + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t)
            )

            layer.W -= self.lr * m_W_nesterov / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_nesterov / (np.sqrt(v_b_hat) + self.eps)


def get_optimizer(name: str, lr: float, weight_decay: float = 0.0, **kwargs):
    """
    Factory function — returns an optimizer object by name.

    Args:
        name:         One of sgd, momentum, nag, rmsprop, adam, nadam
        lr:           Learning rate
        weight_decay: L2 regularization coefficient
    Returns:
        Optimizer object with step(layers) method
    """
    optimizers = {
        "sgd":      SGD,
        "momentum": MomentumSGD,
        "nag":      NAG,
        "rmsprop":  RMSProp,
        "adam":     Adam,
        "nadam":    Nadam,
    }
    name = name.lower()
    if name not in optimizers:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: {list(optimizers.keys())}"
        )
    return optimizers[name](lr=lr, weight_decay=weight_decay, **kwargs)
