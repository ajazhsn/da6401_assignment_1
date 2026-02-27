import numpy as np


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Optimizer:
    def __init__(self, lr: float, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------
class SGD(Optimizer):
    """Vanilla Stochastic Gradient Descent."""

    def step(self, layers):
        for layer in layers:
            # L2 regularization
            layer.grad_W += self.weight_decay * layer.W
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


# ---------------------------------------------------------------------------
# Momentum SGD
# ---------------------------------------------------------------------------
class MomentumSGD(Optimizer):
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

            state["v_W"] = self.beta * state["v_W"] + layer.grad_W
            state["v_b"] = self.beta * state["v_b"] + layer.grad_b

            layer.W -= self.lr * state["v_W"]
            layer.b -= self.lr * state["v_b"]


# ---------------------------------------------------------------------------
# Nesterov Accelerated Gradient (NAG)
# ---------------------------------------------------------------------------
class NAG(Optimizer):
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

            v_W_prev = state["v_W"].copy()
            v_b_prev = state["v_b"].copy()

            state["v_W"] = self.beta * state["v_W"] + layer.grad_W
            state["v_b"] = self.beta * state["v_b"] + layer.grad_b

            layer.W -= self.lr * ((1 + self.beta) * state["v_W"] - self.beta * v_W_prev)
            layer.b -= self.lr * ((1 + self.beta) * state["v_b"] - self.beta * v_b_prev)


# ---------------------------------------------------------------------------
# RMSProp
# ---------------------------------------------------------------------------
class RMSProp(Optimizer):
    def __init__(self, lr, weight_decay=0.0, beta=0.9, eps=1e-8):
        super().__init__(lr, weight_decay)
        self.beta = beta
        self.eps = eps

    def step(self, layers):
        for layer in layers:
            state = layer.optimizer_state
            if "s_W" not in state:
                state["s_W"] = np.zeros_like(layer.W)
                state["s_b"] = np.zeros_like(layer.b)

            layer.grad_W += self.weight_decay * layer.W

            state["s_W"] = self.beta * state["s_W"] + (1 - self.beta) * layer.grad_W ** 2
            state["s_b"] = self.beta * state["s_b"] + (1 - self.beta) * layer.grad_b ** 2

            layer.W -= self.lr * layer.grad_W / (np.sqrt(state["s_W"]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(state["s_b"]) + self.eps)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
class Adam(Optimizer):
    def __init__(self, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

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

            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)


# ---------------------------------------------------------------------------
# Nadam (Nesterov Adam)
# ---------------------------------------------------------------------------
class Nadam(Optimizer):
    def __init__(self, lr, weight_decay=0.0, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

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

            # Nesterov correction
            m_W_nesterov = self.beta1 * m_W_hat + (1 - self.beta1) * layer.grad_W / (1 - self.beta1 ** self.t)
            m_b_nesterov = self.beta1 * m_b_hat + (1 - self.beta1) * layer.grad_b / (1 - self.beta1 ** self.t)

            layer.W -= self.lr * m_W_nesterov / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_nesterov / (np.sqrt(v_b_hat) + self.eps)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_optimizer(name: str, lr: float, weight_decay: float = 0.0, **kwargs):
    """Return optimizer object by name."""
    optimizers = {
        "sgd": SGD,
        "momentum": MomentumSGD,
        "nag": NAG,
        "rmsprop": RMSProp,
        "adam": Adam,
        "nadam": Nadam,
    }
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from: {list(optimizers.keys())}")
    return optimizers[name](lr=lr, weight_decay=weight_decay, **kwargs)
