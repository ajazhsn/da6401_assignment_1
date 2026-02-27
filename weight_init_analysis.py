"""
weight_init_analysis.py — Q2.9: Zeros vs Xavier initialization.

Logs gradients of 5 neurons in the same hidden layer for first 50 iterations.
Run:
    python weight_init_analysis.py
"""

import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from network.mlp import MLP
from network.optimizers import get_optimizer
from network.activations import get_activation, Identity
from network.layers import DenseLayer


# -----------------------------------------------------------------------
# Special MLP with Zeros init support
# -----------------------------------------------------------------------
class MLPWithZeros(MLP):
    """Overrides weight init to support 'zeros' strategy."""

    def __init__(self, input_size, hidden_sizes, output_size,
                 activation="relu", weight_init="xavier", loss="cross_entropy"):
        # Build layers manually to support zeros init
        from network.losses import get_loss
        self.layers = []
        self.loss_fn = get_loss(loss)

        sizes = [input_size] + list(hidden_sizes) + [output_size]
        for i in range(len(sizes) - 1):
            act = get_activation(activation) if i < len(sizes) - 2 else Identity()
            layer = DenseLayer(sizes[i], sizes[i + 1], activation=act,
                               weight_init="xavier")   # init normally first
            if weight_init == "zeros":
                layer.W = np.zeros_like(layer.W)
                layer.b = np.zeros_like(layer.b)
            self.layers.append(layer)


CONFIGS = [
    {"weight_init": "zeros",  "label": "zeros-init"},
    {"weight_init": "xavier", "label": "xavier-init"},
]

BASE_CONFIG = {
    "dataset":       "fashion_mnist",
    "epochs":        5,               # Only need 50 iterations worth
    "batch_size":    64,
    "loss":          "cross_entropy",
    "optimizer":     "sgd",
    "learning_rate": 0.01,
    "weight_decay":  0.0,
    "hidden_sizes":  [128, 64],
    "activation":    "relu",
    "wandb_project": "da6401-assignment1",
    "monitor_layer": 0,               # First hidden layer
    "monitor_neurons": [0, 1, 2, 3, 4],  # 5 neurons to track
}


def load_data():
    from keras.datasets import fashion_mnist
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train, y_train


def train_and_log(cfg, x_train, y_train, x_val, y_val):
    run = wandb.init(
        project=BASE_CONFIG["wandb_project"],
        name=f"Q2.9-{cfg['label']}",
        group="Q2.9-weight-init",
        config={**BASE_CONFIG, **cfg},
        reinit=True,
    )

    model = MLPWithZeros(
        input_size=784,
        hidden_sizes=BASE_CONFIG["hidden_sizes"],
        output_size=10,
        activation=BASE_CONFIG["activation"],
        weight_init=cfg["weight_init"],
        loss=BASE_CONFIG["loss"],
    )
    optimizer = get_optimizer(BASE_CONFIG["optimizer"], lr=BASE_CONFIG["learning_rate"])

    iteration = 0
    monitor_layer = BASE_CONFIG["monitor_layer"]
    monitor_neurons = BASE_CONFIG["monitor_neurons"]

    for epoch in range(1, BASE_CONFIG["epochs"] + 1):
        idx = np.random.permutation(x_train.shape[0])
        xtr, ytr = x_train[idx], y_train[idx]

        num_batches = int(np.ceil(xtr.shape[0] / BASE_CONFIG["batch_size"]))
        for b in range(num_batches):
            xb = xtr[b * BASE_CONFIG["batch_size"]: (b + 1) * BASE_CONFIG["batch_size"]]
            yb = ytr[b * BASE_CONFIG["batch_size"]: (b + 1) * BASE_CONFIG["batch_size"]]

            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb)
            model.backward()
            optimizer.step(model.layers)

            iteration += 1

            # Log gradients for 5 specific neurons in the monitor layer
            layer_grad = model.layers[monitor_layer].grad_W  # (input_size, output_size)
            neuron_grads = {
                f"neuron_{n}_grad_norm": float(np.linalg.norm(layer_grad[:, n]))
                for n in monitor_neurons
            }

            # Also log overall gradient norm
            overall_norm = float(np.linalg.norm(layer_grad))

            log_data = {
                "iteration":         iteration,
                "loss":              float(loss),
                "overall_grad_norm": overall_norm,
                **neuron_grads,
            }

            if iteration <= 50:
                print(f"  [{cfg['label']}] iter={iteration} loss={loss:.4f} "
                      f"grad_norm={overall_norm:.6f}")

            wandb.log(log_data)

            if iteration >= 50:
                # After 50 iterations, only log epoch-level
                pass

        # Epoch-level val accuracy
        val_logits = model.forward(x_val)
        val_acc = (np.argmax(val_logits, axis=1) == y_val).mean()
        wandb.log({"epoch": epoch, "val_accuracy": val_acc})
        print(f"  [{cfg['label']}] Epoch {epoch} val_acc={val_acc:.4f}")

    run.finish()


def main():
    print("Loading data...")
    x_train_full, y_train_full = load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    for cfg in CONFIGS:
        print(f"\n{'='*50}\nConfig: {cfg['label']}\n{'='*50}")
        train_and_log(cfg, x_train, y_train, x_val, y_val)

    print("\nDone! Check W&B. In zeros-init, all 5 neuron gradients should overlap perfectly.")


if __name__ == "__main__":
    main()
