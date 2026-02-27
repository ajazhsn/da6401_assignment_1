"""
gradient_analysis.py — Q2.4: Vanishing gradient analysis (Sigmoid vs ReLU).

Logs gradient norms for the first hidden layer across epochs.
Run:
    python gradient_analysis.py
"""

import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from network.mlp import MLP
from network.optimizers import get_optimizer


CONFIGS = [
    {"activation": "sigmoid", "hidden_sizes": [128, 128, 128], "label": "3L-sigmoid"},
    {"activation": "relu",    "hidden_sizes": [128, 128, 128], "label": "3L-relu"},
    {"activation": "sigmoid", "hidden_sizes": [128, 128, 128, 128, 128], "label": "5L-sigmoid"},
    {"activation": "relu",    "hidden_sizes": [128, 128, 128, 128, 128], "label": "5L-relu"},
]

BASE_CONFIG = {
    "dataset":       "fashion_mnist",
    "epochs":        15,
    "batch_size":    64,
    "loss":          "cross_entropy",
    "optimizer":     "adam",
    "learning_rate": 0.001,
    "weight_decay":  0.0,
    "weight_init":   "xavier",
    "wandb_project": "da6401-assignment1",
}


def load_data():
    from keras.datasets import fashion_mnist
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train, y_train


def train_and_log_gradients(cfg, x_train, y_train, x_val, y_val):
    run = wandb.init(
        project=BASE_CONFIG["wandb_project"],
        name=f"Q2.4-{cfg['label']}",
        group="Q2.4-vanishing-gradient",
        config={**BASE_CONFIG, **cfg},
        reinit=True,
    )

    model = MLP(
        input_size=784,
        hidden_sizes=cfg["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init=BASE_CONFIG["weight_init"],
        loss=BASE_CONFIG["loss"],
    )
    optimizer = get_optimizer(BASE_CONFIG["optimizer"], lr=BASE_CONFIG["learning_rate"])
    num_batches = int(np.ceil(x_train.shape[0] / BASE_CONFIG["batch_size"]))

    for epoch in range(1, BASE_CONFIG["epochs"] + 1):
        idx = np.random.permutation(x_train.shape[0])
        xtr, ytr = x_train[idx], y_train[idx]

        epoch_loss, correct = 0.0, 0
        for b in range(num_batches):
            xb = xtr[b * BASE_CONFIG["batch_size"]: (b + 1) * BASE_CONFIG["batch_size"]]
            yb = ytr[b * BASE_CONFIG["batch_size"]: (b + 1) * BASE_CONFIG["batch_size"]]
            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb)
            model.backward()
            optimizer.step(model.layers)
            epoch_loss += loss * xb.shape[0]
            correct    += (np.argmax(logits, axis=1) == yb).sum()

        train_loss = epoch_loss / x_train.shape[0]
        train_acc  = correct   / x_train.shape[0]
        val_logits = model.forward(x_val)
        val_acc    = (np.argmax(val_logits, axis=1) == y_val).mean()

        # Log gradient norm of FIRST hidden layer
        first_layer_grad_norm = np.linalg.norm(model.layers[0].grad_W)

        # Log gradient norms of ALL layers for comparison
        layer_grad_norms = {
            f"grad_norm_layer_{i+1}": np.linalg.norm(layer.grad_W)
            for i, layer in enumerate(model.layers[:-1])  # exclude output layer
        }

        print(f"  [{cfg['label']}] Epoch {epoch}/{BASE_CONFIG['epochs']} "
              f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
              f"grad_norm_L1={first_layer_grad_norm:.6f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "grad_norm_first_layer": first_layer_grad_norm,
            **layer_grad_norms,
        })

    run.finish()


def main():
    print("Loading data...")
    x_train_full, y_train_full = load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    for cfg in CONFIGS:
        print(f"\n{'='*50}\nConfig: {cfg['label']}\n{'='*50}")
        train_and_log_gradients(cfg, x_train, y_train, x_val, y_val)

    print("\nDone! Check W&B for gradient norm plots.")


if __name__ == "__main__":
    main()
