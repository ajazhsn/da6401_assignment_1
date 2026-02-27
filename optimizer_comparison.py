"""
optimizer_comparison.py — Q2.3: Compare all 6 optimizers with same architecture.

Fixed config: 3 hidden layers, 128 neurons each, ReLU activation.
Run:
    python optimizer_comparison.py
"""

import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from network.mlp import MLP
from network.optimizers import get_optimizer


OPTIMIZERS = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]

CONFIG = {
    "dataset":       "fashion_mnist",
    "epochs":        10,
    "batch_size":    64,
    "loss":          "cross_entropy",
    "learning_rate": 0.001,
    "weight_decay":  0.0,
    "hidden_sizes":  [128, 128, 128],
    "activation":    "relu",
    "weight_init":   "xavier",
    "wandb_project": "da6401-assignment1",
}


def load_data():
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 784).astype(np.float32)  / 255.0
    return x_train, y_train, x_test, y_test


def train_one_optimizer(optimizer_name, x_train, y_train, x_val, y_val):
    run = wandb.init(
        project=CONFIG["wandb_project"],
        name=f"Q2.3-optimizer-{optimizer_name}",
        group="Q2.3-optimizer-showdown",
        config={**CONFIG, "optimizer": optimizer_name},
        reinit=True,
    )

    model = MLP(
        input_size=784,
        hidden_sizes=CONFIG["hidden_sizes"],
        output_size=10,
        activation=CONFIG["activation"],
        weight_init=CONFIG["weight_init"],
        loss=CONFIG["loss"],
    )
    optimizer = get_optimizer(optimizer_name, lr=CONFIG["learning_rate"],
                              weight_decay=CONFIG["weight_decay"])

    num_batches = int(np.ceil(x_train.shape[0] / CONFIG["batch_size"]))

    for epoch in range(1, CONFIG["epochs"] + 1):
        idx = np.random.permutation(x_train.shape[0])
        xtr, ytr = x_train[idx], y_train[idx]

        epoch_loss, correct = 0.0, 0
        for b in range(num_batches):
            xb = xtr[b * CONFIG["batch_size"]: (b + 1) * CONFIG["batch_size"]]
            yb = ytr[b * CONFIG["batch_size"]: (b + 1) * CONFIG["batch_size"]]
            logits = model.forward(xb)
            loss   = model.compute_loss(logits, yb)
            model.backward()
            optimizer.step(model.layers)
            epoch_loss += loss * xb.shape[0]
            correct    += (np.argmax(logits, axis=1) == yb).sum()

        train_loss = epoch_loss / x_train.shape[0]
        train_acc  = correct   / x_train.shape[0]
        val_logits = model.forward(x_val)
        val_loss   = model.compute_loss(val_logits, y_val)
        val_acc    = (np.argmax(val_logits, axis=1) == y_val).mean()

        print(f"  [{optimizer_name}] Epoch {epoch}/{CONFIG['epochs']} "
              f"train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

    run.finish()


def main():
    print("Loading data...")
    x_train_full, y_train_full, x_test, y_test = load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    for opt in OPTIMIZERS:
        print(f"\n{'='*50}\nTraining with optimizer: {opt}\n{'='*50}")
        train_one_optimizer(opt, x_train, y_train, x_val, y_val)

    print("\nAll optimizer runs complete! Check W&B for convergence plots.")


if __name__ == "__main__":
    main()
