"""
fashion_mnist_transfer.py — Q2.10: Test top 3 MNIST configs on Fashion-MNIST.

Edit the TOP_3_CONFIGS below with your actual best configs from the MNIST sweep.
Run:
    python fashion_mnist_transfer.py
"""

import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from network.mlp import MLP
from network.optimizers import get_optimizer


# -----------------------------------------------------------------------
# EDIT THESE with your best 3 configs from the MNIST sweep results
# -----------------------------------------------------------------------
TOP_3_CONFIGS = [
    {
        "label":         "config-1-adam-relu",
        "hidden_sizes":  [128, 128, 128],
        "activation":    "relu",
        "optimizer":     "adam",
        "learning_rate": 0.001,
        "weight_init":   "xavier",
        "loss":          "cross_entropy",
        "weight_decay":  0.0,
    },
    {
        "label":         "config-2-nadam-tanh",
        "hidden_sizes":  [128, 128],
        "activation":    "tanh",
        "optimizer":     "nadam",
        "learning_rate": 0.001,
        "weight_init":   "xavier",
        "loss":          "cross_entropy",
        "weight_decay":  0.0005,
    },
    {
        "label":         "config-3-rmsprop-relu",
        "hidden_sizes":  [128, 128, 128, 128],
        "activation":    "relu",
        "optimizer":     "rmsprop",
        "learning_rate": 0.001,
        "weight_init":   "xavier",
        "loss":          "cross_entropy",
        "weight_decay":  0.0,
    },
]

BASE_CONFIG = {
    "dataset":       "fashion_mnist",
    "epochs":        20,
    "batch_size":    64,
    "wandb_project": "da6401-assignment1",
}

FASHION_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_data():
    from keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 784).astype(np.float32)  / 255.0
    return x_train, y_train, x_test, y_test


def train_and_evaluate(cfg, x_train, y_train, x_val, y_val, x_test, y_test):
    run = wandb.init(
        project=BASE_CONFIG["wandb_project"],
        name=f"Q2.10-{cfg['label']}",
        group="Q2.10-fashion-mnist-transfer",
        config={**BASE_CONFIG, **cfg},
        reinit=True,
    )

    model = MLP(
        input_size=784,
        hidden_sizes=cfg["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init=cfg["weight_init"],
        loss=cfg["loss"],
    )
    optimizer = get_optimizer(cfg["optimizer"], lr=cfg["learning_rate"],
                              weight_decay=cfg.get("weight_decay", 0.0))

    num_batches = int(np.ceil(x_train.shape[0] / BASE_CONFIG["batch_size"]))
    best_val_acc = 0.0
    best_weights = None

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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = model.get_weights()

        print(f"  [{cfg['label']}] Epoch {epoch}/{BASE_CONFIG['epochs']} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_accuracy":   val_acc,
        })

    # Evaluate best model on test set
    model.set_weights(best_weights)
    y_pred   = model.predict(x_test)
    test_acc = (y_pred == y_test).mean()
    test_f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n  [{cfg['label']}] Test Acc={test_acc:.4f}  F1={test_f1:.4f}")
    wandb.summary["test_accuracy"] = test_acc
    wandb.summary["test_f1"]       = test_f1
    run.finish()

    return test_acc, test_f1


def main():
    print("Loading Fashion-MNIST...")
    x_train_full, y_train_full, x_test, y_test = load_data()
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )

    results = []
    for cfg in TOP_3_CONFIGS:
        print(f"\n{'='*55}\nRunning: {cfg['label']}\n{'='*55}")
        acc, f1 = train_and_evaluate(cfg, x_train, y_train, x_val, y_val, x_test, y_test)
        results.append((cfg["label"], acc, f1))

    print("\n" + "="*55)
    print("FINAL RESULTS — Fashion-MNIST Transfer")
    print("="*55)
    for label, acc, f1 in results:
        print(f"  {label:35s}  Acc={acc:.4f}  F1={f1:.4f}")
    print("="*55)


if __name__ == "__main__":
    main()
