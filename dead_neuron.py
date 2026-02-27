"""
dead_neuron.py — Q2.5: Dead neuron investigation (ReLU high LR vs Tanh).

Run:
    python dead_neuron.py
"""

import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from network.mlp import MLP
from network.optimizers import get_optimizer


CONFIGS = [
    {"activation": "relu", "learning_rate": 0.1,   "label": "relu-high-lr"},
    {"activation": "relu", "learning_rate": 0.001, "label": "relu-normal-lr"},
    {"activation": "tanh", "learning_rate": 0.1,   "label": "tanh-high-lr"},
    {"activation": "tanh", "learning_rate": 0.001, "label": "tanh-normal-lr"},
]

BASE_CONFIG = {
    "dataset":      "fashion_mnist",
    "epochs":       15,
    "batch_size":   64,
    "loss":         "cross_entropy",
    "optimizer":    "sgd",
    "weight_decay": 0.0,
    "hidden_sizes": [128, 128, 128],
    "weight_init":  "xavier",
    "wandb_project":"da6401-assignment1",
}


def load_data():
    from keras.datasets import fashion_mnist
    (x_train, y_train), _ = fashion_mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    return x_train, y_train


def count_dead_neurons(model, x_sample):
    """Count neurons that output 0 for ALL inputs in x_sample (only for ReLU layers)."""
    dead_counts = []
    out = x_sample
    for i, layer in enumerate(model.layers[:-1]):  # skip output layer
        out = layer.forward(out)
        # Dead if activation is 0 for every sample
        dead = np.all(out <= 0, axis=0).sum()
        dead_counts.append(dead)
    return dead_counts


def get_activation_stats(model, x_sample):
    """Get mean activation per hidden layer."""
    stats = {}
    out = x_sample
    for i, layer in enumerate(model.layers[:-1]):
        out = layer.forward(out)
        stats[f"mean_activation_layer_{i+1}"] = float(np.mean(out))
        stats[f"std_activation_layer_{i+1}"]  = float(np.std(out))
        stats[f"frac_zero_layer_{i+1}"]        = float(np.mean(out <= 0))
    return stats


def train_and_log(cfg, x_train, y_train, x_val, y_val):
    run = wandb.init(
        project=BASE_CONFIG["wandb_project"],
        name=f"Q2.5-{cfg['label']}",
        group="Q2.5-dead-neuron",
        config={**BASE_CONFIG, **cfg},
        reinit=True,
    )

    model = MLP(
        input_size=784,
        hidden_sizes=BASE_CONFIG["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init=BASE_CONFIG["weight_init"],
        loss=BASE_CONFIG["loss"],
    )
    optimizer = get_optimizer(BASE_CONFIG["optimizer"], lr=cfg["learning_rate"])
    num_batches = int(np.ceil(x_train.shape[0] / BASE_CONFIG["batch_size"]))

    # Use a fixed sample for monitoring activations
    monitor_sample = x_val[:256]

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

        # Activation statistics
        act_stats = get_activation_stats(model, monitor_sample)

        # Dead neuron count (meaningful only for ReLU)
        dead_counts = count_dead_neurons(model, monitor_sample)
        dead_stats  = {f"dead_neurons_layer_{i+1}": d for i, d in enumerate(dead_counts)}

        print(f"  [{cfg['label']}] Epoch {epoch}/{BASE_CONFIG['epochs']} "
              f"val_acc={val_acc:.4f} dead_neurons={dead_counts}")

        wandb.log({
            "epoch":          epoch,
            "train_loss":     train_loss,
            "train_accuracy": train_acc,
            "val_accuracy":   val_acc,
            **act_stats,
            **dead_stats,
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
        train_and_log(cfg, x_train, y_train, x_val, y_val)

    print("\nDone! Check W&B for dead neuron and activation plots.")


if __name__ == "__main__":
    main()
