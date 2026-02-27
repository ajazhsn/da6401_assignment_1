"""
confusion_matrix.py — Q2.8: Plot confusion matrix for best model.

Run:
    python confusion_matrix.py --model best_model.npy --config best_config.json
"""

import argparse
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix, classification_report

from network.mlp import MLP


FASHION_MNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
MNIST_CLASSES = [str(i) for i in range(10)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   type=str, default="best_model.npy")
    parser.add_argument("--config",  type=str, default="best_config.json")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment1")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    return parser.parse_args()


def load_test_data(dataset_name):
    if dataset_name == "mnist":
        from keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
        class_names = MNIST_CLASSES
    else:
        from keras.datasets import fashion_mnist
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
        class_names = FASHION_MNIST_CLASSES
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    return x_test, y_test, class_names


def plot_confusion_matrix(cm, class_names):
    """Standard confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14, fontweight="bold")

    thresh = cm_norm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=8)
    plt.tight_layout()
    return fig


def plot_top_confusions(cm, class_names, top_n=10):
    """Creative visualization: show top misclassified pairs as a bar chart."""
    errors = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], class_names[i], class_names[j]))

    errors.sort(reverse=True)
    top_errors = errors[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{true} → {pred}" for _, true, pred in top_errors]
    counts = [count for count, _, _ in top_errors]

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(counts)))
    bars = ax.barh(labels, counts, color=colors)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)

    ax.set_xlabel("Number of Misclassifications", fontsize=12)
    ax.set_title(f"Top {top_n} Misclassification Pairs\n(True Class → Predicted Class)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    dataset = args.dataset or cfg["dataset"]
    x_test, y_test, class_names = load_test_data(dataset)

    model = MLP(
        input_size=784,
        hidden_sizes=cfg["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init=cfg.get("weight_init", "xavier"),
        loss=cfg.get("loss", "cross_entropy"),
    )
    model.load(args.model)
    y_pred = model.predict(x_test)

    # Metrics
    cm  = confusion_matrix(y_test, y_pred)
    acc = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="Q2.8-confusion-matrix",
        job_type="evaluation",
    )
    wandb.summary["test_accuracy"] = acc

    # Log W&B native confusion matrix
    wandb.log({"confusion_matrix_wandb": wandb.plot.confusion_matrix(
        y_true=y_test.tolist(),
        preds=y_pred.tolist(),
        class_names=class_names,
    )})

    # Log custom matplotlib confusion matrix
    fig1 = plot_confusion_matrix(cm, class_names)
    wandb.log({"confusion_matrix_heatmap": wandb.Image(fig1)})
    fig1.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

    # Log top confusions creative plot
    fig2 = plot_top_confusions(cm, class_names)
    wandb.log({"top_misclassifications": wandb.Image(fig2)})
    fig2.savefig("top_misclassifications.png", dpi=150, bbox_inches="tight")

    plt.close("all")
    print("\nPlots saved and logged to W&B!")
    wandb.finish()


if __name__ == "__main__":
    main()
