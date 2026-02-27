"""
inference.py — Load a saved model and evaluate on test data.

Example usage:
    python inference.py --model best_model.npy --config best_config.json --dataset fashion_mnist
"""

import argparse
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

from network.mlp import MLP


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Inference — DA6401 Assignment 1")
    parser.add_argument("--model",   type=str, default="best_model.npy",
                        help="Path to saved .npy weights.")
    parser.add_argument("--config",  type=str, default="best_config.json",
                        help="Path to best_config.json.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset from config ('mnist' or 'fashion_mnist').")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (reuse train.py helper)
# ---------------------------------------------------------------------------
def load_test_data(dataset_name: str):
    if dataset_name == "mnist":
        from keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
    elif dataset_name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist
        (_, _), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")

    x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32) / 255.0
    return x_test, y_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = json.load(f)

    dataset = args.dataset or cfg["dataset"]
    print(f"Loading test data for: {dataset}")
    x_test, y_test = load_test_data(dataset)

    # Build model from config
    model = MLP(
        input_size=x_test.shape[1],
        hidden_sizes=cfg["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init=cfg.get("weight_init", "xavier"),
        loss=cfg.get("loss", "cross_entropy"),
    )
    model.load(args.model)

    # Predict
    y_pred = model.predict(x_test)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n========== Evaluation Results ==========")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=========================================\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    main()
