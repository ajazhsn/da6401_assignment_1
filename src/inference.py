"""
inference.py — Load a saved model and evaluate on test data.

Outputs: Accuracy, Precision, Recall, F1-score and Confusion Matrix.

Example:
    python inference.py --model models/best_model.npy \
                        --config models/best_config.json \
                        --dataset fashion_mnist
"""

import sys
import os
import argparse
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    classification_report
)

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inference — DA6401 Assignment 1"
    )
    parser.add_argument("--model",   type=str, default="models/best_model.npy",
                        help="Path to saved .npy model weights.")
    parser.add_argument("--config",  type=str, default="models/best_config.json",
                        help="Path to best_config.json.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset from config.")
    return parser.parse_args()


parse_args = parse_arguments

def main():
    args = parse_arguments()

    # Load model configuration
    with open(args.config, "r") as f:
        cfg = json.load(f)

    dataset = args.dataset or cfg["dataset"]

    # Load test data only
    _, _, x_test, _, _, y_test = load_data(dataset)

    # Reconstruct model from config
    model = NeuralNetwork(
        input_size=x_test.shape[1],
        hidden_sizes=cfg["hidden_sizes"],
        output_size=10,
        activation=cfg["activation"],
        weight_init=cfg.get("weight_init", "xavier"),
        loss=cfg.get("loss", "cross_entropy"),
    )
    model.load(args.model)

    # Run inference
    y_pred = model.predict(x_test)

    # Compute metrics
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
    print("\nPer-class Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(cm)
    print("=========================================\n")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    main()
