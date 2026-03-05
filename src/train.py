"""
train.py — CLI training script for DA6401 Assignment 1.

Trains a NumPy MLP on MNIST or Fashion-MNIST with full W&B logging.

Example:
    python train.py -d fashion_mnist -e 20 -b 64 -l cross_entropy \
        -o adam -lr 0.001 -wd 0.0 -nhl 3 -sz 128 -a relu -w_i xavier
"""

import sys
import os
import argparse
import json
import numpy as np
import wandb

# Make src importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_data


# ---------------------------------------------------------------------------
# Argument Parser — flags match assignment spec exactly
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NumPy MLP — DA6401 Assignment 1"
    )

    # Required assignment arguments
    parser.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                        choices=["mnist", "fashion_mnist", "fashion-mnist"],
                        help="Dataset to train on.")
    parser.add_argument("-e",   "--epochs",        type=int,   default=10,
                        help="Number of training epochs.")
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64,
                        help="Mini-batch size.")
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error", "mse"],
                        help="Loss function.")
    parser.add_argument("-o",   "--optimizer",     type=str,   default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Optimizer.")
    parser.add_argument("-lr",  "--learning_rate", type=float, default=1e-3,
                        help="Initial learning rate.")
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0,
                        help="L2 regularization coefficient.")
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3,
                        help="Number of hidden layers.")
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128],
                        help="Neurons per hidden layer. One value = same for all layers.")
    parser.add_argument("-a",   "--activation",    type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"],
                        help="Hidden layer activation function.")
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier"],
                        help="Weight initialization strategy.")

    # Optional arguments
    parser.add_argument("--val_split",     type=float, default=0.1,
                        help="Validation split fraction.")
    parser.add_argument("--no_wandb",      action="store_true",
                        help="Disable W&B logging.")
    parser.add_argument("--wandb_project", type=str,   default="da6401-assignment1",
                        help="W&B project name.")
    parser.add_argument("--wandb_entity",  type=str,   default=None,
                        help="W&B entity/username.")
    parser.add_argument("--save_path",     type=str,   default="models/best_model.npy",
                        help="Path to save best model weights.")
    parser.add_argument("--config_path",   type=str,   default="models/best_config.json",
                        help="Path to save best model config.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train(args):
    # Build hidden_sizes list from -sz and -nhl arguments
    if len(args.hidden_size) == 1:
        # Broadcast single value to all layers
        hidden_sizes = args.hidden_size * args.num_layers
    elif len(args.hidden_size) == args.num_layers:
        hidden_sizes = args.hidden_size
    else:
        raise ValueError(
            f"-sz must have 1 value (broadcast) or exactly -nhl={args.num_layers} values."
        )

    config = vars(args)
    config["hidden_sizes"] = hidden_sizes

    # Initialise W&B run
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
        )
        # W&B sweep may override config values
        config = dict(wandb.config)

    # Load data
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(
        args.dataset, val_split=args.val_split
    )

    # Build model
    model = NeuralNetwork(
        input_size=x_train.shape[1],      # 784
        hidden_sizes=hidden_sizes,
        output_size=10,                   # 10 classes
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss,
    )

    # Build optimizer
    optimizer = get_optimizer(
        name=args.optimizer,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_val_acc = 0.0
    num_batches  = int(np.ceil(x_train.shape[0] / args.batch_size))

    for epoch in range(1, args.epochs + 1):

        # Shuffle training data each epoch
        idx = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[idx], y_train[idx]

        epoch_loss = 0.0
        correct    = 0

        # Mini-batch training
        for b in range(num_batches):
            xb = x_train[b * args.batch_size: (b + 1) * args.batch_size]
            yb = y_train[b * args.batch_size: (b + 1) * args.batch_size]

            logits     = model.forward(xb)           # forward pass
            loss       = model.compute_loss(logits, yb)
            model.backward()                         # backward pass
            optimizer.step(model.layers)             # weight update

            epoch_loss += loss * xb.shape[0]
            correct    += (np.argmax(logits, axis=1) == yb).sum()

        # Compute epoch-level metrics
        train_loss = epoch_loss / x_train.shape[0]
        train_acc  = correct   / x_train.shape[0]

        val_logits = model.forward(x_val)
        val_loss   = model.compute_loss(val_logits, y_val)
        val_acc    = (np.argmax(val_logits, axis=1) == y_val).mean()

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch":          epoch,
                "train_loss":     train_loss,
                "train_accuracy": train_acc,
                "val_loss":       val_loss,
                "val_accuracy":   val_acc,
            })

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            model.save(args.save_path)
            with open(args.config_path, "w") as f:
                json.dump({
                    "dataset":       args.dataset,
                    "hidden_sizes":  hidden_sizes,
                    "activation":    args.activation,
                    "weight_init":   args.weight_init,
                    "loss":          args.loss,
                    "optimizer":     args.optimizer,
                    "learning_rate": args.learning_rate,
                    "weight_decay":  args.weight_decay,
                    "best_val_acc":  float(best_val_acc),
                }, f, indent=2)

    # Evaluate best model on test set
    model.load(args.save_path)
    test_logits = model.forward(x_test)
    test_acc    = (np.argmax(test_logits, axis=1) == y_test).mean()

    print(f"\nBest val acc : {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    if use_wandb:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["test_accuracy"]     = test_acc
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    train(args)
