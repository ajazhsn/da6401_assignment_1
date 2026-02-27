"""
train.py — Training script for DA6401 Assignment 1 MLP.

Example usage:
    python train.py -d fashion_mnist -e 20 -b 64 -l cross_entropy \
        -o adam -lr 1e-3 -wd 0 -nhl 3 -sz 128 128 128 -a relu -w_i xavier
"""

import argparse
import json
import numpy as np
import wandb
from sklearn.model_selection import train_test_split

from network.mlp import MLP
from network.optimizers import get_optimizer


def load_data(dataset_name: str):
    """Load MNIST or Fashion-MNIST via Keras."""
    if dataset_name == "mnist":
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32) / 255.0
    x_test  = x_test.reshape(x_test.shape[0], -1).astype(np.float32)  / 255.0
    return x_train, y_train, x_test, y_test


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP — DA6401 Assignment 1")
    parser.add_argument("-d",   "--dataset",       type=str,   default="fashion_mnist",
                        choices=["mnist", "fashion_mnist", "fashion-mnist"])
    parser.add_argument("-e",   "--epochs",        type=int,   default=10)
    parser.add_argument("-b",   "--batch_size",    type=int,   default=64)
    parser.add_argument("-l",   "--loss",          type=str,   default="cross_entropy",
                        choices=["cross_entropy", "mean_squared_error", "mse"])
    parser.add_argument("-o",   "--optimizer",     type=str,   default="adam",
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr",  "--learning_rate", type=float, default=1e-3)
    parser.add_argument("-wd",  "--weight_decay",  type=float, default=0.0)
    parser.add_argument("-nhl", "--num_layers",    type=int,   default=3)
    parser.add_argument("-sz",  "--hidden_size",   type=int,   nargs="+", default=[128])
    parser.add_argument("-a",   "--activation",    type=str,   default="relu",
                        choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("-w_i", "--weight_init",   type=str,   default="xavier",
                        choices=["random", "xavier"])
    parser.add_argument("--val_split",     type=float, default=0.1)
    parser.add_argument("--no_wandb",      action="store_true")
    parser.add_argument("--wandb_project", type=str,   default="da6401-assignment1")
    parser.add_argument("--wandb_entity",  type=str,   default=None)
    parser.add_argument("--save_path",     type=str,   default="best_model.npy")
    parser.add_argument("--config_path",   type=str,   default="best_config.json")
    return parser.parse_args()


def train(args):
    if len(args.hidden_size) == 1:
        hidden_sizes = args.hidden_size * args.num_layers
    elif len(args.hidden_size) == args.num_layers:
        hidden_sizes = args.hidden_size
    else:
        raise ValueError(
            f"--hidden_size must have 1 value or exactly --num_layers={args.num_layers} values."
        )

    config = vars(args)
    config["hidden_sizes"] = hidden_sizes

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
        )
        config = dict(wandb.config)

    print(f"Loading dataset: {args.dataset}")
    x_train_full, y_train_full, x_test, y_test = load_data(args.dataset)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=args.val_split, random_state=42, stratify=y_train_full
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    model = MLP(
        input_size=x_train.shape[1],
        hidden_sizes=hidden_sizes,
        output_size=10,
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss,
    )
    optimizer = get_optimizer(
        args.optimizer, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    num_batches  = int(np.ceil(x_train.shape[0] / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        idx = np.random.permutation(x_train.shape[0])
        x_train, y_train = x_train[idx], y_train[idx]

        epoch_loss, correct = 0.0, 0
        for b in range(num_batches):
            xb = x_train[b * args.batch_size: (b + 1) * args.batch_size]
            yb = y_train[b * args.batch_size: (b + 1) * args.batch_size]
            logits     = model.forward(xb)
            loss       = model.compute_loss(logits, yb)
            model.backward()
            optimizer.step(model.layers)
            epoch_loss += loss * xb.shape[0]
            correct    += (np.argmax(logits, axis=1) == yb).sum()

        train_loss = epoch_loss / x_train.shape[0]
        train_acc  = correct   / x_train.shape[0]
        val_logits = model.forward(x_val)
        val_loss   = model.compute_loss(val_logits, y_val)
        val_acc    = (np.argmax(val_logits, axis=1) == y_val).mean()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch":          epoch,
                "train_loss":     train_loss,
                "train_accuracy": train_acc,
                "val_loss":       val_loss,
                "val_accuracy":   val_acc,
            })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
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

    model.load(args.save_path)
    test_logits = model.forward(x_test)
    test_acc    = (np.argmax(test_logits, axis=1) == y_test).mean()
    print(f"\nBest val acc : {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    if use_wandb:
        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary["test_accuracy"]     = test_acc
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
