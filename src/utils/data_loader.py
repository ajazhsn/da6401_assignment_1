# src/utils/data_loader.py
# Utility for loading and preprocessing MNIST / Fashion-MNIST datasets.
# Uses keras.datasets for downloading — no PyTorch/TensorFlow ops used.

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(dataset_name: str, val_split: float = 0.1):
    """
    Load, flatten, and normalise MNIST or Fashion-MNIST.
    Splits training data into train/val sets.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        val_split:    Fraction of training data to use for validation (default 0.1)

    Returns:
        x_train, x_val, x_test: float32 arrays of shape (N, 784), normalised to [0,1]
        y_train, y_val, y_test: int arrays of shape (N,) with class labels 0-9
    """
    if dataset_name == "mnist":
        from keras.datasets import mnist
        (x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

    elif dataset_name in ("fashion_mnist", "fashion-mnist"):
        from keras.datasets import fashion_mnist
        (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choose 'mnist' or 'fashion_mnist'."
        )

    # Flatten 28x28 images to 784-dimensional vectors and normalise to [0, 1]
    x_train_full = x_train_full.reshape(-1, 784).astype(np.float32) / 255.0
    x_test       = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    # Stratified train/val split to preserve class balance
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=val_split,
        random_state=42,
        stratify=y_train_full,
    )

    print(
        f"Dataset: {dataset_name} | "
        f"Train: {x_train.shape[0]} | "
        f"Val: {x_val.shape[0]} | "
        f"Test: {x_test.shape[0]}"
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
