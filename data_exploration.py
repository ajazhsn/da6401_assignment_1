"""
data_exploration.py — Q2.1: Log sample images from each class to W&B.

Run:
    python data_exploration.py --dataset fashion_mnist
"""

import argparse
import numpy as np
import wandb

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
MNIST_CLASSES = [str(i) for i in range(10)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fashion_mnist",
                        choices=["mnist", "fashion_mnist"])
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment1")
    parser.add_argument("--wandb_entity", type=str, default=None)
    return parser.parse_args()


def load_data(dataset_name):
    if dataset_name == "mnist":
        from keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()
        class_names = MNIST_CLASSES
    else:
        from keras.datasets import fashion_mnist
        (x_train, y_train), _ = fashion_mnist.load_data()
        class_names = FASHION_MNIST_CLASSES
    return x_train, y_train, class_names


def main():
    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="Q2.1-data-exploration",
        job_type="exploration",
    )

    x_train, y_train, class_names = load_data(args.dataset)
    print(f"Dataset: {args.dataset}, Shape: {x_train.shape}")

    # ---- Log 5 samples per class as a W&B Table ----
    columns = ["image", "label", "class_name"]
    table = wandb.Table(columns=columns)

    for class_idx, class_name in enumerate(class_names):
        # Get indices for this class
        indices = np.where(y_train == class_idx)[0]
        # Pick 5 random samples
        chosen = np.random.choice(indices, size=5, replace=False)
        for idx in chosen:
            img = x_train[idx]  # (28, 28)
            table.add_data(
                wandb.Image(img, caption=class_name),
                class_idx,
                class_name,
            )
        print(f"  Class {class_idx} ({class_name}): logged 5 samples")

    wandb.log({"sample_images": table})

    # ---- Log class distribution bar chart ----
    unique, counts = np.unique(y_train, return_counts=True)
    dist_table = wandb.Table(
        columns=["class_name", "count"],
        data=[[class_names[i], int(counts[i])] for i in unique]
    )
    wandb.log({"class_distribution": wandb.plot.bar(
        dist_table, "class_name", "count", title="Class Distribution"
    )})

    print("\nDone! Check your W&B dashboard for the image table.")
    wandb.finish()


if __name__ == "__main__":
    main()
