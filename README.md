# DA6401 Assignment 1 — Multi-Layer Perceptron

A fully NumPy-based, configurable Multi-Layer Perceptron (MLP) for image classification on MNIST and Fashion-MNIST, built as part of the DA6401 Introduction to Deep Learning course (IIT Madras, 2026).

---

## Project Structure

```
da6401_assignment_1/
├── models/
│   ├── best_model.npy         # Best model weights (serialised NumPy)
│   └── best_config.json       # Best model hyperparameter configuration
├── notebooks/
│   └── wandb_demo.ipynb       # Demo notebook for training + W&B logging
├── src/
│   ├── ann/
│   │   ├── __init__.py                # Package exports
│   │   ├── activations.py             # Sigmoid, Tanh, ReLU, Identity
│   │   ├── neural_layer.py            # NeuralLayer (forward + backward, grad_W / grad_b)
│   │   ├── neural_network.py          # NeuralNetwork MLP class (build, forward, backward, save/load)
│   │   ├── objective_functions.py     # CrossEntropyLoss, MSELoss
│   │   └── optimizers.py              # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py                # Package exports
│   │   └── data_loader.py             # MNIST / Fashion-MNIST loader + train/val split
│   ├── train.py                       # CLI training script
│   └── inference.py                   # CLI evaluation script
├── sweep.yaml                 # W&B Bayesian hyperparameter sweep config
├── requirements.txt           # Python dependencies
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch, TensorFlow, and JAX are **not** required or permitted for training. All mathematical operations use NumPy only.

---

## Training

Run from the project root:

```bash
python src/train.py \
  -d   fashion_mnist \   # dataset: mnist | fashion_mnist
  -e   20 \              # epochs
  -b   64 \              # batch size
  -l   cross_entropy \   # loss: cross_entropy | mean_squared_error
  -o   adam \            # optimizer: sgd | momentum | nag | rmsprop | adam | nadam
  -lr  0.001 \           # learning rate
  -wd  0.0 \             # weight decay (L2 regularization)
  -nhl 3 \               # number of hidden layers
  -sz  128 128 128 \     # neurons per hidden layer
  -a   relu \            # activation: sigmoid | tanh | relu
  -w_i xavier            # weight init: random | xavier
```

Metrics are automatically logged to **Weights & Biases**. Add `--no_wandb` to disable logging.

Best model weights are saved to `models/best_model.npy` and config to `models/best_config.json`.

---

## Inference

```bash
python src/inference.py \
  --model   models/best_model.npy \
  --config  models/best_config.json \
  --dataset fashion_mnist
```

Outputs: **Accuracy, Precision, Recall, F1-score** and a full **Confusion Matrix**.

---

## Hyperparameter Sweep (W&B)

```bash
# Initialise sweep (run from project root)
wandb sweep sweep.yaml

# Launch agent (replace <SWEEP_ID> with ID printed above)
wandb agent <ENTITY>/<PROJECT>/<SWEEP_ID> --count 100
```

The sweep uses **Bayesian optimization** over: optimizer, learning rate, activation, batch size, hidden size, number of layers, weight decay, weight initialization, and loss function.

---

## Implementation Notes

| Component | Details |
|-----------|---------|
| **Forward pass** | `NeuralLayer.forward()` caches input `x` and pre-activation `z` for backprop |
| **Backward pass** | `NeuralLayer.backward()` computes and stores `self.grad_W` and `self.grad_b` |
| **Loss** | `CrossEntropyLoss` / `MSELoss` — both accept raw logits; softmax applied internally |
| **Optimizers** | All 6 optimizers maintain per-layer `optimizer_state` dicts for moment tracking |
| **Serialisation** | `model.save(path)` stores `[(W, b), ...]` via `np.save(..., allow_pickle=True)` |
| **Data loading** | `data_loader.py` handles download, flatten, normalise, and stratified train/val split |

---

## Results

| Dataset | Val Accuracy | Test Accuracy |
|---------|-------------|---------------|
| Fashion-MNIST | 90.57% | 89.27% |

Best config: **Adam optimizer, ReLU activation, 2 hidden layers × 128 neurons, lr=0.001, Xavier init, Cross-Entropy loss**

---

## W&B Report

[Link to public W&B report](https://wandb.ai/da25m006-iitmaana/da6401-assignment1/reports/Introduction-to-Deep-Learning---VmlldzoxNjA1MTA2Mg?accessToken=yn2exrc0imj5z5xuw33kuephb8aiyn5m0fwk9gvagbbas30rgbkpwqjk7vla3io9)

---

## Academic Integrity

This implementation was written individually for DA6401 (IIT Madras, 2026). AI tools were used only as conceptual aids; all mathematical implementations are original NumPy code.