"""
Run from project root: python retrain_and_save.py
"""
import numpy as np, json, os, sys
sys.path.insert(0, 'src')
for k in list(sys.modules.keys()):
    if 'ann' in k or 'utils' in k: del sys.modules[k]

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data

print("Loading Fashion-MNIST...")
result = load_data('fashion_mnist')
print(f"load_data returned {len(result)} items")

# load_data returns: x_train, x_val, x_test, y_train, y_val, y_test
# (all y are integer labels OR one-hot — let's detect)
x_train, x_val, x_test = result[0], result[1], result[2]
y_train, y_val, y_test  = result[3], result[4], result[5]

# Convert to integer labels if one-hot
def to_int(y):
    if y.ndim == 2: return np.argmax(y, axis=1)
    return y.astype(int)

y_train = to_int(y_train)
y_val   = to_int(y_val)
y_test  = to_int(y_test)

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_val:   {x_val.shape},   y_val:   {y_val.shape}")
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")

cfg = {
    "dataset": "fashion_mnist",
    "hidden_sizes": [128, 128],
    "activation": "relu",
    "weight_init": "xavier",
    "loss": "cross_entropy",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
}

model = NeuralNetwork(784, cfg['hidden_sizes'], 10,
                      activation=cfg['activation'],
                      weight_init=cfg['weight_init'],
                      loss=cfg['loss'])

class Adam:
    def __init__(self, lr=0.001):
        self.lr=lr; self.b1=0.9; self.b2=0.999; self.eps=1e-8
        self.m={}; self.v={}; self.t=0
    def update(self, layer):
        self.t += 1
        lid = id(layer)
        if lid not in self.m:
            self.m[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
            self.v[lid]={'W':np.zeros_like(layer.W),'b':np.zeros_like(layer.b)}
        for p,g in [('W',layer.grad_W),('b',layer.grad_b)]:
            self.m[lid][p]=self.b1*self.m[lid][p]+(1-self.b1)*g
            self.v[lid][p]=self.b2*self.v[lid][p]+(1-self.b2)*g**2
            mh=self.m[lid][p]/(1-self.b1**self.t)
            vh=self.v[lid][p]/(1-self.b2**self.t)
            if p=='W': layer.W -= self.lr*mh/(np.sqrt(vh)+self.eps)
            else:      layer.b -= self.lr*mh/(np.sqrt(vh)+self.eps)

opt = Adam(lr=0.001)
best_val_acc = 0.0
best_weights = None
EPOCHS = 15; BS = 32

print(f"Training {EPOCHS} epochs with Adam lr=0.001...")
for epoch in range(1, EPOCHS+1):
    idx = np.random.permutation(len(x_train))
    xtr, ytr = x_train[idx], y_train[idx]
    total_loss = 0; nb = 0
    for i in range(0, len(xtr), BS):
        xb, yb = xtr[i:i+BS], ytr[i:i+BS]
        logits = model.forward(xb)
        total_loss += model.compute_loss(logits, yb)
        model.backward()
        for layer in model.layers: opt.update(layer)
        nb += 1
    val_preds = model.predict(x_val)
    val_acc = np.mean(val_preds == y_val)
    print(f"Epoch {epoch:2d}: loss={total_loss/nb:.4f}  val_acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = model.get_weights()

model.set_weights(best_weights)
from sklearn.metrics import f1_score
f1 = f1_score(y_test, model.predict(x_test), average='weighted')
print(f"\nBest val_acc={best_val_acc:.4f}  Test F1={f1:.4f}")

cfg['best_val_acc'] = float(best_val_acc)
for path in ['models/best_model.npy','src/best_model.npy']:
    model.save(path)
for path in ['models/best_config.json','src/best_config.json']:
    with open(path,'w') as f: json.dump(cfg,f,indent=2)
    print(f"Saved {path}")

print(f"\nDone! F1={f1:.4f}")
print("git add -A")
print('git commit -m "Retrained model F1={:.4f}"'.format(f1))
print("git push origin main --force")
