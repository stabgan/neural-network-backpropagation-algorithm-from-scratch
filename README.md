# Neural Network — Backpropagation from Scratch

A minimal two-layer neural network built entirely with NumPy. No frameworks, no magic — just matrix math and the chain rule.

## What It Does

Implements a fully-connected feedforward neural network with one hidden layer, trained via stochastic gradient descent and manual backpropagation.

**Architecture:**

```
input → Dense(w1, b1) → ReLU → Dense(w2, b2) → linear output
```

The included demo learns the function **y = 2·x₁ + 3·x₂** from 200 random samples and converges to near-zero loss in ~100 epochs.

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3.8+ | Language |
| 🔢 NumPy | Matrix operations, vectorized math |

## Getting Started

```bash
pip install -r requirements.txt
python backprop.py
```

**Expected output:**

```
Training …
  Epoch  10/100  loss = 0.000889
  ...
  Epoch 100/100  loss = 0.000076

Predictions vs targets (first 5 samples):
  x=[0.37 0.95]  target=3.6012  pred=3.5920
  ...
```

## How It Works

1. **Forward pass** — input flows through the hidden layer (ReLU activation) and output layer (linear activation)
2. **Loss** — mean squared error: L = ½(ŷ − y)²
3. **Backward pass** — gradients are computed layer-by-layer using the chain rule
4. **Update** — weights and biases are adjusted via vanilla gradient descent

## API

```python
from backprop import NeuralNet
import numpy as np

w1 = np.random.randn(8, 2) * np.sqrt(2.0 / 2)
w2 = np.random.randn(1, 8) * np.sqrt(2.0 / 8)
b1 = np.zeros((8, 1))
b2 = np.zeros((1, 1))

nn = NeuralNet(w1, w2, b1, b2, lr=0.005, epochs=100)

# training_data = [(x_col_vec, y_col_vec), ...]
nn.train_neural_network(training_data)

prediction = nn.predict(x_col_vec)
```

## Bugs Fixed (from original)

| Bug | Fix |
|-----|-----|
| Missing `import numpy as np` | Added at top of file |
| `relu_der` used scalar `if/else` — fails on arrays | Replaced with vectorized `(x > 0).astype(float)` |
| `lr` referenced instead of `self.lr` in `train()` | Fixed to `self.lr` |
| `train_neural_network` referenced undefined `train` variable | Now accepts `training_data` parameter |
| ReLU on output layer caused dead neurons (zero gradients) | Changed to linear output activation |
| Single bias for both layers | Added separate `b1`, `b2` biases |
| Inconsistent shapes between `train()` and `predict()` | Unified to column-vector convention |
| No demo or main guard | Added `if __name__ == "__main__"` with working example |

## License

[MIT](LICENSE)
