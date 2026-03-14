# 🧠 Neural Network Backpropagation from Scratch

A minimal, dependency-light implementation of a **feedforward neural network** trained entirely with the **backpropagation algorithm** — no deep-learning frameworks, just NumPy and linear algebra.

---

## 📖 Description

This project builds a 2-layer neural network (input → hidden → output) from first principles to demonstrate how backpropagation actually works under the hood. Every gradient is derived by hand using the chain rule and applied via vanilla gradient descent.

The goal is educational: strip away the abstraction layers of PyTorch / TensorFlow and show the raw math that powers modern deep learning.

---

## 🔬 Methodology

| Step | Detail |
|------|--------|
| **Forward pass** | Input is multiplied by weight matrices, biases are added, and ReLU activations are applied at each layer. |
| **Loss** | Mean Squared Error (MSE) — `L = 0.5 * ‖ŷ − y‖²` for a single sample. |
| **Backward pass** | Gradients are computed layer-by-layer using the **chain rule**: `∂L/∂W = ∂L/∂a · ∂a/∂h · ∂h/∂W`, propagating error from output back to input. |
| **Weight update** | Standard gradient descent: `W ← W − α · ∂L/∂W` where α is the learning rate. |

### Network Architecture

```
Input (n)  ──▶  Hidden (h, ReLU)  ──▶  Output (m, ReLU)
         W1, b                  W2
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Core language |
| 🔢 NumPy | Matrix operations & vectorized math |

---

## 📦 Dependencies

```
numpy
```

Install with:

```bash
pip install numpy
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/stabgan/neural-network-backpropagation-algorithm-from-scratch.git
cd neural-network-backpropagation-algorithm-from-scratch

# (Optional) Create a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install numpy

# Run the demo
python backprop.py
```

**Expected output** — the network learns the mapping `y = 2x`:

```
Predictions after training:
  x=0.5  target=1.0  predicted=0.9987
  x=1.0  target=2.0  predicted=1.9991
  x=1.5  target=3.0  predicted=2.9995
  x=2.0  target=4.0  predicted=3.9999
```

### Using the API in your own code

```python
import numpy as np
from backprop import NeuralNet

w1 = np.random.randn(8, 3) * 0.5   # 8 hidden units, 3 inputs
w2 = np.random.randn(1, 8) * 0.5   # 1 output
b  = np.zeros((8, 1))

nn = NeuralNet(w1, w2, b, lr=0.01, epochs=100)
nn.train_neural_network(your_dataset)   # list of (x, y) column-vector pairs
prediction = nn.predict(x_new)
```

---

## ⚠️ Known Issues

- **ReLU-only activations** — using ReLU on the output layer limits the network to non-negative predictions. Swap to a linear output or sigmoid for other tasks.
- **Single-sample gradient descent** — no mini-batch support; training can be slow and noisy on larger datasets.
- **No regularization** — the model may overfit on small datasets.
- **Dying ReLU** — neurons can "die" (output zero for all inputs) if learning rate is too high; consider Leaky ReLU for more robust training.

---

## 📄 License

This project is open source. See the repository for license details.
