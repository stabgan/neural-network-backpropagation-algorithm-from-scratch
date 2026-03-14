"""
Neural Network with Backpropagation from Scratch
=================================================
A minimal two-layer neural network implemented using only NumPy.
Demonstrates forward propagation, backpropagation, and gradient descent.

Architecture
------------
input → Dense(w1, b1) → ReLU → Dense(w2, b2) → output  (linear output)
"""

import numpy as np


def relu(x):
    """ReLU activation function (vectorized)."""
    return np.maximum(x, 0)


def relu_der(x):
    """Derivative of ReLU activation (vectorized)."""
    return (x > 0).astype(float)


class NeuralNet:
    """A simple two-layer neural network for regression.

    Architecture: input → Dense(w1, b1) → ReLU → Dense(w2, b2) → linear output

    Parameters
    ----------
    w1 : np.ndarray, shape (hidden_size, input_size)
    w2 : np.ndarray, shape (output_size, hidden_size)
    b1 : np.ndarray, shape (hidden_size, 1)
    b2 : np.ndarray, shape (output_size, 1)
    lr : float, optional
        Learning rate (default 0.01).
    epochs : int, optional
        Number of training epochs (default 40).
    """

    def __init__(self, w1, w2, b1, b2, lr=0.01, epochs=40):
        self.w1 = w1.copy()
        self.w2 = w2.copy()
        self.b1 = b1.copy()
        self.b2 = b2.copy()
        self.lr = lr
        self.epochs_to_train = epochs

    def _forward(self, x):
        """Forward pass for a single column-vector sample.

        Parameters
        ----------
        x : np.ndarray, shape (input_size, 1)

        Returns
        -------
        h1 : pre-activation of hidden layer
        a1 : post-activation of hidden layer (ReLU)
        a2 : output (linear, no activation)
        """
        h1 = self.w1 @ x + self.b1       # hidden pre-activation
        a1 = relu(h1)                     # hidden activation
        a2 = self.w2 @ a1 + self.b2       # output (linear)
        return h1, a1, a2

    def train_step(self, x, y):
        """Run one forward + backward pass and update weights.

        Uses MSE loss: L = 0.5 * (a2 - y)^2

        Parameters
        ----------
        x : np.ndarray, shape (input_size, 1)
            Single training sample (column vector).
        y : np.ndarray, shape (output_size, 1)
            Target value (column vector).
        """
        # --- forward pass ---
        h1, a1, a2 = self._forward(x)

        # --- backward pass (chain rule) ---
        # Output layer (linear activation → derivative is 1)
        delta2 = a2 - y                        # dL/dh2 = dL/da2 * 1

        dloss_dw2 = delta2 @ a1.T              # dL/dw2
        dloss_db2 = delta2                     # dL/db2

        # Hidden layer
        dloss_da1 = self.w2.T @ delta2         # dL/da1
        delta1 = dloss_da1 * relu_der(h1)      # dL/dh1

        dloss_dw1 = delta1 @ x.T              # dL/dw1
        dloss_db1 = delta1                     # dL/db1

        # --- gradient descent update ---
        self.w1 -= self.lr * dloss_dw1
        self.w2 -= self.lr * dloss_dw2
        self.b1 -= self.lr * dloss_db1
        self.b2 -= self.lr * dloss_db2

    def predict(self, x):
        """Predict output for a single sample.

        Parameters
        ----------
        x : np.ndarray, shape (input_size, 1)
            Input sample (column vector).

        Returns
        -------
        float
            Scalar prediction.
        """
        _, _, a2 = self._forward(x)
        return a2.item()

    def train_neural_network(self, training_data):
        """Train the network over multiple epochs.

        Parameters
        ----------
        training_data : list of (x, y) tuples
            Each x is shape (input_size, 1), each y is shape (output_size, 1).

        Returns
        -------
        list of float
            Mean loss per epoch.
        """
        history = []
        for epoch in range(self.epochs_to_train):
            epoch_loss = 0.0
            for x, y in training_data:
                _, _, a2 = self._forward(x)
                epoch_loss += 0.5 * float(np.sum((a2 - y) ** 2))
                self.train_step(x, y)
            mean_loss = epoch_loss / len(training_data)
            history.append(mean_loss)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1:>3d}/{self.epochs_to_train}  "
                      f"loss = {mean_loss:.6f}")
        return history


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # --- tiny dataset: y = 2*x1 + 3*x2 ---
    INPUT_SIZE = 2
    HIDDEN_SIZE = 8
    OUTPUT_SIZE = 1
    N_SAMPLES = 200

    X_raw = np.random.rand(N_SAMPLES, INPUT_SIZE)
    Y_raw = 2 * X_raw[:, 0] + 3 * X_raw[:, 1]

    # Convert to list of (column_vector_x, column_vector_y) pairs
    training_data = [
        (X_raw[i].reshape(-1, 1), np.array([[Y_raw[i]]]))
        for i in range(N_SAMPLES)
    ]

    # --- initialise weights (He initialisation for ReLU hidden layer) ---
    w1 = np.random.randn(HIDDEN_SIZE, INPUT_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
    w2 = np.random.randn(OUTPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE)
    b1 = np.zeros((HIDDEN_SIZE, 1))
    b2 = np.zeros((OUTPUT_SIZE, 1))

    nn = NeuralNet(w1, w2, b1, b2, lr=0.005, epochs=100)

    print("Training …")
    nn.train_neural_network(training_data)

    # --- quick sanity check ---
    print("\nPredictions vs targets (first 5 samples):")
    for i in range(5):
        x_col = X_raw[i].reshape(-1, 1)
        pred = nn.predict(x_col)
        print(f"  x={X_raw[i]}  target={Y_raw[i]:.4f}  pred={pred:.4f}")
