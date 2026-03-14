import numpy as np


def relu(x):
    """ReLU activation function."""
    return np.maximum(x, 0)


def relu_der(x):
    """Derivative of ReLU activation function (vectorized)."""
    return (x > 0).astype(float)


class NeuralNet:
    """
    A simple 2-layer neural network trained with backpropagation from scratch.

    Architecture: Input -> Hidden (ReLU) -> Output (ReLU)
    Loss: 0.5 * (prediction - target)^2  (MSE for a single sample)
    """

    def __init__(self, w1, w2, b, lr=0.01, epochs=40):
        """
        Parameters
        ----------
        w1 : np.ndarray, shape (hidden_size, input_size)
        w2 : np.ndarray, shape (output_size, hidden_size)
        b  : np.ndarray, shape (hidden_size, 1)
        lr : float – learning rate
        epochs : int – number of training epochs
        """
        self.w1 = w1.astype(float)
        self.w2 = w2.astype(float)
        self.b = b.astype(float)
        self.lr = lr
        self.epochs_to_train = epochs

    # ------------------------------------------------------------------ #
    #  Forward pass (column-vector convention)                            #
    # ------------------------------------------------------------------ #
    def _forward(self, x):
        """
        Forward pass for a single sample (column vector).

        Returns all intermediate values needed for backprop.
        """
        h1 = np.dot(self.w1, x) + self.b   # (hidden, 1)
        a1 = relu(h1)                       # (hidden, 1)
        h2 = np.dot(self.w2, a1)            # (output, 1)
        a2 = relu(h2)                       # (output, 1)
        return h1, a1, h2, a2

    # ------------------------------------------------------------------ #
    #  Single-sample training step                                        #
    # ------------------------------------------------------------------ #
    def train(self, x, y):
        """
        One gradient-descent step on a single (x, y) pair.

        Parameters
        ----------
        x : np.ndarray, shape (input_size, 1)  – input column vector
        y : np.ndarray, shape (output_size, 1)  – target column vector
        """
        # --- forward ---
        h1, a1, h2, a2 = self._forward(x)

        # --- backward (MSE loss: L = 0.5 * ||a2 - y||^2) ---
        dloss_da2 = a2 - y                          # (output, 1)
        da2_dh2 = relu_der(h2)                       # (output, 1)
        dloss_dh2 = dloss_da2 * da2_dh2              # (output, 1)

        # Backprop through layer 2
        dloss_da1 = np.dot(self.w2.T, dloss_dh2)     # (hidden, 1)
        da1_dh1 = relu_der(h1)                        # (hidden, 1)
        dloss_dh1 = dloss_da1 * da1_dh1               # (hidden, 1)

        # Gradients for weights and bias
        dloss_dw2 = np.dot(dloss_dh2, a1.T)           # (output, hidden)
        dloss_dw1 = np.dot(dloss_dh1, x.T)            # (hidden, input)
        dloss_db = dloss_dh1                           # (hidden, 1)

        # --- update ---
        self.w1 -= self.lr * dloss_dw1
        self.w2 -= self.lr * dloss_dw2
        self.b  -= self.lr * dloss_db

    # ------------------------------------------------------------------ #
    #  Prediction                                                         #
    # ------------------------------------------------------------------ #
    def predict(self, x):
        """
        Predict output for a single input column vector.

        Parameters
        ----------
        x : np.ndarray, shape (input_size, 1)

        Returns
        -------
        float – scalar prediction
        """
        _, _, _, a2 = self._forward(x)
        return a2.item()

    # ------------------------------------------------------------------ #
    #  Full training loop                                                 #
    # ------------------------------------------------------------------ #
    def train_neural_network(self, dataset):
        """
        Train over the full dataset for ``self.epochs_to_train`` epochs.

        Parameters
        ----------
        dataset : iterable of (x, y) pairs
            Each x is shape (input_size, 1), each y is shape (output_size, 1).
        """
        for epoch in range(self.epochs_to_train):
            for x, y in dataset:
                self.train(x, y)


# ====================================================================== #
#  Quick demo / smoke test                                                #
# ====================================================================== #
if __name__ == "__main__":
    np.random.seed(42)

    # Tiny dataset: learn y = 2*x  (scalar, treated as 1-D vectors)
    data = [
        (np.array([[0.5]]), np.array([[1.0]])),
        (np.array([[1.0]]), np.array([[2.0]])),
        (np.array([[1.5]]), np.array([[3.0]])),
        (np.array([[2.0]]), np.array([[4.0]])),
    ]

    input_size = 1
    hidden_size = 4
    output_size = 1

    w1 = np.random.randn(hidden_size, input_size) * 0.5
    w2 = np.random.randn(output_size, hidden_size) * 0.5
    b  = np.zeros((hidden_size, 1))

    nn = NeuralNet(w1, w2, b, lr=0.005, epochs=500)
    nn.train_neural_network(data)

    print("Predictions after training:")
    for x, y in data:
        pred = nn.predict(x)
        print(f"  x={x.item():.1f}  target={y.item():.1f}  predicted={pred:.4f}")
