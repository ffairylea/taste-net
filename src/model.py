"""
model.py

A small neural network, built from raw numpy. No PyTorch, no
TensorFlow, no sklearn. Every piece of math here is math YOU
are responsible for understanding — that's the entire point of
this project.

Architecture: input features -> hidden layer -> 1 output (probability)
This is the simplest possible "deep" network: one hidden layer.
Whether the hidden layer is even NECESSARY (vs. a single linear
layer = logistic regression) is itself one of the interesting
questions this project can answer about your data.
"""

import numpy as np


def sigmoid(z):
    """Squashes any real number into (0, 1) - turns a raw score
    into something we can treat as a probability.
    sigmoid(z) = 1 / (1 + e^-z)"""
    z = np.clip(z, -500, 500)  # avoid overflow on extreme values
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid, expressed in terms of its OWN output.
    This identity (a * (1-a)) is why sigmoid is convenient for
    backprop by hand - you don't need to recompute from scratch."""
    return a * (1 - a)


def relu(z):
    """max(0, z) - the standard hidden-layer activation. Introduces
    non-linearity so the network can learn curved decision boundaries,
    not just straight lines."""
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


class TasteNet:
    def __init__(self, n_features, n_hidden=8, seed=42):
        rng = np.random.default_rng(seed)
        # Small random weights to start - if weights start at zero,
        # every hidden neuron would learn the identical thing (symmetry
        # problem), so we break symmetry with small random noise.
        self.W1 = rng.normal(0, 0.5, size=(n_features, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, 0.5, size=(n_hidden, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        """X: (n_samples, n_features)
        Returns the predicted probability for each sample, plus the
        intermediate values we'll need for backprop."""
        Z1 = X @ self.W1 + self.b1        # linear step 1
        A1 = relu(Z1)                     # non-linearity
        Z2 = A1 @ self.W2 + self.b2       # linear step 2
        A2 = sigmoid(Z2).flatten()        # squash to probability
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy: the standard loss for yes/no
        classification. Penalizes confident WRONG predictions
        much more heavily than uncertain ones.
        loss = -[y*log(p) + (1-y)*log(1-p)], averaged over samples."""
        eps = 1e-8  # avoid log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true, cache):
        """Backpropagation, by hand: the chain rule applied layer by
        layer, from the loss back to every weight.
        This is THE core calculus of the whole project - gradients
        tell us which direction to nudge each weight to reduce loss."""
        X, Z1, A1, Z2, A2 = cache
        n = X.shape[0]

        # dLoss/dZ2: for sigmoid + cross-entropy together, this
        # simplifies beautifully to (prediction - truth).
        dZ2 = (A2 - y_true).reshape(-1, 1)
        dW2 = A1.T @ dZ2 / n
        db2 = dZ2.mean(axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = X.T @ dZ1 / n
        db1 = dZ1.mean(axis=0)

        return dW1, db1, dW2, db2

    def train_step(self, X, y, lr=0.05):
        """One full pass: forward, compute loss, backward, update
        weights by gradient descent (weight -= lr * gradient)."""
        y_pred, cache = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        dW1, db1, dW2, db2 = self.backward(y, cache)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2.flatten()

        return loss

    def predict(self, X):
        y_pred, _ = self.forward(X)
        return y_pred

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
