"""
model_multiclass.py

Stage 2 of the two-stage plan (see PHASE2_MULTIBOARD_PLAN.md):
given an image that already passed the yes/no gate, predict WHICH
board it belongs to. Same skeleton as model.py — the only real
changes are at the output end:

    sigmoid              ->  softmax
    binary cross-entropy ->  categorical cross-entropy
    y as a single 0/1    ->  y as a one-hot vector

Everything else (ReLU hidden layer, the backward pass through W1,
gradient descent updates) is IDENTICAL to the binary model. That's
the point worth making in the write-up: multiclass isn't new
machinery, it's the same machinery with a wider output.
"""

import numpy as np

from model import relu, relu_derivative


def softmax(Z):
    """Turns a vector of raw scores into a probability distribution:
    every entry in (0, 1), and each row sums to exactly 1.

        softmax(z_i) = e^z_i / sum_j e^z_j

    Sigmoid is literally the 2-class special case of this (derived
    in PHASE2_MULTIBOARD_PLAN.md).

    The max-subtraction below is the standard stability trick:
    softmax(z) = softmax(z - c) for any constant c (the e^c cancels
    top and bottom), so we shift by the row max to keep e^z from
    overflowing. Changes nothing mathematically, saves us from inf."""
    Z = Z - Z.max(axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / expZ.sum(axis=1, keepdims=True)


class BoardNet:
    def __init__(self, n_features, n_classes, n_hidden=8, seed=42):
        rng = np.random.default_rng(seed)
        # Same symmetry-breaking logic as TasteNet: zero init would
        # make every hidden neuron identical forever.
        self.W1 = rng.normal(0, 0.5, size=(n_features, n_hidden))
        self.b1 = np.zeros(n_hidden)
        # Only structural difference from TasteNet: the output layer
        # is (n_hidden, n_classes) instead of (n_hidden, 1). One
        # column of W2 per board.
        self.W2 = rng.normal(0, 0.5, size=(n_hidden, n_classes))
        self.b2 = np.zeros(n_classes)
        self.n_classes = n_classes

    def forward(self, X):
        """X: (n_samples, n_features)
        Returns (n_samples, n_classes) of probabilities — each row
        is a full distribution over boards, summing to 1."""
        Z1 = X @ self.W1 + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = softmax(Z2)                # rows sum to 1
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, Y_true, Y_pred):
        """Categorical cross-entropy.

        Y_true is one-hot: exactly one 1 per row, marking the true
        board. So the sum over classes picks out just ONE term —
        -log(probability the model gave to the correct board).
        Confidently right -> loss near 0. Confidently wrong -> the
        correct class got probability near 0 -> -log explodes.
        Exactly the same behavior as BCE, generalized.

            loss = -mean over samples of sum_k y_k * log(p_k)"""
        eps = 1e-8
        Y_pred = np.clip(Y_pred, eps, 1.0)
        return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))

    def backward(self, Y_true, cache):
        """Backprop. The headline result (derived in
        PHASE2_MULTIBOARD_PLAN.md): softmax + categorical
        cross-entropy cancels down to

            dLoss/dZ2 = A2 - Y_true

        — the SAME (prediction - truth) shape as sigmoid + BCE,
        except now both sides are vectors per sample instead of
        single numbers. The cancellation isn't a coincidence twice;
        it's the same underlying identity, and the binary case is
        just this with 2 classes.

        Everything after dZ2 is copy-paste identical to model.py,
        because from W2 backwards the network doesn't know or care
        what activation produced the output error."""
        X, Z1, A1, Z2, A2 = cache
        n = X.shape[0]

        dZ2 = (A2 - Y_true)             # (n_samples, n_classes)

        dW2 = A1.T @ dZ2 / n
        db2 = dZ2.mean(axis=0)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * relu_derivative(Z1)

        dW1 = X.T @ dZ1 / n
        db1 = dZ1.mean(axis=0)

        return dW1, db1, dW2, db2

    def train_step(self, X, Y, lr=0.05):
        """Forward -> loss -> backward -> gradient descent update.
        Identical rhythm to TasteNet.train_step."""
        Y_pred, cache = self.forward(X)
        loss = self.compute_loss(Y, Y_pred)
        dW1, db1, dW2, db2 = self.backward(Y, cache)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

        return loss

    def predict_proba(self, X):
        """Full probability distribution over boards, per sample."""
        Y_pred, _ = self.forward(X)
        return Y_pred

    def predict(self, X):
        """The single most likely board index per sample.
        argmax = 'which entry is biggest' — the hard decision on
        top of the soft distribution."""
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        d = np.load(path)
        self.W1, self.b1, self.W2, self.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        self.n_classes = self.W2.shape[1]