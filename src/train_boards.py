"""
train_boards.py

Stage 2 pipeline: load board folders -> features -> normalize ->
train the softmax network -> evaluate -> confusion matrix.

Run from inside src/:
    python train_boards.py

If data/boards/ doesn't exist yet (images not collected), this
falls back to a SYNTHETIC sanity check instead of dying:
  1. a numerical gradient check — the same "compute the slope by
     nudging a weight and re-running the loss" trick used to verify
     the sigmoid+BCE cancellation, now verifying dZ2 = A2 - Y for
     softmax+CCE. If analytic and numerical gradients match to ~6
     decimal places, the backward pass is provably correct.
  2. training on fake clustered data the model SHOULD be able to
     learn. If it can't hit high accuracy on easy synthetic
     clusters, something is wrong in the code, not the data.

That means the code can be verified as correct before any board
images exist to make sure it isn't about the data set.
"""

import os
import numpy as np

from dataset import normalize
from dataset_boards import build_board_dataset, one_hot
from model_multiclass import BoardNet


# ---------------------------------------------------------------
# sanity checks (run when real data isn't there yet)
# ---------------------------------------------------------------

def numerical_gradient_check(seed=0):
    """Verify the analytic gradients from backward() against
    brute-force numerical slopes: nudge one weight by +h and -h,
    re-run the loss both times, and the slope is
        (loss(w+h) - loss(w-h)) / (2h).
    Just the definition of a derivative. If this
    matches what the chain-rule code produces, the math AND the
    code are right."""
    rng = np.random.default_rng(seed)
    n_samples, n_features, n_classes = 5, 4, 3

    X = rng.normal(size=(n_samples, n_features))
    labels = rng.integers(0, n_classes, size=n_samples)
    Y = one_hot(labels, n_classes)

    net = BoardNet(n_features, n_classes, n_hidden=6, seed=1)
    Y_pred, cache = net.forward(X)
    dW1, db1, dW2, db2 = net.backward(Y, cache)

    h = 1e-5
    worst = 0.0
    # check a handful of individual weights in each matrix
    for (name, W, dW) in [("W1", net.W1, dW1), ("W2", net.W2, dW2)]:
        for _ in range(5):
            i = rng.integers(W.shape[0])
            j = rng.integers(W.shape[1])
            orig = W[i, j]

            W[i, j] = orig + h
            loss_plus = net.compute_loss(Y, net.forward(X)[0])
            W[i, j] = orig - h
            loss_minus = net.compute_loss(Y, net.forward(X)[0])
            W[i, j] = orig  # restore

            numerical = (loss_plus - loss_minus) / (2 * h)
            analytic = dW[i, j]
            diff = abs(numerical - analytic)
            worst = max(worst, diff)
            print(f"  {name}[{i},{j}]  analytic {analytic:+.8f}  "
                  f"numerical {numerical:+.8f}  diff {diff:.2e}")

    print(f"\nWorst disagreement: {worst:.2e} "
          f"({'PASS' if worst < 1e-6 else 'FAIL'} — should be < 1e-6)")
    return worst < 1e-6


def synthetic_training_check(seed=0):
    """Fake data the model should definitely learn: 3 clearly
    separated Gaussian clusters in feature space, one per fake
    'board'. If the network can't classify THIS, the bug is in the
    code, full stop."""
    rng = np.random.default_rng(seed)
    n_per_class, n_features, n_classes = 60, 29, 3

    # Feature count here is arbitrary on purpose — the model must
    # work for ANY n_features (the real pipeline currently produces
    # 32: 5 stats + 3 avg colors + 24 histogram bins).
    centers = rng.normal(0, 3, size=(n_classes, n_features))
    X = np.vstack([
        centers[c] + rng.normal(0, 1, size=(n_per_class, n_features))
        for c in range(n_classes)
    ])
    labels = np.repeat(np.arange(n_classes), n_per_class)
    Y = one_hot(labels, n_classes)

    X_norm, _, _ = normalize(X)

    net = BoardNet(n_features, n_classes, n_hidden=8)
    for epoch in range(300):
        loss = net.train_step(X_norm, Y, lr=0.1)
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}  loss {loss:.4f}")

    acc = (net.predict(X_norm) == labels).mean()
    print(f"\nSynthetic accuracy: {acc:.2%} "
          f"({'PASS' if acc > 0.95 else 'FAIL'} — separated clusters "
          f"should be nearly perfectly learnable)")
    return acc > 0.95


# ---------------------------------------------------------------
# the real pipeline
# ---------------------------------------------------------------

def confusion_matrix(y_true, y_pred, n_classes):
    """M[i, j] = how many images from true board i were predicted
    as board j. The diagonal is where the model is right; every
    off-diagonal cell is a specific, nameable confusion
    ('winter_minimal keeps getting called pink'). This is the real
    evidence piece from the Phase 2 plan."""
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def print_confusion(M, boards):
    width = max(len(b) for b in boards) + 2
    header = " " * width + "".join(f"{b[:8]:>10s}" for b in boards)
    print(header)
    for i, b in enumerate(boards):
        row = "".join(f"{M[i, j]:>10d}" for j in range(len(boards)))
        print(f"{b:<{width}s}{row}")


def train_test_split_stratified(X, Y, labels, test_frac=0.2, seed=42):
    """Like dataset.train_test_split, but STRATIFIED: split each
    board separately, so a small board can't end up with zero test
    images by bad luck. Matters more here than in the binary case
    because boards may be different sizes."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = rng.permutation(idx)
        n_test = max(1, int(len(idx) * test_frac))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    return (X[train_idx], Y[train_idx], labels[train_idx],
            X[test_idx], Y[test_idx], labels[test_idx])


def main():
    boards_dir = "../data/boards"

    if not os.path.isdir(boards_dir):
        print(f"No {boards_dir}/ yet — running sanity checks on the "
              f"code itself instead.\n")
        print("=" * 60)
        print("1. Numerical gradient check (is dZ2 = A2 - Y actually")
        print("   the right gradient? let's measure the slope directly)")
        print("=" * 60)
        ok1 = numerical_gradient_check()
        print()
        print("=" * 60)
        print("2. Synthetic training check (can it learn easy clusters?)")
        print("=" * 60)
        ok2 = synthetic_training_check()
        print()
        if ok1 and ok2:
            print("Both checks PASS — the Stage 2 code is verified "
                  "correct. Collect board images into "
                  "data/boards/<board_name>/ and rerun.")
        else:
            print("A check FAILED — there is a bug to find before "
                  "collecting data.")
        return

    X, Y, labels, boards, paths = build_board_dataset(boards_dir)
    n_classes = len(boards)
    print(f"\nBoards ({n_classes}): {boards}")

    (X_train, Y_train, l_train,
     X_test, Y_test, l_test) = train_test_split_stratified(X, Y, labels)

    X_train_norm, mean, std = normalize(X_train)
    X_test_norm, _, _ = normalize(X_test, mean, std)

    net = BoardNet(n_features=X.shape[1], n_classes=n_classes, n_hidden=8)

    epochs = 500
    for epoch in range(epochs):
        loss = net.train_step(X_train_norm, Y_train, lr=0.1)
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss:.4f}")

    train_acc = (net.predict(X_train_norm) == l_train).mean()
    test_acc = (net.predict(X_test_norm) == l_test).mean()
    print(f"\nFinal train accuracy: {train_acc:.2%}")
    print(f"Final test accuracy:  {test_acc:.2%}")
    print(f"(random guessing baseline with {n_classes} boards: "
          f"{1/n_classes:.2%} — always compare against this, not 50%)")

    print("\nConfusion matrix (rows = true board, cols = predicted):")
    M = confusion_matrix(l_test, net.predict(X_test_norm), n_classes)
    print_confusion(M, boards)

    net.save("../board_net_weights.npz")
    # Save the board order too — the weights are meaningless without
    # knowing which output index means which board.
    with open("../board_net_classes.txt", "w") as f:
        f.write("\n".join(boards))
    print("\nSaved weights to ../board_net_weights.npz "
          "and class order to ../board_net_classes.txt")


if __name__ == "__main__":
    main()