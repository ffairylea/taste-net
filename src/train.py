"""
train.py

Runs the whole pipeline: load images -> extract features -> normalize
-> train the network with gradient descent -> evaluate -> report
which features mattered most.

Run this from inside src/:
    python train.py
"""

import numpy as np
from dataset import build_dataset, normalize, train_test_split
from model import TasteNet
from features import FEATURE_NAMES


def main():
    X, y, paths = build_dataset(yes_dir="../data/yes", no_dir="../data/no")

    if len(X) < 20:
        print("\nNot enough images yet — you need real data in "
              "../data/yes and ../data/no before this will mean anything.")
        return

    X_train, y_train, p_train, X_test, y_test, p_test = train_test_split(X, y, paths)

    X_train_norm, mean, std = normalize(X_train)
    X_test_norm, _, _ = normalize(X_test, mean, std)

    net = TasteNet(n_features=X.shape[1], n_hidden=8)

    epochs = 500
    losses = []
    for epoch in range(epochs):
        loss = net.train_step(X_train_norm, y_train, lr=0.1)
        losses.append(loss)
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss:.4f}")

    # Evaluate
    train_preds = net.predict(X_train_norm)
    test_preds = net.predict(X_test_norm)
    train_acc = ((train_preds > 0.5) == y_train).mean()
    test_acc = ((test_preds > 0.5) == y_test).mean()

    print(f"\nFinal train accuracy: {train_acc:.2%}")
    print(f"Final test accuracy:  {test_acc:.2%}")
    print("(if train >> test accuracy, that's overfitting — the model "
          "memorized instead of generalizing. Expected with small data, "
          "worth discussing honestly in your write-up, not hiding.)")

    # Rough feature importance: average absolute weight magnitude
    # feeding OUT of each input feature, across the hidden layer.
    importance = np.abs(net.W1).mean(axis=1)
    order = np.argsort(-importance)
    print("\nRough feature importance (not a rigorous method, but a")
    print("real signal for which features the model leaned on):")
    for i in order[:8]:
        print(f"  {FEATURE_NAMES[i]:16s} {importance[i]:.4f}")

    net.save("../taste_net_weights.npz")
    print("\nSaved weights to ../taste_net_weights.npz")


if __name__ == "__main__":
    main()
