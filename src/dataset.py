"""
dataset.py

Loads images from data/yes (things you'd pin) and data/no (things
you wouldn't), extracts features for each, and builds the X, y
arrays the model trains on.

Also handles normalization - a real, necessary step: features like
"color_variance" and "brightness" live on very different scales.
Without normalizing, the model would effectively ignore small-scale
features no matter how important they are, just because their
numbers are tiny compared to others. This is a genuine gradient
descent gotcha, not busywork.
"""

import os
import numpy as np
from features import extract_features, FEATURE_NAMES

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def load_folder(folder, label):
    X, y, paths = [], [], []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith(VALID_EXT):
            continue
        fpath = os.path.join(folder, fname)
        try:
            feats = extract_features(fpath)
            X.append(feats)
            y.append(label)
            paths.append(fpath)
        except Exception as e:
            print(f"  skipped {fname}: {e}")
    return X, y, paths


def build_dataset(yes_dir="data/yes", no_dir="data/no"):
    X_yes, y_yes, p_yes = load_folder(yes_dir, 1)
    X_no, y_no, p_no = load_folder(no_dir, 0)

    print(f"Loaded {len(X_yes)} 'yes' images, {len(X_no)} 'no' images")

    X = np.array(X_yes + X_no)
    y = np.array(y_yes + y_no, dtype=float)
    paths = p_yes + p_no

    return X, y, paths


def normalize(X, mean=None, std=None):
    """Z-score normalization: (x - mean) / std, per feature column.
    If mean/std aren't given, compute them from X (do this on
    TRAINING data only, then reuse those same values on test data -
    otherwise you leak information from test into train)."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1  # avoid divide-by-zero on constant columns
    return (X - mean) / std, mean, std


def train_test_split(X, y, paths, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return (
        X[train_idx], y[train_idx], [paths[i] for i in train_idx],
        X[test_idx], y[test_idx], [paths[i] for i in test_idx],
    )
