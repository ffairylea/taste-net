"""
dataset_boards.py

Stage 2 data loading. Instead of yes/no folders, each board gets
its own folder (see PHASE2_MULTIBOARD_PLAN.md):

    data/boards/
      pink/
      winter_minimal/
      ...

Board names are discovered from the folder names, so adding a new
board = making a new folder and dropping images in. No code change.

Labels here are one-hot vectors instead of single 0/1 numbers:
if the boards are [pink, winter_minimal, cottage], then a
winter_minimal image gets the label [0, 1, 0]. One-hot is what
makes the categorical cross-entropy sum "pick out" exactly the
predicted probability of the correct class.

Normalization and train/test splitting are reused straight from
dataset.py - that logic doesn't care how many classes exist.
"""

import os
import numpy as np
from features import extract_features

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def discover_boards(boards_dir="data/boards"):
    """Each subfolder = one board = one class. Sorted so the
    class order (and therefore the meaning of each one-hot slot)
    is stable across runs."""
    boards = sorted(
        d for d in os.listdir(boards_dir)
        if os.path.isdir(os.path.join(boards_dir, d))
    )
    return boards


def build_board_dataset(boards_dir="data/boards"):
    """Returns:
        X      (n_samples, n_features)  - feature vectors
        Y      (n_samples, n_classes)   - one-hot labels
        labels (n_samples,)             - class indices (for
                                          confusion matrix / splits)
        boards list of board names       - index i <-> boards[i]
        paths  list of file paths
    """
    boards = discover_boards(boards_dir)
    if len(boards) < 2:
        raise ValueError(
            f"Need at least 2 board folders inside {boards_dir}, "
            f"found {len(boards)}: {boards}"
        )

    X, labels, paths = [], [], []
    for class_idx, board in enumerate(boards):
        folder = os.path.join(boards_dir, board)
        count = 0
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(VALID_EXT):
                continue
            fpath = os.path.join(folder, fname)
            try:
                X.append(extract_features(fpath))
                labels.append(class_idx)
                paths.append(fpath)
                count += 1
            except Exception as e:
                print(f"  skipped {fname}: {e}")
        print(f"Loaded {count} images for board '{board}'")

    X = np.array(X)
    labels = np.array(labels)
    Y = one_hot(labels, n_classes=len(boards))
    return X, Y, labels, boards, paths


def one_hot(labels, n_classes):
    """Turn class indices into one-hot rows.
    label 2 with 4 classes -> [0, 0, 1, 0].
    np.eye(n) is the identity matrix; indexing its rows by label
    is the standard one-liner for this."""
    return np.eye(n_classes)[labels]