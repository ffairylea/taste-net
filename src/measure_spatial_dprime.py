"""
measure_spatial_dprime.py

Step 1 of NEXT_EXPERIMENT_SPATIAL_FEATURES.md: how well does each new
spatial feature separate the boards, with no model involved?

For every feature, d' (from probe_representation.py) on:

    fleurs vs pink             the split color features fail on
    pink-ish vs winter_minimal the split color features handle

next to the same numbers for the 32 color features, so the
comparison is like for like. The bar to beat on fleurs vs pink is
the best color feature, d' = 0.63.

Commit the prediction BEFORE running this. d' on real data is
already a result, and a prediction written after seeing it is not a
prediction.

Run from inside src/:
    python3 measure_spatial_dprime.py
"""

import os
import numpy as np
from dataset_boards import discover_boards, VALID_EXT
from features import extract_features, FEATURE_NAMES
from features_spatial import extract_spatial_features, SPATIAL_FEATURE_NAMES
from probe_representation import d_prime

BOARDS_DIR = "../data/boards"


def load(boards_dir=BOARDS_DIR):
    boards = discover_boards(boards_dir)
    color, spatial, labels = [], [], []
    for i, board in enumerate(boards):
        folder = os.path.join(boards_dir, board)
        files = [f for f in sorted(os.listdir(folder))
                 if f.lower().endswith(VALID_EXT)]
        for fname in files:
            path = os.path.join(folder, fname)
            try:
                c = extract_features(path)
                s = extract_spatial_features(path)
            except Exception as e:
                print(f"  skipped {fname}: {e}")
                continue
            color.append(c)
            spatial.append(s)
            labels.append(i)
        print(f"Loaded {labels.count(i)} images for '{board}'")
    return np.array(color), np.array(spatial), np.array(labels), boards


def separation(X, labels, a_mask, b_mask):
    return np.array([d_prime(X[a_mask][:, j], X[b_mask][:, j])
                     for j in range(X.shape[1])])


def main():
    if not os.path.isdir(BOARDS_DIR):
        raise SystemExit(f"No {BOARDS_DIR}/ - board images stay local, "
                         "so run this on the machine that has them.")
    color, spatial, labels, boards = load()
    fl, pk = boards.index("fleurs"), boards.index("pink")
    wm = boards.index("winter_minimal")
    fleurs, pink, winter = labels == fl, labels == pk, labels == wm
    pinkish = fleurs | pink

    c_hard = separation(color, labels, fleurs, pink)
    c_easy = separation(color, labels, pinkish, winter)
    s_hard = separation(spatial, labels, fleurs, pink)
    s_easy = separation(spatial, labels, pinkish, winter)

    print("\nd' per spatial feature, sorted by fleurs vs pink:\n")
    print(f"  {'feature':<22}{'fleurs/pink':>13}{'pinkish/winter':>16}")
    for j in np.argsort(-s_hard):
        print(f"  {SPATIAL_FEATURE_NAMES[j]:<22}"
              f"{s_hard[j]:>13.2f}{s_easy[j]:>16.2f}")

    print("\nSummary            fleurs/pink        pinkish/winter")
    print(f"  color   (32)   best {c_hard.max():.2f} "
          f"({FEATURE_NAMES[c_hard.argmax()]}), mean {c_hard.mean():.2f}"
          f"   best {c_easy.max():.2f}, mean {c_easy.mean():.2f}")
    print(f"  spatial (13)   best {s_hard.max():.2f} "
          f"({SPATIAL_FEATURE_NAMES[s_hard.argmax()]}), "
          f"mean {s_hard.mean():.2f}"
          f"   best {s_easy.max():.2f}, mean {s_easy.mean():.2f}")

    print("""
Reading it: a spatial feature with d' clearly above the best color
feature on fleurs vs pink supports the representation claim from a
new direction. If no spatial feature beats it, these particular
features don't capture what separates the boards, which is a result
too. d' looks at one feature at a time, so it cannot see features
that only work in combination; the ablation runs are for that.""")


if __name__ == "__main__":
    main()
