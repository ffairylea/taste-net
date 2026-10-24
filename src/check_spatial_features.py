"""
check_spatial_features.py

Sanity check for features_spatial.py on synthetic images, before any
real data is involved. Same idea as the gradient check in
train_boards.py: if the features don't behave as intended on images
built to have a known answer, a result on real boards means nothing.

Three synthetic images, 128x128:

    room       axis-aligned rectangles (walls, a window, a bed),
               sharp everywhere, not symmetric
    flower     an eight-petal rose curve, sharp, centered, on a
               blurred noisy background
    branching  a recursive tree of lines, the kind of structure
               box-counting dimension is designed to measure

Each check is a directional prediction written before running it.

Run from inside src/:
    python3 check_spatial_features.py
"""

import numpy as np
from features_spatial import (SIZE, SPATIAL_FEATURE_NAMES, gradients,
                              edge_mask, fractal_dimension,
                              spatial_features_from_gray)


def blur(img, passes=4):
    """Repeated 3x3 box blur, edges padded so the size is kept."""
    for _ in range(passes):
        p = np.pad(img, 1, mode="edge")
        img = sum(p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
                  for dy in range(3) for dx in range(3)) / 9
    return img


def make_room():
    img = np.full((SIZE, SIZE), 0.85)
    img[80:, :] = 0.6                      # floor
    img[20:60, 70:115] = 1.0               # window
    img[20:60, 91:94] = 0.4                # window frame
    img[38:41, 70:115] = 0.4
    img[85:110, 10:70] = 0.95              # bed
    img[78:85, 10:70] = 0.5                # headboard
    img[90:120, 90:100] = 0.3              # table leg
    return img


def make_flower(seed=0):
    rng = np.random.default_rng(seed)
    background = blur(rng.random((SIZE, SIZE)), passes=8) * 0.6
    y, x = np.mgrid[0:SIZE, 0:SIZE] - SIZE / 2 + 0.5
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    petals = r < 40 * np.abs(np.cos(4 * theta)) + 8
    img = background.copy()
    img[petals] = 0.9
    img[r < 7] = 0.3                        # center
    return img


def make_branching():
    img = np.zeros((SIZE, SIZE))

    def branch(x, y, angle, length, depth):
        if depth == 0 or length < 2:
            return
        x2 = x + length * np.cos(angle)
        y2 = y - length * np.sin(angle)
        for t in np.linspace(0, 1, int(length * 2) + 1):
            xi, yi = int(x + t * (x2 - x)), int(y + t * (y2 - y))
            if 0 <= xi < SIZE and 0 <= yi < SIZE:
                img[yi, xi] = 1.0
        branch(x2, y2, angle + 0.5, length * 0.72, depth - 1)
        branch(x2, y2, angle - 0.5, length * 0.72, depth - 1)

    branch(SIZE / 2, SIZE - 2, np.pi / 2, 34, 9)
    return img


def main():
    images = {"room": make_room(), "flower": make_flower(),
              "branching": make_branching()}
    feats = {k: spatial_features_from_gray(v) for k, v in images.items()}

    print(f"{'feature':<22}" + "".join(f"{k:>12}" for k in images))
    for j, name in enumerate(SPATIAL_FEATURE_NAMES[:4]):
        print(f"{name:<22}" + "".join(f"{feats[k][j]:>12.3f}"
                                      for k in images))

    idx = {n: i for i, n in enumerate(SPATIAL_FEATURE_NAMES)}
    f = lambda img, name: feats[img][idx[name]]

    # Edges here are thin 1-pixel lines, so box-count the line pixels
    # directly as a reference for what a straight-line room scores.
    mag, _ = gradients(images["room"])
    room_dim = fractal_dimension(edge_mask(mag))

    checks = [
        ("flower edges spread over more angles than room edges",
         f("flower", "orientation_entropy") > f("room", "orientation_entropy")),
        ("flower is sharper in the middle than the room is",
         f("flower", "center_sharpness") > f("room", "center_sharpness")),
        ("flower is more mirror-symmetric than the room",
         f("flower", "mirror_symmetry") > f("room", "mirror_symmetry")),
        ("branching edges have higher box-counting dimension than room",
         f("branching", "fractal_dimension") > room_dim),
    ]
    print()
    for desc, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {desc}")
    print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks pass")


if __name__ == "__main__":
    main()
