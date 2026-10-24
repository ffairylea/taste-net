"""
features_spatial.py

Spatial features for the next experiment
(NEXT_EXPERIMENT_SPATIAL_FEATURES.md).

Every feature in features.py averages or bins over the whole image,
so position is gone before the network sees anything. docs/15 argues
that this is why fleurs vs pink sits at chance. The features here
keep some information about *where* things are and *what shape*
they have, while staying hand-computed and interpretable.

13 numbers per image:

    orientation_entropy     1   are edges straight (rooms) or spread
                                over every angle (petals)?
    fractal_dimension       1   how much the edge map branches
    center_sharpness        1   sharp middle, blurred border?
    mirror_symmetry         1   left-right symmetry
    grid_brightness_0..8    9   coarse layout, 3x3 grid

Only numpy and Pillow, like the rest of the project. The thresholds
below are fixed in advance on purpose: choosing them after looking at
fleurs vs pink results would be tuning on the test question.
"""

import numpy as np
from PIL import Image

SIZE = 128            # larger than features.py's 64: edges need detail
N_ORIENT_BINS = 12    # 15 degrees per bin
EDGE_QUANTILE = 0.85  # "edge" = strongest 15% of gradient magnitudes

SPATIAL_FEATURE_NAMES = (
    ["orientation_entropy", "fractal_dimension",
     "center_sharpness", "mirror_symmetry"]
    + [f"grid_brightness_{i}" for i in range(9)]
)


def load_gray(path, size=SIZE):
    """Greyscale, fixed size, values in [0, 1]."""
    img = Image.open(path).convert("L").resize((size, size))
    return np.array(img, dtype=np.float64) / 255.0


def conv3x3(img, kernel):
    """Slide a 3x3 kernel over the image (valid region only, so the
    output is 2 pixels smaller in each direction). Written out with
    slicing instead of a library call so every step is visible: for
    each of the 9 kernel positions, shift the image and add it in,
    weighted.

    Strictly this is cross-correlation (the kernel isn't flipped),
    which is also what CNN layers compute. For the symmetric and
    antisymmetric kernels used here the difference is at most a sign.
    """
    h, w = img.shape
    out = np.zeros((h - 2, w - 2))
    for dy in range(3):
        for dx in range(3):
            out += kernel[dy, dx] * img[dy:dy + h - 2, dx:dx + w - 2]
    return out


SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T
LAPLACIAN = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float64)


def gradients(gray):
    """Edge strength and edge orientation at every pixel."""
    gx = conv3x3(gray, SOBEL_X)
    gy = conv3x3(gray, SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    # Orientation, not direction: a dark-to-light edge and a
    # light-to-dark edge along the same line are the same line, so
    # angles are folded into [0, pi).
    angle = np.mod(np.arctan2(gy, gx), np.pi)
    return magnitude, angle


def edge_mask(magnitude):
    """The strongest 15% of gradients count as edges. A quantile
    rather than a fixed value, so a dim photo and a bright one are
    judged on the same terms.

    The "> 1e-6" matters for images with large flat areas: if more
    than 85% of pixels have zero gradient, the quantile itself is 0,
    and ">= 0" would mark every pixel as an edge. (The synthetic
    check caught exactly this.)"""
    threshold = np.quantile(magnitude, EDGE_QUANTILE)
    return (magnitude >= threshold) & (magnitude > 1e-6)


def orientation_entropy(magnitude, angle):
    """Histogram of edge angles, weighted by edge strength, then its
    entropy, scaled to [0, 1].

    0 = every edge at one angle. 1 = edges spread evenly over all
    angles. A room (horizontal floor lines, vertical walls) should
    sit low; petals curve through every angle and should sit high.
    """
    mask = edge_mask(magnitude)
    hist, _ = np.histogram(angle[mask], bins=N_ORIENT_BINS,
                           range=(0, np.pi), weights=magnitude[mask])
    if hist.sum() == 0:
        return 0.0
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(N_ORIENT_BINS))


def fractal_dimension(mask, box_sizes=(2, 4, 8, 16, 32)):
    """Box-counting dimension of the edge map.

    Cover the image with a grid of boxes of side s and count how many
    contain at least one edge pixel, N(s). For a straight line, halving
    s doubles N (dimension 1). For something that fills the plane,
    halving s quadruples N (dimension 2). Branching, organic edges sit
    in between. The dimension is the slope of log N(s) against
    log(1/s).
    """
    h, w = mask.shape
    counts = []
    for s in box_sizes:
        hh, ww = (h // s) * s, (w // s) * s
        boxes = mask[:hh, :ww].reshape(hh // s, s, ww // s, s)
        counts.append(boxes.any(axis=(1, 3)).sum())
    counts = np.maximum(np.array(counts, dtype=np.float64), 1)
    slope, _ = np.polyfit(np.log(1 / np.array(box_sizes)),
                          np.log(counts), 1)
    return float(slope)


def center_sharpness(gray):
    """log( sharpness in the middle / sharpness at the border ).

    Sharpness = variance of the Laplacian, a standard focus measure:
    the Laplacian responds to rapid changes, and blur removes them.
    The middle is the central half of the image in each direction
    (a quarter of the area); the border is everything else.

    > 0: sharper in the middle (typical flower close-up with a
    blurred background). Around 0: sharp throughout (typical room).
    """
    lap = conv3x3(gray, LAPLACIAN)
    h, w = lap.shape
    center = np.zeros_like(lap, dtype=bool)
    center[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
    eps = 1e-8
    return float(np.log((lap[center].var() + eps)
                        / (lap[~center].var() + eps)))


def mirror_symmetry(gray):
    """Correlation between the image and its left-right flip.
    1 = perfectly mirror-symmetric, around 0 = unrelated halves."""
    a = gray - gray.mean()
    b = a[:, ::-1]
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def grid_brightness(gray):
    """Mean brightness in each cell of a 3x3 grid, read row by row.
    The cheapest way to put position back in: "bright top, dark
    bottom" becomes visible, where a single average brightness
    cannot tell it apart from the reverse."""
    return [float(cell.mean())
            for band in np.array_split(gray, 3, axis=0)
            for cell in np.array_split(band, 3, axis=1)]


def spatial_features_from_gray(gray):
    magnitude, angle = gradients(gray)
    return np.array([
        orientation_entropy(magnitude, angle),
        fractal_dimension(edge_mask(magnitude)),
        center_sharpness(gray),
        mirror_symmetry(gray),
        *grid_brightness(gray),
    ], dtype=np.float64)


def extract_spatial_features(path):
    return spatial_features_from_gray(load_gray(path))
