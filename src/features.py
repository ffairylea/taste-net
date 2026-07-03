"""
features.py

Turns an image into a feature vector — a list of numbers describing it.
This is the "linear algebra" step: an image goes from a grid of pixels
to a single point in N-dimensional space. Everything downstream (the
model, the math, the learning) happens on these vectors, not on raw
pixels. Raw pixels would need way more data than we have (150-300
images) to learn anything real — that's the classic overfitting trap.

Every feature here is something YOU could compute by hand on a small
image if you had a calculator and a lot of patience. Nothing is a
black box.
"""

from PIL import Image
import numpy as np


def load_image(path, size=(64, 64)):
    """Load an image, force it to RGB, resize to a fixed size.
    Fixed size matters: every image needs to produce a feature
    vector of the SAME length so we can compare / train on them."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return np.array(img) / 255.0  # scale pixel values to [0, 1]


def average_color(img_array):
    """Mean of R, G, B across the whole image. 3 numbers.
    This is just an average — sum every pixel's R value, divide by
    count. Same for G and B."""
    return img_array.mean(axis=(0, 1))  # shape: (3,)


def brightness(img_array):
    """Overall brightness = average of the average color channels.
    Standard luminance-ish approximation."""
    avg = average_color(img_array)
    return float(0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2])


def saturation(img_array):
    """How 'colorful' vs 'gray' the image is, on average.
    For each pixel: saturation = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
    Then we average that over every pixel in the image."""
    r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0)
    return float(sat.mean())


def color_variance(img_array):
    """How much color varies across the image (busy vs calm).
    Variance = average squared distance from the mean. High variance
    = lots of different colors/tones. Low variance = flat, minimal."""
    return float(img_array.var())


def contrast(img_array):
    """Standard deviation of brightness across pixels.
    Std dev = sqrt(variance). Measures how much light/dark the image
    swings between light and dark, not just how varied the color is."""
    gray = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2]
    return float(gray.std())


def warmth(img_array):
    """Red+yellow vs blue balance. Positive = warm, negative = cool.
    A simple heuristic: (R - B) averaged over the image."""
    avg = average_color(img_array)
    return float(avg[0] - avg[2])


def color_histogram(img_array, bins=8):
    """A coarse histogram of colors present, per channel.
    This is a real distribution: counts how many pixels fall into
    each of `bins` brightness ranges, for each channel, then
    normalizes so it sums to 1 (turns counts into probabilities)."""
    hist = []
    for ch in range(3):
        h, _ = np.histogram(img_array[..., ch], bins=bins, range=(0, 1))
        h = h / (h.sum() + 1e-8)
        hist.extend(h.tolist())
    return hist  # length = bins * 3


def extract_features(path):
    """The full feature vector for one image.
    Returns a 1D numpy array. This is the point in N-dimensional
    space that represents this image for the model."""
    img = load_image(path)
    feats = [
        brightness(img),
        saturation(img),
        color_variance(img),
        contrast(img),
        warmth(img),
        *average_color(img),       # 3 numbers
        *color_histogram(img),     # 24 numbers (8 bins x 3 channels)
    ]
    return np.array(feats, dtype=np.float64)


FEATURE_NAMES = (
    ["brightness", "saturation", "color_variance", "contrast", "warmth",
     "avg_r", "avg_g", "avg_b"]
    + [f"hist_{ch}_{b}" for ch in ("r", "g", "b") for b in range(8)]
)
