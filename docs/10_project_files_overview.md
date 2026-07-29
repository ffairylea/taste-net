# 10 — What does each file in this project actually do?

A map of every file in src/, for anyone (including future me)
trying to understand how the pieces fit together.

## features.py

Turns one image into the 29-number feature vector described in file
01 — brightness, saturation, warmth, color histogram, etc. Each
function computes one real, hand-checkable statistic from the raw
pixel grid. This is the only file that touches actual image pixels;
everything downstream only ever sees the 29 numbers it produces.

## dataset.py

Loads every image from data/yes and data/no, runs each one through
features.py to get its feature vector, and assembles the full X
(features) and y (labels) arrays the model trains on. Also handles:

- Normalization (z-score: subtract the mean, divide by the
  standard deviation, per feature column) — necessary because
  features like brightness (0 to 1) and color_variance (a very
  different scale) would otherwise cause the model to effectively
  ignore small-scale features regardless of how important they are,
  since gradient descent responds to raw magnitude.
- Train/test split — holds out a portion of images the model
  never trains on, so accuracy can be checked on data it hasn't
  memorized. This is what produces the train vs. test accuracy gap
  I use to check for overfitting.

## model.py

The actual neural network — covered in full in files 03-09.
Everything mathematical lives here: forward pass, loss, backward
pass, weight updates.

## train.py

Ties dataset.py and model.py together: loads the data, normalizes
it, splits it, creates a TasteNet, runs the training loop for 500
epochs (calling train_step from model.py each time), then reports
final accuracy and rough feature importance (which input features
ended up with the largest average weight magnitude feeding into the
hidden layer — a rough signal for what the model leaned on most).

## dedupe.py

Not part of the model at all — a data-cleaning utility. Finds exact
duplicate images (by content hash, not filename) and removes them,
since duplicates in the training data would bias the model toward
whatever happened to be repeated.

## fetch_unsplash_no.py

Also not part of the model — a data-collection utility. Pulls
varied, randomized photos from Unsplash's API to build the "no"
contrast set, since manually downloading ~150 images one at a time
isn't practical. Tracks what's already been downloaded per category
so it can resume rather than restart if the API rate limit is hit
partway through.

## The dependency chain, start to finish

    fetch_unsplash_no.py / manual Pinterest download
              |
              v
        data/yes/, data/no/  (raw images)
              |
              v
         dedupe.py  (remove exact duplicates)
              |
              v
        features.py  (image -> 29 numbers)
              |
              v
        dataset.py  (assemble X, y; normalize; split)
              |
              v
         model.py  (the actual network)
              |
              v
         train.py  (runs everything, reports results)





## features.py — implementation details

### load_image

    def load_image(path, size=(64, 64)):
        img = Image.open(path).convert("RGB")
        img = img.resize(size)
        return np.array(img) / 255.0

`.convert("RGB")` forces every image into the same 3-channel
(red/green/blue) format — some images might be grayscale or have a
4th "transparency" channel (RGBA), and without this conversion,
feature vectors would come out inconsistent in shape, breaking
everything downstream.

`.resize(size)` forces every image to exactly 64x64 pixels,
regardless of original size. Two images of different original
dimensions would otherwise produce different-length pixel arrays,
and the model needs every feature vector to be the exact same
length to compare/train on them.

`/ 255.0` — raw pixel values are integers 0-255 (standard for
images). Dividing by 255 rescales everything into the 0-to-1 range —
a normalization step, similar in spirit to why dataset.py normalizes
features later.

### saturation

    def saturation(img_array):
        r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = np.where(mx > 0, (mx - mn) / (mx + 1e-8), 0)
        return float(sat.mean())

`img_array[..., 0]` — the `...` means "every row, every column," and
`0` picks out just the red channel (index 0 of the 3 RGB channels),
so `r` becomes a whole 64x64 grid of just red values.

`np.where(mx > 0, (mx-mn)/(mx+1e-8), 0)` is a conditional, computed
for every pixel at once: IF the max channel value is greater than 0,
compute the saturation formula; OTHERWISE (a pure black pixel, all
channels 0), just use 0 — avoiding a divide-by-zero error.

`.mean()` computes saturation for every individual pixel, then
averages across all 4096 pixels (64x64) into one single number
representing the whole image's overall saturation.

## dataset.py — implementation details

### normalize

    def normalize(X, mean=None, std=None):
        if mean is None:
            mean = X.mean(axis=0)
        if std is None:
            std = X.std(axis=0)
            std[std == 0] = 1
        return (X - mean) / std, mean, std

`axis=0` — X is a grid (many images x 29 features). `axis=0` means
"compute the mean/std down each column separately" (one mean value
per feature, computed across all images), not one single overall
average mixing everything together. Each of the 29 features gets
its own personalized rescaling, since features like brightness and
color_variance live on very different natural scales.

`std[std==0]=1` — if some feature happened to be exactly identical
across every image (zero variation, std=0), dividing by that zero
would crash. Setting it to 1 instead means "don't rescale this
feature at all" rather than erroring out.

### train_test_split

    def train_test_split(X, y, paths, test_frac=0.2, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_test = int(len(X) * test_frac)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

`rng.permutation(len(X))` generates a random shuffling of all index
numbers (0, 1, 2, ... up to however many images there are), so
images get randomly assigned to train or test — not just "first 80%
vs last 20%," which could accidentally group similar images together
if they were added to the folder in some non-random order.

Why hold out test data at all? This is the mechanism behind the
train vs. test accuracy comparison in the actual training results —
the model never sees the test images during training, so testing on
them checks whether it learned a real, generalizable pattern versus
just memorizing the specific training images.