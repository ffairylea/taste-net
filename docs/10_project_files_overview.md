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