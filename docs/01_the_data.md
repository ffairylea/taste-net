# 01 — What is the input, actually?

An image starts as a giant grid of pixel numbers — way too many
(12,000+ for a 64x64 image) to learn from directly with only ~450
images. So `features.py` compresses each image down into just 29
numbers instead: brightness, saturation, warmth, color histogram
bins, and a few others.

Every one of these 29 numbers is a real, hand-computable statistic,
not a black box. For example, saturation for one pixel is:

    (max(R,G,B) - min(R,G,B)) / max(R,G,B)

averaged across every pixel in the image. Brightness, warmth, and the
rest are similarly plain arithmetic — see features.py for the exact
formulas.

## The feature vector

This list of 29 numbers is called a **feature vector**. Think of it
as one point in a 29-dimensional space — the same idea as a point
(x, y) on a normal 2D graph, just with 29 coordinates instead of 2.
An image isn't "a picture" to the model anymore at this stage — it's
just a location in this 29-dimensional space.

## Why this matters — the actual bet the whole project makes

The entire project rests on one geometric bet: if my real "yes"
(pink board) images cluster together in one region of this 29D
space, and "no" images cluster somewhere else, then a model just
needs to find a boundary separating those two regions.

Everything in the rest of this documentation — every neuron, every
layer, every gradient — exists purely to find that boundary.