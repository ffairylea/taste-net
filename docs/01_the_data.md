# 01: What is the input, actually?

An image starts as a giant grid of pixel numbers; way too many
(12,000+ for a 64x64 image) to learn from directly with only ~350
images. So `features.py` compresses each image down into just 32
numbers instead: brightness, saturation, warmth, color histogram
bins, and a few others.

Every one of these 32 numbers is a real, hand-computable statistic
averaged across every pixel in the image. Brightness, warmth, and the
rest are plain arithmetic.

## Every formula, worked on a 2×2 image

Rather than describe the features, here is a tiny image with all 32
computed by hand. Four pixels, RGB values scaled to [0, 1]:

    pixel (0,0) = (1.0, 0.0, 0.0)   pure red
    pixel (0,1) = (0.0, 0.0, 1.0)   pure blue
    pixel (1,0) = (0.5, 0.5, 0.5)   mid grey
    pixel (1,1) = (1.0, 1.0, 0.0)   yellow

### Average color: 3 numbers

Mean of each channel across all pixels.

    avg_r = (1.0 + 0.0 + 0.5 + 1.0) / 4 = 0.625
    avg_g = (0.0 + 0.0 + 0.5 + 1.0) / 4 = 0.375
    avg_b = (0.0 + 1.0 + 0.5 + 0.0) / 4 = 0.375

### Brightness: 1 number

Not a plain average of the three channels. The coefficients are the
standard luminance weights:

    brightness = 0.299·avg_r + 0.587·avg_g + 0.114·avg_b
               = 0.299(0.625) + 0.587(0.375) + 0.114(0.375)
               = 0.4498

Green is weighted about five times more heavily than blue because
human vision is far more sensitive to green light. A pure green
patch genuinely looks brighter than a pure blue one of the same
numeric intensity, and an unweighted mean would call them equal.

### Saturation: 1 number

Per pixel, how far it is from grey, then averaged:

    sat = (max(R,G,B) − min(R,G,B)) / max(R,G,B)

    pure red:  (1.0 − 0.0) / 1.0 = 1.0
    pure blue: (1.0 − 0.0) / 1.0 = 1.0
    mid grey:  (0.5 − 0.5) / 0.5 = 0.0
    yellow:    (1.0 − 0.0) / 1.0 = 1.0

    saturation = (1.0 + 1.0 + 0.0 + 1.0) / 4 = 0.75

Grey is the only unsaturated pixel: its channels are all equal, so
the numerator is zero. That is what "distance from grey" means
arithmetically.

### Color variance: 1 number

Variance over every channel of every pixel at once (12 values here),
computed as the mean squared distance from their overall mean:

    color_variance = 0.1858

High means busy and varied; low means flat and minimal.

### Contrast: 1 number

Convert each pixel to greyscale using the same luminance weights,
then take the standard deviation across pixels:

    grey(0,0) = 0.299(1.0) + 0.587(0.0) + 0.114(0.0) = 0.299
    grey(0,1) = 0.114
    grey(1,0) = 0.500
    grey(1,1) = 0.886

    contrast = std([0.299, 0.114, 0.500, 0.886]) = 0.2865

Note how this differs from color_variance: contrast asks how much
the *lightness* swings, ignoring hue entirely. An image of red and
green at equal brightness has high color variance and near-zero
contrast.

### Warmth: 1 number

    warmth = avg_r − avg_b = 0.625 − 0.375 = 0.25

Positive means red-dominant, negative blue-dominant. The crudest
feature in the set, and empirically one of the most useful. It has
the highest d′ of any feature separating the pink-ish boards from
winter_minimal.

### Color histogram: 24 numbers

For each channel, split [0, 1] into 8 equal bins and count how many
pixels fall in each, then divide by the total so the counts become
proportions.

Red channel values are 1.0, 0.0, 0.5, 1.0. Bin 0 covers [0, 0.125),
bin 4 covers [0.5, 0.625), bin 7 covers [0.875, 1.0]:

    counts     = [1, 0, 0, 0, 1, 0, 0, 2]
    normalized = [0.25, 0, 0, 0, 0.25, 0, 0, 0.5]

Repeat for green and blue: 8 × 3 = 24 numbers.

Normalizing matters because it makes the histogram independent of
image size. A 64×64 and a 512×512 image of the same scene produce
the same proportions, but very different raw counts.

### Total

    5 statistics + 3 average channels + 24 histogram bins = 32

### What every one of these has in common

They all reduce the image by averaging or counting over all pixels
at once. Position never enters any formula. Two images with
identical color distributions and completely different spatial
arrangements produce identical feature vectors, which is exactly
the limitation the fleurs experiment in file 15 was designed to
test.


## The feature vector

This list of 32 numbers is called a **feature vector**. Think of it
as one point in a 32-dimensional space, the same idea as a point
(x, y) on a normal 2D graph, just with 32 coordinates instead of 2.
An image isn't "a picture" to the model anymore at this stage. It's
just a location in this 32-dimensional space.

## Why this matters

The entire project rests on one geometric bet: if my real "yes"
(pink board) images cluster together in one region of this 29D
space, and "no" images cluster somewhere else, then a model just
needs to find a boundary separating those two regions.

Everything in the rest of this documentation (every neuron, every
layer, every gradient) exists purely to find that boundary.