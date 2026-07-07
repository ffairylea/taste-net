*I'm building this documentation to prove I actually understand the
math, not just copy-pasted code. I worked through this with an AI
tutor across multiple sessions, asking it to slow down and re-explain
with real numbers whenever a symbol-based explanation didn't click.
The confusion and mistakes are documented honestly in LEARNING_LOG.md.

## Part 1: What is the input, actually?

An image starts as a giant grid of pixel numbers — way too many
(12,000+) to learn from directly with only ~450 images. So
`features.py` compresses each image down into just 29 numbers:
brightness, saturation, warmth, color histogram bins, etc. Every one
of these is a real, hand-computable statistic — e.g. saturation for
one pixel is (max(R,G,B) - min(R,G,B)) / max(R,G,B), averaged across
the whole image.

This list of 29 numbers is called a **feature vector**. Think of it
as one point in a 29-dimensional space — same idea as a point (x, y)
on a 2D graph, just with 29 coordinates instead of 2.

**Why this matters:** the entire project rests on one geometric bet —
if my real "yes" (pink board) images cluster together in one region
of this 29D space, and "no" images cluster somewhere else, then a
model just needs to find a boundary separating those two regions.
Everything below exists purely to find that boundary.

## Part 2: What does one neuron actually compute?

A neuron takes in numbers and produces one number out, using a
**weighted sum**. Worked with real numbers first:

Say for one photo: warmth = 0.6, saturation = 0.2 (simplified to just
2 features here, instead of the real 29, to keep this readable).

I decide (well — the model eventually learns, but imagine choosing):
warmth matters a lot and pushes toward "yes" → weight it 0.5.
Saturation matters less and actually pushes toward "no" → weight it
-0.3. I also have some fixed baseline lean, +0.1, regardless of the
photo.

    (0.6 × 0.5) + (0.2 × -0.3) + 0.1 = 0.3 - 0.06 + 0.1 = 0.34

That's the entire computation. 0.34 is the neuron's raw score.

**Naming the pieces**, purely for shorthand (not new concepts):
- The inputs (0.6, 0.2) = **x**
- The multipliers (0.5, -0.3) = **W** (weights)
- The fixed add-on (0.1) = **b** (bias)
- The result (0.34) = **z**

So z = xW + b is just "multiply each input by its weight, add them
up, add the bias" — written with letters so the same recipe can be
reused for all 29 real features, 8 neurons, ~450 images, and 500
rounds of training without rewriting the arithmetic every time.

**Where do W and b actually come from?** Not chosen by me — they
start as small random numbers, then get automatically adjusted by
training (gradient descent) based on what reduces error on my real
labeled images.

## Part 2.5: Why do we even add a bias?

Without b, z = xW forces z to be exactly 0 whenever every input is 0
(e.g. an imaginary all-black, featureless image) — no matter what the
weights are. The model would have zero flexibility to say "even for a
neutral input, I still lean toward yes/no."

Adding a bias, e.g. b = 2, means even with all-zero inputs, z = 2 —
letting the model encode a baseline tendency independent of the
specific input. Concretely: if the vast majority of my real pins skew
warm-toned in general, a learned bias might end up positive, encoding
"lean toward yes by default, only overturn that if features strongly
say otherwise."

Geometrically (tying back to Part 1): the model's job is finding a
boundary line separating yes-points from no-points in feature space.
Without bias, that boundary is forced to pass through the exact
origin (the all-zeros point) — an arbitrary, unhelpful restriction.
Bias lets the boundary sit wherever the real data actually needs it.
Weights control the boundary's angle/tilt; bias controls its position.

## Part 3 onward: coming next
- Why one weighted sum isn't enough (the linear-collapses-to-linear
  proof, and what non-linearity/ReLU actually buys geometrically)
- The full forward pass, worked by hand
- The chain rule, backward pass, and why sigmoid + cross-entropy pair
  up mathematically