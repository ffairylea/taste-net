# 03 — What does one neuron actually compute?

A neuron takes in numbers and produces one number out, using a
**weighted sum**. 

Say for one photo: warmth = 0.6, saturation = 0.2 (simplified to
just 2 features here, instead of the real 29, to keep this
readable).

I decide (well — the model eventually learns this, but imagine
choosing it manually first): warmth matters a lot and pushes toward
"yes" → weight it 0.5. Saturation matters less and actually pushes
toward "no" → weight it -0.3. There's also some fixed baseline lean,
+0.1, regardless of the photo.

    (0.6 × 0.5) + (0.2 × -0.3) + 0.1 = 0.3 - 0.06 + 0.1 = 0.34

That's the entire computation. 0.34 is the neuron's raw score.

## Naming the pieces

- The inputs (0.6, 0.2) = **x**
- The multipliers (0.5, -0.3) = **W** (weights)
- The fixed add-on (0.1) = **b** (bias)
- The result (0.34) = **z**

So z = xW + b is just "multiply each input by its weight, add them
up, add the bias" — written with letters so the same recipe can be
reused for all 29 real features, 8 neurons, ~450 images, and 500
rounds of training, without rewriting the arithmetic every time.

Where do W and b actually come from? Not chosen by me — they start
as small random numbers, then get automatically adjusted by training
(gradient descent) based on what reduces error on my real labeled
images.

## Why do we even add a bias?

Without b, z = xW forces z to be exactly 0 whenever every input is 0
(e.g. an imaginary all-black, featureless image) — no matter what
the weights are. The model would have zero flexibility to say "even
for a neutral input, I still lean toward yes/no."

Adding a bias, e.g. b = 2, means even with all-zero inputs, z = 2 —
letting the model encode a baseline tendency independent of the
specific input. Concretely: if the vast majority of my real pins
skew warm-toned in general, a learned bias might end up positive,
encoding "lean toward yes by default, only overturn that if features
strongly say otherwise."

Geometrically: the model's job is finding a boundary line separating
yes-points from no-points in feature space (the abstract 29-dimensional space where each image is one point, based on its 29 feature values (brightness, saturation, etc.) — same idea as a normal graph with x & y axes, just with 29 axes instead of 2.). Without bias, that
boundary is forced to pass through the exact origin (the all-zeros
point) —an arbitrary, unhelpful restriction. Bias lets the boundary
sit wherever the real data actually needs it. Weights control the
boundary's angle/tilt; bias controls its position.

## Why 8 neurons specifically?

(picked 8 as a reasonable middle-ground guess for ~450 images — not
derived from a formula, worth testing other values later)

More neurons = more capacity to detect complex patterns, but also
more risk of overfitting with limited data. There's no rule saying
8 is correct — trying 4, 16, or 32 and comparing results is an experiment worth running later.

Every single hidden neuron sees all 29 input features. What differs between the 8 neurons is their weights: each has
its own separate set of 29 weights plus its own bias. So neuron 1
might learn weights that heavily emphasize blue-histogram bins and
barely care about brightness; neuron 2 might learn the opposite
emphasis. Same 29 inputs everywhere, but 8 different "opinions,"
because each neuron combines those same 29 numbers differently.

Concretely, in `model.py`:

    self.W1 = rng.normal(0, 0.5, size=(n_features, n_hidden))

This shape is (29, 8) — 29 rows (one per feature), 8 columns (one
per hidden neuron). Column 1 = neuron 1's personal 29 weights.
Column 2 = neuron 2's completely separate 29 weights.

Why is this useful, rather than each neuron only seeing a slice of
features? Because interesting patterns often require combinations
across many features at once — e.g. "warm and low-blue-bin-6 and
not-too-saturated" might be the real pink signature, and a single
neuron needs access to all three simultaneously to detect that
specific combination. If neurons only saw isolated slices, they
could never learn cross-feature relationships like that.

## What does the output neuron actually do?

The 8 hidden neurons each detect their own pattern — like 8 separate specialists, each with their own opinion about the
image (though they're not assigned these roles; they emerge from
training). Each one outputs a single number saying roughly "how
strongly do I detect my pattern here."

8 separate opinions isn't a decision yet, though — there needs to be
one final answer, not 8 different numbers. The output neuron's
entire job is to combine those 8 opinions into one final verdict.

It's not a plain average. A plain average would treat every hidden
neuron's opinion as equally important. Instead, the output neuron
learns its own weight for each of the 8 hidden neurons, so it can
say "hidden neuron 3's opinion matters a lot to my final decision,
but hidden neuron 7 barely matters at all":

    z2 = (a1_neuron1 × w1) + (a1_neuron2 × w2) + ... + (a1_neuron8 × w8) + b2

Same weighted-sum recipe as before — except now the "inputs" aren't
the original 29 features anymore, they're the 8 hidden neurons'
outputs. The output neuron never sees the original image directly —
it only sees how strongly each of its 8 specialists reacted, and
combines those reactions (weighting the more useful specialists more
heavily) into one final raw score, which then goes through sigmoid
to become the actual predicted probability.

Concretely: if hidden neuron 3 turns out, through training, to be
really good at detecting my pink signature, while neuron 7 ends up
detecting mostly noise, the output neuron would learn a large weight
for neuron 3 and a near-zero weight for neuron 7 — effectively
"listen closely to specialist 3, mostly ignore specialist 7" —
without ever being told which neuron was actually useful. That
assignment of importance is discovered by gradient descent during
training, not designed by me.