*I'm building this documentation to prove I actually understand the math, not just copy-pasted code. I worked through this with an AI tutor across multiple sessions, asking it to slow down and re-explain.


## Part 1: What is the input, actually?

An image starts as a giant grid of pixel numbers — way too many (12,000+) to learn from directly with only ~450 images. So `features.py` compresses each image down into just 29 numbers: brightness, saturation, warmth, color histogram bins, etc. Every one of these is a real, hand-computable statistic — e.g. saturation for one pixel is (max(R,G,B) - min(R,G,B)) / max(R,G,B), averaged across the whole image.

This list of 29 numbers is called a **feature vector**. Think of it as one point in a 29-dimensional space — same idea as a point (x, y) on a 2D graph, just with 29 coordinates instead of 2.

**Why this matters:** the entire project rests on one geometric bet — if my real "yes" (pink board) images cluster together in one region
of this 29D space, and "no" images cluster somewhere else, then a model just needs to find a boundary separating those two regions.
Everything below exists purely to find that boundary.

## Part 2: What does one neuron actually compute?

A neuron takes in numbers and produces one number out, using a **weighted sum** (w). Worked with real numbers first:

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

**The 8 Neurons**
Why 8? 
8 is an arbitrary choice, not derived from some formula. I picked a reasonable-sized number for a small dataset. There's no rule saying "8 is correct" — I could try 4, 16, 32 and get different results (this is actually a great, easy experiment for my project: does accuracy change if I try different n_hidden values?). More neurons = more capacity to detect complex patterns, but also more risk of overfitting with limited data (~450 images). 8 is a reasonable middle-ground guess for this data size, not a mathematically proven optimum.

What does eaach neuron do?
Every single hidden neuron sees all 29 input features. What actually differs between the 8 neurons is their weights — each neuron has its own separate set of 29 weights (one weight per feature) plus its own bias. So neuron 1 might have learned weights that heavily emphasize blue-histogram bins and barely care about brightness; neuron 2 might have learned the opposite emphasis. Same 29 inputs going in everywhere, but 8 different "opinions" coming out, because each neuron combines those same 29 numbers differently.
Concretely, in my actual code — look at W1 in model.py:
pythonself.W1 = rng.normal(0, 0.5, size=(n_features, n_hidden))
This shape is (29, 8) — 29 rows (one per feature), 8 columns (one per hidden neuron). Column 1 = neuron 1's personal set of 29 weights. Column 2 = neuron 2's completely separate set of 29 weights. Same input matrix multiplies against all 8 columns simultaneously (that's what the matrix multiplication X @ W1 does) — producing 8 different z1​ values, one per neuron, from the exact same 29 inputs.

Why is this useful, rather than each neuron only seeing a slice of features? 
Because interesting patterns often require combinations across many features at once — e.g., "warm and low-blue-bin-6 and not-too-saturated" might be the real pink signature, and a single neuron needs access to all three of those simultaneously to detect that specific combination. If neurons only saw isolated slices, they could never learn cross-feature relationships like that.


## Part 2.5: Why do we even add a bias?

Without b, z = xW forces z to be exactly 0 whenever every input is 0 (e.g. an imaginary all-black, featureless image) — no matter what the weights are. The model would have zero flexibility to say "even for a
neutral input, I still lean toward yes/no."

Adding a bias, e.g. b = 2, means even with all-zero inputs, z = 2 — letting the model encode a baseline tendency independent of the specific input. Concretely: if the vast majority of my real pins skew warm-toned in general, a learned bias might end up positive, encoding "lean toward yes by default, only overturn that if features strongly say otherwise."

Geometrically: the model's job is finding a
boundary line separating yes-points from no-points in feature space.
Without bias, that boundary is forced to pass through the exact
origin (the all-zeros point) — an arbitrary, unhelpful restriction.
Bias lets the boundary sit wherever the real data actually needs it.
Weights control the boundary's angle/tilt; bias controls its position.

## Part 3: Why one weighted sum alone isn't enough

Stack two layers that are JUST weighted sums (no ReLU), with real
numbers: x=2, w1=3, w2=4.

    layer 1: z1 = 2 × 3 = 6
    layer 2: z2 = 6 × 4 = 24

Could I have gotten 24 directly from x=2 using one single weight,
skipping layer 1 entirely? Yes: 2 × 12 = 24, and 12 = w1 × w2 (3×4).
The two layers collapse into exactly one layer with one combined
weight. 

**This always happens with plain multiplication (and addition, with
messier algebra) — stacking gains nothing if every layer is linear.**
No matter how many layers I stack, if they're all just weighted sums,
it's mathematically identical to one single layer.

## What is ReLU?

ReLU stands for "Rectified Linear Unit" — 
it's the simplest possible function: ReLU(z) = max(0, z). If z is
positive, output z unchanged. If z is negative, output 0 instead.
It's applied to every hidden
neuron's z value, right after the weighted sum, before passing the
result to the next layer.

## What ReLU actually breaks, with real numbers

Same setup, but apply ReLU between layers. x=-4, w1=3:

    z1 = -4 × 3 = -12
    a1 = ReLU(-12) = max(0, -12) = 0
    z2 (using a1, w2=5) = 0 × 5 = 0

Now flip x to +4, same weights:

    z1 = 4 × 3 = 12
    a1 = ReLU(12) = 12  (positive, passes through unchanged)
    z2 = 12 × 5 = 60

Flipping the sign of x took the output from 0 to 60 — not a simple
proportional relationship. That asymmetry (one side gets crushed to
zero, the other doesn't) is exactly what makes this non-linear, and
it's specifically because ReLU treats positive and negative
differently. No single weight can reproduce "zero for negative input,
scaled for positive input" — that shape genuinely requires the
layered, non-linear structure.

## Why kill negatives and keep positives — what this represents

Think of each hidden neuron as a **detector for one specific pattern**
— say, hypothetically, one neuron ends up detecting "warm + low blue,"
roughly my pink signature. When an image strongly matches, z comes
out positive and large: "yes, I'm detecting my pattern strongly here."
When an image doesn't match — or is the opposite (cool, high blue) —
z comes out negative, and ReLU says: contribute *nothing* to the final
decision, don't report a negative amount.

It doesn't really make sense to ask "how much *negative* pink did this
neuron detect" — either the pattern is present (pass the signal on) or
it isn't (pass along silence). ReLU encodes exactly that: no such
thing as negative contribution for one specific narrow pattern.

**The neuroplasticity connection:** 
this is structurally similar to how real neurons work — fire (send a
signal, roughly proportional to how strongly driven) past a
threshold, or stay silent below it. No such thing as a real neuron
firing "backward." ReLU is a simplified cartoon of exactly that idea.
Worth being honest about where the metaphor breaks though: real
neurons integrate signals over time, have refractory periods, use
different neurotransmitters with different effects — nothing as
clean as max(0,z). ReLU borrows the *shape* of all-or-nothing,
threshold-based firing, without claiming to simulate actual biology.

## Part 4: Deriving sigmoid's own derivative

## What is sigmoid?
it's formula: σ(z) = 1 / (1 + e^-z)

It takes any real number (positive, negative, huge, tiny) and
squashes it into a value strictly between 0 and 1 — so it can be
read as a probability. We use it **only** at the very last layer, after
combining all the hidden neurons together into one final z2, because
that's the one place we need an actual "how confident" number, not
just an internal detector value. It came from inverting the log-odds
formula.

We've been using σ'(z) = σ(z)(1-σ(z)) as a fact. Here's where it
actually comes from since we covered derivatives in highschool :).

σ(z) = 1/(1+e^-z), rewritten as (1+e^-z)^-1 so chain rule + power
rule apply.

Outer function: u^-1, derivative -u^-2. Inner: u = 1+e^-z, derivative
-e^-z. Multiply:

    σ'(z) = -u^-2 × (-e^-z) = e^-z / (1+e^-z)^2

Rewrite in terms of σ(z) itself: split into two fractions,

    e^-z / (1+e^-z)^2 = [1/(1+e^-z)] × [e^-z/(1+e^-z)]

First fraction is σ(z). Second fraction, add/subtract 1 in numerator:

    e^-z/(1+e^-z) = [(1+e^-z) - 1]/(1+e^-z) = 1 - σ(z)

So: σ'(z) = σ(z) × (1-σ(z))

Checked with real numbers: σ(0.272) ≈ 0.568, so
σ'(0.272) = 0.568 × 0.432 ≈ 0.2454

This is maximized (0.25) when σ(z)=0.5 (z=0) — sigmoid's steepest
point — and shrinks toward 0 as σ(z) approaches 0 or 1 (confident
predictions). This formula is the literal mathematical reason sigmoid
"goes flat" at extremes, not just a visual description.

## Why we even need this derivative — the actual reason, not just curiosity

We want ∂L/∂z2 (how loss changes if z2 changes) to know how to correct
the weights that produced z2. But L doesn't touch z2 directly — only
through ŷ (since ŷ = σ(z2)). Chain rule:

    ∂L/∂z2 = (∂L/∂ŷ) × (∂ŷ/∂z2)

That second piece IS sigmoid's derivative — since ŷ = σ(z2), "how ŷ
changes as z2 changes" is exactly asking for σ'(z2). This isn't
curiosity about probability — it's a required link letting the chain
rule hop backward past the sigmoid step, converting "loss's
sensitivity to the prediction" into "loss's sensitivity to the raw
score."

(Side note: for THIS specific sigmoid+cross-entropy pairing, this
derivative ends up canceling away in the combined formula — see the
∂L/∂z2 = ŷ-y result. But it's still needed standalone to verify that
cancellation is real, and it's necessary for layer 1's ReLU step,
which doesn't get this convenient cancellation.)


## Part 5: The full cancellation proof, verified with real numbers

Piece 1 — cross-entropy's derivative w.r.t. ŷ:

    ∂L/∂ŷ = -[y/ŷ - (1-y)/(1-ŷ)]

Plug in y=1, ŷ=0.568:

    ∂L/∂ŷ = -[1/0.568 - 0/0.432] = -1.7606

Piece 2 — sigmoid's derivative (derived in Part 4):

    ∂ŷ/∂z2 = σ(z2)(1-σ(z2)) = 0.568 × 0.432 = 0.2454

Multiply (chain rule):

    ∂L/∂z2 = (-1.7606) × (0.2454) ≈ -0.432

Compare to the "shortcut" formula ŷ - y = 0.568 - 1 = -0.432.

They match exactly — not a coincidence, this is the actual algebraic
cancellation happening, verified numerically rather than just
asserted. The messy 1/ŷ and ŷ(1-ŷ) terms combine and simplify to a
plain subtraction. This is why `model.py`'s backward() function can
just write `dZ2 = A2 - y_true` — that one line already has this
entire cancellation baked in, and I now know exactly why it's allowed
to be that short.