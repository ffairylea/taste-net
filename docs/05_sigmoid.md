# 05 — Sigmoid: turning a raw score into a probability

By the end of the forward pass so far (files 03-04), the network
produces one final raw score, z2 — the output layer's weighted
combination of all 8 hidden neurons' opinions. But z2 can be any
real number: -50, 3.7, 1000, anything, depending on the weights.
That's not usable as a prediction yet. We need an actual probability
— a number strictly between 0 and 1, meaning "how confident is the
model this is a yes." Sigmoid is the function that does that
conversion.

## What is sigmoid?

    σ(z) = 1 / (1 + e^-z)

It takes any real number and squashes it into a value strictly
between 0 and 1. We use it only at the very last layer, after
combining all the hidden neurons together into one final z2, because
that's the one place we need an actual "how confident" number, not
just an internal detector value.

## Where this formula actually comes from (log-odds)

Start with what's needed: a way to represent a probability p (between
0 and 1) using a raw score that can be any real number — since
z = xW+b can come out as anything, positive or negative, huge or tiny.

Take the "odds" of p: odds = p / (1-p). This ranges from 0
(impossible) to infinity (certain) — better than plain probability,
but still can't go negative.

Take the natural log of the odds — called "log-odds" or "logit":

    z = log(p / (1-p))

Now z can be any real number: as p approaches 0, log-odds goes to
negative infinity; as p approaches 1, log-odds goes to positive
infinity. This finally matches what z = xW+b can naturally produce.

Sigmoid is just this equation solved backwards — given z, get back p:

    z = log(p/(1-p))
    e^z = p/(1-p)
    e^z (1-p) = p
    e^z = p + p·e^z
    e^z = p(1+e^z)
    p = e^z / (1+e^z)

Multiply top and bottom by e^-z:

    p = 1 / (e^-z + 1) = 1 / (1+e^-z)

That's sigmoid. It's an S-shaped curve, but more importantly it's the exact inverse of the log-odds function, which is the natural
bridge between "any real number" (what linear math produces) and "a
valid probability" (what's actually needed).

## Sigmoid's own derivative

Later (file 07), the backward pass needs to know how sigmoid's
output changes as its input changes — sigmoid's own derivative.

Deriving it here, since it belongs with the function itself.

Rewrite σ(z) = (1+e^-z)^-1, so chain rule + power rule apply.

Outer function: u^-1, derivative -u^-2. Inner: u = 1+e^-z, derivative
-e^-z. Multiply:

    σ'(z) = -u^-2 × (-e^-z) = e^-z / (1+e^-z)^2

Rewrite in terms of σ(z) itself — split into two fractions:

    e^-z / (1+e^-z)^2 = [1/(1+e^-z)] × [e^-z/(1+e^-z)]

First fraction is σ(z). Second fraction, add/subtract 1 in numerator:

    e^-z/(1+e^-z) = [(1+e^-z) - 1]/(1+e^-z) = 1 - σ(z)

So:

    σ'(z) = σ(z) × (1-σ(z))

This is maximized (0.25) when σ(z)=0.5 (z=0) — sigmoid's steepest
point — and shrinks toward 0 as σ(z) approaches 0 or 1 (confident
predictions). This formula is the literal mathematical reason
sigmoid "goes flat" at extremes, not just a visual description.

## Why we'll need this derivative later — the actual reason

In the backward pass (file 07), we need to know how loss changes if
z2 changes — because that tells us how to correct the weights that
produced z2 are. But loss doesn't touch z2 directly, only through ŷ
("y-hat" — the model's predicted probability, its guess, as opposed
to y, the true label, a known fact) since ŷ = σ(z2). By chain rule,
this requires knowing how ŷ changes as z2 changes — which is exactly
σ'(z2), the formula just derived above. It's a required link letting
the correction signal hop backward past the sigmoid step.

## Why does it matter that sigmoid's derivative goes flat at extremes?

This connects to two different things and they both matter in different ways.

### Reason 1 — it explains why sigmoid can be a bad choice for hidden layers (vanishing gradients)

Recall: σ'(z) = σ(z)(1-σ(z)). Plug in extreme values to see this
concretely. If σ(z) = 0.99 (very confident), σ'(z) = 0.99 × 0.01 =
0.0099 — small. If σ(z) = 0.999, σ'(z) = 0.999 × 0.001 = 0.000999 —
even smaller. As predictions get more extreme (closer to 0 or 1),
this derivative shrinks toward zero.

Why does a small derivative matter? Backprop's chain rule works by
multiplying derivatives together across layers, to trace how an
early weight affects the final loss. Imagine a network with several
sigmoid layers stacked (not my network, but a deeper one, for
illustration). If even one of those layers happens to land in this
flat zone, its near-zero derivative gets multiplied into the whole
chain — and multiplying by something close to zero crushes the
entire result close to zero too, no matter how healthy the other
layers' derivatives are.

Concretely: say 3 sigmoid layers have derivatives of 0.2, 0.2, and
0.0099 (one landed in the flat zone). Multiply them:

    0.2 × 0.2 × 0.0099 ≈ 0.0004

That tiny number is what actually gets used to update an early
weight: W ← W - lr × gradient. If the gradient is 0.0004, the weight
barely moves at all, no matter how many training rounds happen — the
network effectively stops learning at that point in the chain. This
is called the "vanishing gradient" problem, and it's a well-known,
real historical issue that early deep networks ran into when they
used sigmoid for every layer.

This is specifically why my network uses ReLU for the hidden layer
instead of sigmoid. ReLU's derivative is always exactly 1 (if the
neuron fired) or exactly 0 (if it didn't) — never some small
in-between value that quietly strangles the gradient as it passes
through. Multiplying by 1 repeatedly doesn't shrink anything; ReLU
either passes a gradient through completely intact or blocks it
entirely, with no slow decay building up across layers.

### Why does my network only have 2 layers, when others have way more?

My network is deliberately small — 2 layers is enough for ~450
images and 29 hand-built features; more layers would likely just
overfit given how little data I have. Real production networks
(the ones with dozens or hundreds of layers) need extra tricks
specifically to survive the vanishing gradient problem at that
depth, since even ReLU alone doesn't fully solve every issue that
comes with extreme depth. A few of the real techniques (not used
in my network, just for context): "residual connections" (letting
the gradient skip around layers via shortcut paths, so it has an
alternate route besides multiplying through every single one),
"batch normalization" (rescaling values between layers so they
don't drift into extreme, flat-derivative territory to begin with),
and simply defaulting to ReLU (or variants of it) almost everywhere
instead of sigmoid. My 2-layer network is small enough that none of
this machinery is necessary — the problem these techniques solve
mostly shows up at much greater depth than what I'm working with.

### Reason 2 — what this means for my output layer specifically, where I do use sigmoid

My output layer uses sigmoid (correct choice there — it's only one
layer deep, not stacked with other sigmoid layers, so the vanishing
gradient risk described above is much smaller in this specific
position).

But the flatness still matters here in a different, more direct way:
if the model becomes extremely confident (very close to 0 or 1) on a
prediction, sigmoid's derivative at that point is tiny — meaning, in
principle, very little correction signal would flow back from that
example, even if the model is wrong.

Concretely: imagine the model is 99.9% confident an image is "yes,"
but it's actually "no." Cross-entropy loss (L = -[y·log(ŷ) +
(1-y)·log(1-ŷ)], which punishes confident wrong answers especially
hard because log(x) shoots toward negative infinity as x approaches
0 — full explanation in file 06) is huge here. Normally, you'd expect
a huge loss to produce a strong correction. But sigmoid's derivative
at that extreme point is nearly zero — which sounds like it should
weaken the correction signal exactly when it's needed most.

### How does a derivative's size actually relate to "how much to correct"?

This is worth making explicit, since it's easy to nod past. The gradient (the derivative we compute) isn't just a direction (positive
or negative) — its SIZE directly controls how big the correction is.
Recall the update rule: W ← W - lr × gradient. A gradient of -0.4
produces a bigger weight change than a gradient of -0.004, given the
same learning rate. So if sigmoid's derivative shrinks toward zero,
and that derivative is a multiplied piece of the final gradient, the
whole correction shrinks too — literally "the model computes that the
needed adjustment is tiny, so it only nudges the weight a tiny bit,"
even in a case where intuitively (looking at the huge loss) a bigger
correction feels warranted. That mismatch — huge loss, but a
shrunken correction — is exactly the flaw this next part shows how to
avoid.

### The algebraic interaction — why cross-entropy specifically fixes this

This is where the cancellation proof from earlier becomes important. Because I specifically pair sigmoid with
cross-entropy loss (not some other loss function), their derivatives
combine and cancel down to a plain subtraction, ŷ - y — with no
leftover σ'(z) term at all. The "flat derivative" problem that would
otherwise weaken the correction on confident wrong predictions simply
doesn't appear in the final formula, because it canceled away during
the derivation.

Concretely, why is cross-entropy "good" here specifically: its own
derivative (∂L/∂ŷ) contains a 1/[ŷ(1-ŷ)]-shaped term buried inside it
— and this happens to be the exact reciprocal of what sigmoid's
derivative contributes. When multiplied together (chain rule), the
ŷ(1-ŷ) from sigmoid's derivative and the matching term from
cross-entropy's derivative cancel out algebraically, leaving just
ŷ - y. This isn't true for every loss function paired with sigmoid —
it's specifically true for this pairing, which is exactly why
sigmoid + cross-entropy is the standard, deliberate combination for
yes/no classification, rather than an arbitrary convention. The full
worked-out algebra, with real numbers, showing this cancellation step
by step, is in file 07.