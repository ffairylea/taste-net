# 04 — Why one weighted sum alone isn't enough

So far, every neuron just computes a weighted sum — multiply inputs
by weights, add a bias. Here's why stacking multiple layers of just
that, with nothing else, doesn't actually gain anything.

## What is ReLU, quickly, before the proof below

ReLU stands for "Rectified Linear Unit" — despite the fancy name,
it's a simple function: ReLU(z) = max(0, z). If z is
positive, output z unchanged. If z is negative, output 0 instead.
That's the entire definition. It's applied to every hidden neuron's
z value, right after the weighted sum, before passing the result to
the next layer.

## The proof: stacking plain weighted sums collapses to one layer

Two layers that are just weighted sums (no ReLU), real numbers:
x=2, w1=3, w2=4.

    layer 1: z1 = 2 × 3 = 6
    layer 2: z2 = 6 × 4 = 24

Could I have gotten 24 directly from x=2 using one single weight,
skipping layer 1 entirely? Yes: 2 × 12 = 24, and 12 = w1 × w2 (3×4).
The two layers collapse into exactly one layer with one combined
weight.

This always happens with plain multiplication (and addition, with
messier algebra) — stacking gains nothing if every layer is linear.
No matter how many layers I stack, if they're all just weighted
sums, it's mathematically identical to one single layer.

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
differently. No single weight can reproduce "zero for negative
input, scaled for positive input" — that shape genuinely requires
the layered, non-linear structure.

## Where exactly does ReLU get applied — mapping the toy example back to the real network

The toy example above uses "layer 1" and "layer 2" as stand-ins for
my actual two layers: layer 1 = the hidden layer (shrunk to 1
neuron instead of 8, just to keep the arithmetic small), layer 2 =
the output layer (already just 1 neuron in real life).

ReLU is applied ONLY to the hidden layer's output (z1), right after
it's computed, before that result gets passed forward to become the
output layer's input. It is not applied after the output layer — the
output layer uses sigmoid instead, a different function with a
different job (turning the final score into a probability, not
zeroing out negatives).

So the real sequence, matching the actual network exactly:

    x → z1 = xW1+b1 → a1 = ReLU(z1) → z2 = a1×W2+b2 → ŷ = σ(z2)

In the FIRST proof above (no ReLU at all), z1 was used directly as
layer 2's input, with nothing in between — that's what let the two
layers collapse into one.

In the SECOND proof (with ReLU), ReLU was inserted right between
them: compute z1, apply ReLU to get a1, THEN use a1 (not z1) as
layer 2's input. That one inserted step is what prevented the
collapse.

ReLU sits exactly once, at the hidden layer, applied to z1 — never
applied a second time after z2.

## Why kill negatives and keep positives — what this represents

Think of each hidden neuron as a detector for one specific pattern —
say, hypothetically, one neuron ends up detecting "warm + low blue,"
roughly my pink signature. When an image strongly matches, z comes
out positive and large: "yes, I'm detecting my pattern strongly
here." When an image doesn't match — or is the opposite (cool, high
blue) — z comes out negative, and ReLU says: contribute nothing to
the final decision, don't report a negative amount.

It doesn't really make sense to ask "how much negative pink did this
neuron detect." Either the pattern is present (pass the signal on)
or it isn't (pass along silence). ReLU encodes exactly that: no such
thing as negative contribution for one specific narrow pattern.

The neuroplasticity connection: this is structurally similar to how
real neurons work — fire (send a signal, roughly proportional to how
strongly driven) past a threshold, or stay silent below it. No such
thing as a real neuron firing "backward." ReLU is a simplified
cartoon of exactly that idea. Worth being honest about where the
metaphor breaks though: real neurons integrate signals over time,
have refractory periods, use different neurotransmitters with
different effects — nothing as clean as max(0,z). ReLU borrows the
shape of all-or-nothing, threshold-based firing.