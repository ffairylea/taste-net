# 09 — Implementation details: the practical layer on top of the math

Everything in files 01-08 is the pure math. This file covers a few
things in model.py that are necessary for real, working code but
aren't new mathematical concepts — just practical engineering on
top of the math, explained in the order they appear in the file.

## import numpy as np

Loads numpy — the library providing arrays, matrix multiplication
(the @ operator), and math functions (np.exp, np.log, etc.) used
throughout the rest of the file.

## sigmoid(z)

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

np.clip(z, -500, 500) sets a ceiling and a floor: if a number in z is
bigger than 500, it gets forced down to exactly 500. If smaller than
-500, forced up to exactly -500. If already between the two, nothing
changes.

Why 500 specifically, and why doesn't capping mess up the math?
Sigmoid at z=500 is already so close to 1 (something like
0.999999...) that whether the true value would've been 500 or 5000,
sigmoid's actual output is indistinguishable either way — already
maxed out for practical purposes. Capping here doesn't change the
real answer; it just avoids Python's floating-point arithmetic
breaking (computing e^-5000 can cause a computational error) before
it would've rounded to basically the same result anyway.

The actual formula after that, 1 / (1 + np.exp(-z)), is exactly
sigmoid as derived in file 05.

## relu_derivative(z)

    def relu_derivative(z):
        return (z > 0).astype(float)

z > 0 produces an array of True/False values — one per number in z,
asking "is this number bigger than 0?" .astype(float) converts those
True/False values into actual numbers: True becomes 1.0, False
becomes 0.0.

This matters because ReLU's derivative needs to literally be the
number 1 or 0, to multiply against other numbers later in backprop.
True/False alone can't be multiplied the same way numbers can in
every context, so this line converts the yes/no answer into the
actual 1/0 number ReLU's derivative is supposed to produce.

## __init__ — why seed=42?

    rng = np.random.default_rng(seed)

This makes the "random" initial weights reproducible — the same
seed always generates the exact same sequence of random numbers.
Without a fixed seed, every training run would start from different
random weights, making it impossible to compare runs fairly or debug
consistently. If something looks wrong, rerunning with the same seed
reproduces the exact same starting point to investigate.

## What rng.normal(0, 0.5, size=(n_features, n_hidden)) actually means, piece by piece

    self.W1 = rng.normal(0, 0.5, size=(n_features, n_hidden))

- rng = a random number generator object (created above) — a machine
  that produces random numbers whenever asked.
- .normal(...) = specifically requests numbers following a normal
  distribution (the classic bell-curve shape) — most generated
  numbers cluster near the center, fewer appear far from it,
  symmetric in both directions.
- 0 = the center of that bell curve (the mean) — most random numbers
  will land near 0.
- 0.5 = how spread out the bell curve is (the standard deviation) —
  a small number like 0.5 means most generated values land close to
  0 (roughly between -1 and 1), not wildly far away.
- size=(n_features, n_hidden) = how many random numbers to generate
  and how to shape them — not just one number, but a whole grid with
  n_features rows and n_hidden columns (29 rows, 8 columns in the
  real network) — one random number for every combination of "one
  feature, one hidden neuron."

So this one line generates 29×8 = 232 individual small random
numbers at once, arranged into the exact grid shape W1 needs.

## Why small random weights, not zeros or huge values?

If weights started at exactly zero, every hidden neuron would
compute the identical value and learn the identical thing during
training — wasting 7 of the 8 neurons on redundant copies of the
same detector. Small random noise breaks this symmetry so each
neuron starts slightly different and can specialize differently.

Why small specifically, not large random values? If initial weights
were huge, z values would start extremely large, pushing sigmoid
into its flat zone (see file 05's vanishing gradient discussion) —
training would start crippled before it even begins.

## np.zeros(n_hidden) — why bias starts at exactly zero, when weights don't

    self.b1 = np.zeros(n_hidden)

np.zeros(n_hidden) creates an array containing exactly n_hidden
zeros — if n_hidden=8, this produces [0, 0, 0, 0, 0, 0, 0, 0], one
zero per hidden neuron's starting bias. Same idea as rng.normal
generating a grid of random numbers, except this fills the array
with zeros instead.

Bias doesn't have the symmetry problem weights have — since the
weights are already different per neuron (breaking symmetry), every
neuron can safely share the same starting bias (0) without causing
the "all neurons learn the same thing" issue.

## cache = (X, Z1, A1, Z2, A2) — what the parentheses mean

This is a tuple — Python's way of bundling several separate values
into one container, so they can be passed around together as a
single thing instead of as 5 separate variables.

A tuple is similar to an array/list in that it holds multiple things
in order, but the difference: arrays/lists are usually meant for
many items of the SAME kind (like the 29 features, all just
numbers), while this tuple groups 5 DIFFERENT things together (the
input, two raw scores, two activated outputs) purely so backward()
can receive all of them at once as one bundle, since it needs every
one of them to compute gradients.

## The eps trick in compute_loss

    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)

Cross-entropy uses log(ŷ). If ŷ ever became exactly 0 or 1 (possible
in floating-point, especially with very confident predictions),
log(0) is undefined and would break training. This nudges ŷ a tiny
amount away from the exact edges (0.00000001 instead of 0) — purely
a safeguard, doesn't change the actual math, just prevents a crash.

## Why divide by n in the gradients?

    n = X.shape[0]
    dW2 = A1.T @ dZ2 / n

Every hand-calculation in file 07 was for ONE image. Real training
processes an entire batch of images at once via matrix multiplication
(file 08). Dividing by n averages the gradient across every image in
the batch — so one unusually weird image doesn't wildly overcorrect
the weights. The update reflects the typical correction needed
across the whole batch, not any single image's quirk.