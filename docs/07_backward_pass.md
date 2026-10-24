# 07: The backward pass: how the model corrects itself

By the end of the forward pass (files 03-06), we have a prediction
(ŷ) and a loss (L) measuring how wrong it was. But the network
doesn't automatically know which weights caused that wrongness, or
by how much, or in which direction to adjust them. That's the entire
purpose of the backward pass: mathematically trace, weight by
weight, "if I nudge this specific number up or down slightly, does
loss get better or worse, and by how much", so the model knows
exactly how to correct itself.

## The running example, recapped

    x = [0.6, 0.2], W1 = [0.5, -0.3], b1 = 0.1, W2 = 0.8, b2 = 0
    z1 = 0.34, a1 = 0.34, z2 = 0.272, ŷ ≈ 0.568, y = 1

## Step 1: the cancellation result

Full derivation of this in file 05 (sigmoid's derivative) and here,
worked through both the long way and the shortcut way:

Piece 1: cross-entropy's (the loss function I'm using) derivative
w.r.t. ŷ:

    ∂L/∂ŷ = -[y/ŷ - (1-y)/(1-ŷ)]

Plug in y=1, ŷ=0.568:

    ∂L/∂ŷ = -[1/0.568 - 0/0.432] = -1.7606

Piece 2: sigmoid's derivative (derived in file 05):

    ∂ŷ/∂z2 = σ(z2)(1-σ(z2)) = 0.568 × 0.432 = 0.2454

Multiply (chain rule):

    ∂L/∂z2 = (-1.7606) × (0.2454) ≈ -0.432

Compare to the shortcut formula:

    ŷ - y = 0.568 - 1 = -0.432

They match exactly. This is the actual algebraic cancellation
happening, verified numerically. The 1/ŷ and ŷ(1-ŷ) terms
combine and simplify to a plain subtraction. This is why
`model.py`'s backward() function can just write `dZ2 = A2 - y_true`:
that one line already has this entire cancellation baked in.

What this number means: -0.432 says "z2 should have been higher;
nudge it up to reduce loss." Makes sense: true label was y=1, model
only predicted 56.8% confidence, so it should've scored this higher.

## Step 2: gradient for W2

    ∂L/∂W2 = a1 × ∂L/∂z2 = 0.34 × (-0.432) = -0.14688

Why multiply by a1 specifically? W2's whole job was scaling a1 up
before it became z2. If a1 had been 0 (that neuron never fired),
W2 couldn't have contributed anything to the error: multiplying by
0 correctly gives "W2 gets zero blame, it had nothing to work with."
Here a1 was a decent 0.34, so W2 gets a real, non-zero share of the
correction.

## Step 3: push the error backward into the hidden layer

    ∂L/∂a1 = ∂L/∂z2 × W2 = (-0.432)(0.8) = -0.3456

Then through ReLU's derivative. Since z1 = 0.34 > 0, this neuron
fired, so ReLU'(z1) = 1, and the gradient passes through completely
unchanged:

    ∂L/∂z1 = -0.3456 × 1 = -0.3456

(If z1 had been negative instead, ReLU'(z1) = 0, and this whole
gradient would become exactly 0: the neuron gets zero blame, zero
learning signal, since it contributed nothing to begin with.)

## Step 4: gradient for W1

    ∂L/∂W1 = x × ∂L/∂z1 = [0.6, 0.2] × (-0.3456) = [-0.2074, -0.0691]

Two numbers, one per input feature, since W1 has one weight per
feature feeding into this neuron. Same "what fed in × how wrong"
pattern as W2's gradient in Step 2, just one layer earlier.

## Step 5: actually updating the weights (gradient descent)

Rule: W ← W - lr × gradient. Using learning rate 0.1:

    W2_new = 0.8 - (0.1)(-0.14688) = 0.81469
    W1_new = [0.5, -0.3] - (0.1)[-0.2074, -0.0691] = [0.5207, -0.2931]

Sanity check: W2 went UP slightly (0.8 → 0.8147). True label was
"yes" and the model wasn't confident enough (only 56.8%), so
increasing W2 pushes future predictions on similar inputs higher,
exactly the right direction. Both W1 values also shifted in the
direction that would make z1 (and downstream, a1, z2, ŷ) larger next
time: the whole system consistently correcting itself toward the
true answer, in one small step.

## What this one example actually represents

Everything above is a single training step, for a single image,
computed entirely by hand. My actual train.py does this exact same
sequence of calculations, automatically, for every image in the
training set, repeated 500 times (epochs). Nothing about the real
training loop is conceptually different: same forward pass, same
loss calculation, same backward pass, same weight update, just
scaled up from 1 neuron and 2 toy features to 8 neurons and 32 real
features, and repeated many times instead of once.

The loss curve I saw when running train.py (starting around 0.86,
ending near 0.21) is literally thousands of repetitions of the
single correction shown above, each one nudging the weights a tiny
bit closer to correctly separating my real "yes" and "no" images.