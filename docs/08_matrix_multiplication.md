# 08 — Matrix multiplication: how this scales to real data

Everything in files 03-07 was written for one image, computed by
hand, one number at a time. Real training needs this exact same
computation repeated for 8 neurons and ~360 images (one training
batch) at once. Matrix multiplication is just this same
multiply-and-add recipe, done for every combination simultaneously,
instead of writing a loop that repeats it one at a time.

## What actually happens

X is a matrix (a grid of numbers) with shape (360, 29) — 360 images,
29 features each. W1 has shape (29, 8) — 29 features feeding into 8
neurons.

    X @ W1

produces a (360, 8) result — for every one of 360 images, the
weighted sum for every one of 8 neurons, all computed in one
operation.

Each individual number in that (360, 8) result is computed exactly
like the hand example in file 03: take one image's row of 29
features, take one neuron's column of 29 weights, multiply pairs
together, sum them up. Matrix multiplication just organizes doing
this many times at once, instead of one image/one neuron at a time
like I did by hand.

## Tiny example first, to remember the mechanic

Say X has 2 images, 3 features each (shape 2×3), and W has 3
features feeding into 2 neurons (shape 3×2):

    X = [[1, 2, 3],
         [4, 5, 6]]

    W = [[1, 0],
         [0, 1],
         [1, 1]]

X @ W produces a 2×2 result. Each output number = (one row of X)
dotted with (one column of W) — multiply matching positions, sum
them up. For row 1 of X, column 1 of W:

    (1×1) + (2×0) + (3×1) = 1 + 0 + 3 = 4

Row 1, column 2 of W:

    (1×0) + (2×1) + (3×1) = 0 + 2 + 3 = 5

Row 2, column 1:

    (4×1) + (5×0) + (6×1) = 4 + 0 + 6 = 10

Row 2, column 2:

    (4×0) + (5×1) + (6×1) = 0 + 5 + 6 = 11

Result:

    X @ W = [[4, 5],
             [10, 11]]

Same "multiply pairs, sum them" recipe as the single-neuron hand
example in file 03 — just done for 2 images and 2 neurons in one
shot, instead of one calculation at a time.

## Why this matters practically

Without matrix multiplication, `model.py` would need nested loops —
one loop over every image, another loop inside that over every
neuron — recomputing the exact same z = xW+b arithmetic each time.
Matrix multiplication (the @ operator, via numpy) does all of that
in one line, and it's also dramatically faster computationally,
since numpy runs these operations in optimized, low-level code
rather than plain Python loops.

Every step in files 03-07 — z1, a1, z2, ŷ, the backward pass
gradients — is written in `model.py` using this same matrix form,
computing the result for the whole batch of images simultaneously
rather than one at a time. 