# 12: Normalization, and the state that has to be saved with it

File 01 mentioned that features get normalized before training. This
file covers what that actually does, why the model can't work
without it, and a bug it caused in this project, one that would
have failed silently rather than crashing.

## The problem: features on wildly different scales

The 32 features from `features.py` don't live in comparable ranges.
Rough typical values from the real dataset:

    brightness        0.56
    saturation        0.24
    color_variance    0.04
    contrast          0.17
    hist_b_6          0.03

`brightness` is roughly fourteen times larger than
`color_variance`. That difference is an accident of how each was
defined: variance is an average of *squared* distances, so it's
naturally small, while brightness is an average of values already
between 0 and 1.

This matters because of the weighted sum from file 03. The model
computes $z = w_1x_1 + w_2x_2 + \ldots$, and gradient descent nudges
each $w$ in proportion to the $x$ it multiplies. A feature whose
values are tiny produces tiny gradients, so its weight moves
slowly, not because the feature is uninformative, but purely
because its numbers happen to be small. The model would take far
longer to learn anything from `color_variance` than from
`brightness`, for a reason that has nothing to do with either
feature's actual usefulness.

Normalization removes that accident.

## Standard deviation, worked by hand

Take five images with these brightness values:

$$0.4,\quad 0.5,\quad 0.6,\quad 0.7,\quad 0.8$$

**Mean**: add them, divide by how many:

$$\mu = \frac{0.4+0.5+0.6+0.7+0.8}{5} = \frac{3.0}{5} = 0.6$$

**Standard deviation** answers a different question: not "what's
typical" but "how far from typical do these usually sit."

Distance of each value from the mean:

$$-0.2,\quad -0.1,\quad 0,\quad 0.1,\quad 0.2$$

Square each one. This is the step that matters: without squaring,
these sum to exactly zero every time, because the negatives and
positives cancel by construction (that's what a mean *is*). Squaring
makes everything positive so the distances can actually accumulate:

$$0.04,\quad 0.01,\quad 0,\quad 0.01,\quad 0.04$$

Average them:

$$\sigma^2 = \frac{0.10}{5} = 0.02$$

That average of squared distances is the **variance**. But squaring
also changed the units. We're now in "brightness squared," which
isn't comparable to the original numbers. Taking the square root
undoes that:

$$\sigma = \sqrt{0.02} \approx 0.141$$

That's the **standard deviation**: brightness values in this set
typically sit about 0.14 away from 0.6.

(Note this is the same variance idea as `color_variance` in
`features.py`, which measures spread of color *within* one image.
Here it measures spread of one feature *across* images. Same
formula, different axis.)

## Z-scoring: converting values into "how unusual is this"

$$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

Applied to the five values above:

$$\frac{0.8-0.6}{0.141} = +1.42 \qquad \text{(1.42 std devs above typical)}$$
$$\frac{0.6-0.6}{0.141} = 0 \qquad \text{(exactly typical)}$$
$$\frac{0.4-0.6}{0.141} = -1.42 \qquad \text{(1.42 below)}$$

Every feature now reads on the same scale, not "0.56 brightness"
but "half a standard deviation above average brightness." A
`color_variance` of 0.04 and a `brightness` of 0.56 might both
normalize to $+0.3$, and the model can now weigh them against each
other on equal footing.

The key idea: after normalization, a feature's numbers describe its
position *relative to the training distribution*, not its raw
magnitude.

## What that implies about saved models

The weights learned in this normalized coordinate system. A weight
attached to `brightness` doesn't mean "per unit of brightness"; it
means "per standard deviation of brightness, measured against the
training set."

So the numbers $\mu$ and $\sigma$ aren't scratch work. They're part
of the model's definition. Without them, the weights have no
meaning, because there's no way to know what coordinate system they
were expressed in.

There are 32 means and 32 standard deviations. `train.py` computed
them, used them, and then saved only the weights, throwing them
away. That's the bug.

## Why the obvious fix produces all zeros

The tempting repair is: normalize the new image the same way the
training images were normalized. Subtract its mean, divide by its
standard deviation.

But there is only **one** image.

What is the mean of a single number? Take brightness 0.8:

$$\mu = \frac{0.8}{1} = 0.8$$

The number itself. And the standard deviation? Distance from its own
mean:

$$0.8 - 0.8 = 0$$

Square it: 0. Average: 0. Square root:

$$\sigma = 0$$

So the z-score is:

$$\frac{0.8 - 0.8}{0} = \frac{0}{0}$$

`dataset.normalize` has a guard for zero standard deviation
(`std[std == 0] = 1`), which turns this into $0/1 = 0$.

**Zero.** And not just for brightness. Every feature's distance from
its own value is zero, so all 32 features become zero. The feature
vector is $[0, 0, 0, \ldots, 0]$.

Run it on a completely different image, brightness 0.3? Same
arithmetic, same result: all zeros. Every image in existence maps to
the identical vector.

## Why this failure is worse than a crash

Feed that all-zero vector into the forward pass:

$$Z_1 = XW_1 + b_1 = 0 \cdot W_1 + b_1 = b_1$$

The input vanishes entirely. $Z_1$ is just the bias, a constant
that doesn't depend on the image at all. Everything downstream is
therefore also constant, and the model returns the same probability
for a sunset, a parking garage, and a pink Pinterest image. Forever.

Nothing raises an error. The shapes are all correct, the arithmetic
is all valid, the output is a plausible-looking probability between
0 and 1. The code runs cleanly and produces a number, and the number
is meaningless.

This is worth stating plainly because it's the general shape of the
most dangerous class of bug in machine learning: the pipeline runs,
the output looks reasonable, and the only way to detect the problem
is to reason about what the operations *mean* rather than whether
they execute.

The zero-standard-deviation guard, incidentally, exists for a
legitimate reason: a feature that's constant across the whole
training set has $\sigma = 0$ and would divide by zero. Setting it
to 1 makes that feature normalize to 0 for every image, which is
correct: a feature that never varies carries no information. The
guard isn't wrong. It just also silently absorbs this very different
situation.

## The fix

`save()` now stores $\mu$ and $\sigma$ alongside the weights, and
`load()` returns them:

    def save(self, path, mean=None, std=None):
        arrays = dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        if mean is not None:
            arrays["mean"] = mean
        if std is not None:
            arrays["std"] = std
        np.savez(path, **arrays)

`predict.py` then applies those saved values to any new image:

    def apply_normalization(x, mean, std):
        std = np.where(std == 0, 1, std)
        return (x - mean) / std

Note what this is *not* doing: it never computes a mean or standard
deviation from the new image. It only applies numbers that came from
training. The new image is placed into the training set's coordinate
system, which is the only system where the weights mean anything.

`predict.py` also refuses to run against a weights file that has no
saved normalization stats, rather than guessing. Given that the
wrong behavior here is invisible, failing loudly is the only safe
option.

## Related: why normalization stats come from training only

`dataset.py` already noted this, and it's the same principle from a
different angle:

    X_train_norm, mean, std = normalize(X_train)
    X_test_norm, _, _ = normalize(X_test, mean, std)

The test set is normalized with the *training* mean and standard
deviation, not its own. Computing them from the test set would let
information about the test data influence how the test data gets
processed, a mild form of data leakage, and the reported accuracy
would be slightly optimistic.

A single new image at prediction time is just the extreme version of
that same rule: test data never gets to define its own coordinate
system.