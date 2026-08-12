# 14 — Measuring the optimizers, and getting it wrong first

File 13 derived momentum and Adam. This file is about testing
whether they actually help on this dataset — and about a mistake in
the first attempt at that test, which is the more useful half.

## The first comparison, and why it was invalid

The initial version of `compare_optimizers.py` trained the same
network three times, changing only the update rule, and reported:

| optimizer | epochs to reach loss 0.3 | test accuracy |
|---|---|---|
| SGD, $lr = 0.1$ | 220 | 89.2% |
| Momentum, $lr = 0.1$ | 230 | 88.2% |
| Adam, $lr = 0.01$ | 38 | 95.7% |

Adam reaching the target in 38 epochs against SGD's 220 looks like
a decisive result — a roughly sixfold speedup, plus six points of
accuracy.

It was not a decisive result. It was not a result at all.

The learning rates in that table were not chosen by any principled
procedure. $lr = 0.1$ was simply what `train.py` had always used,
and $lr = 0.01$ is a commonly cited default for Adam. Two arbitrary
numbers, one per method.

Sweeping SGD's learning rate shows what that concealed:

| SGD $lr$ | final loss | epochs to reach 0.3 | test accuracy |
|---|---|---|---|
| 0.01 | 0.4160 | never (in 500) | 78.5% |
| 0.05 | 0.2861 | 438 | 79.6% |
| 0.1 | 0.1909 | 220 | 89.2% |
| 0.3 | 0.0927 | 74 | 94.6% |
| 0.5 | 0.0757 | 45 | 94.6% |
| 1.0 | 0.0470 | 25 | 92.5% |
| 2.0 | 0.0170 | **10** | 92.5% |

SGD at $lr = 2.0$ reaches the target in **10 epochs** — faster than
Adam's 38. The reported speedup was an artifact of comparing Adam at
a reasonable setting against SGD at a poor one.

The general failure: **comparing two methods at one arbitrary
hyperparameter each measures the hyperparameters, not the methods.**
Any result obtained that way can be reversed by choosing different
arbitrary values, which means it carries no information.

What made it worth checking was that the result was too flattering.
A sixfold speedup from swapping an update rule on a two-layer
network with 373 training images is not a plausible magnitude. The
suspicion came from the size of the claim, not from spotting the
error directly.

## The corrected comparison

The current script sweeps the learning rate for every optimizer and
reports the full range. Everything else is held fixed: same network,
same data, same starting weights (`seed=42`), 500 epochs.

**SGD** — table above.

**Momentum**

| $lr$ | final loss | epochs to 0.3 | test accuracy |
|---|---|---|---|
| 0.01 | 0.4171 | never | 78.5% |
| 0.05 | 0.2886 | 448 | 79.6% |
| 0.1 | 0.1967 | 230 | 88.2% |
| 0.3 | 0.0971 | 84 | 94.6% |
| 0.5 | 0.0688 | 54 | 94.6% |
| 1.0 | 0.0395 | 32 | 93.5% |

**Adam**

| $lr$ | final loss | epochs to 0.3 | test accuracy |
|---|---|---|---|
| 0.001 | 0.2084 | 289 | 88.2% |
| 0.005 | 0.0612 | 66 | 94.6% |
| 0.01 | 0.0280 | 38 | 95.7% |
| 0.05 | 0.0024 | 14 | 95.7% |

### What survives

| optimizer | best test accuracy | spread across $lr$ |
|---|---|---|
| SGD | 94.6% | 16.1 points |
| Momentum | 94.6% | 16.1 points |
| Adam | 95.7% | 7.5 points |

The best achievable accuracy is close across all three. A
well-tuned SGD is competitive with Adam on this problem, and the
dramatic speedup claimed by the first version does not exist.

What differs is the **spread**. SGD's result swings 16 points
depending on a number that has to be guessed in advance; Adam's
swings 7.5, and every setting from 0.005 to 0.05 lands in the
94.6–95.7% band.

That matches the mechanism derived in file 13. Dividing each step by
$\sqrt{\hat v}$ gives every weight an effective learning rate scaled
to its own gradient magnitude, which is precisely a reduction in
sensitivity to the global $lr$. The measurement confirms the
mechanism rather than a speed claim.

The honest summary is therefore **"needs less tuning"** rather than
**"converges faster"** — a smaller claim, and the one the evidence
supports.

`docs/evidence/optimizer_comparison.png` shows this directly: SGD's
loss curves fan out widely across learning rates, Adam's cluster
together. The spread within each panel is the finding.

## Two things that fell out of the sweep

### Momentum does essentially nothing here

Momentum matches plain SGD almost exactly — same best accuracy, same
16.1-point spread, near-identical curves at every learning rate.

This is not a bug, and the reason follows from file 13. Momentum
exists to average out *noise* in the gradient. But training here is
**full-batch**: every gradient is computed over all 373 training images at
once, so successive gradients are near-identical rather than noisy
samples. Averaging near-identical values changes almost nothing.

Momentum's benefit appears with mini-batches, where each gradient
comes from a small random subset and is genuinely noisy. The
training setup in this project removes the problem momentum solves.

Worth stating plainly rather than omitting: an implemented technique
that produced no measurable improvement, with a mechanistic
explanation for why.

### Overfitting is visible in the numbers

For every optimizer, the learning rate producing the **lowest
training loss** is not the one producing the **best test accuracy**:

| optimizer | lowest training loss | best test accuracy |
|---|---|---|
| SGD | $lr = 2.0$ → loss 0.0170, test 92.5% | $lr = 0.3$ → test 94.6% |
| Momentum | $lr = 1.0$ → loss 0.0395, test 93.5% | $lr = 0.3$ → test 94.6% |
| Adam | $lr = 0.05$ → loss 0.0024, test 95.7% | $lr = 0.01$ → test 95.7% |

SGD at $lr = 2.0$ drives training loss down to 0.017 — the lowest of
any SGD run — while scoring 2 points *worse* on the test set than
$lr = 0.3$, which has five times the training loss.

That is overfitting, appearing in this project's own measurements
rather than as a textbook warning: fitting the training data more
closely stopped corresponding to performing better on unseen data.

The plot makes the mechanism visible too. The $lr = 2.0$ curve is
visibly jagged rather than smooth — the steps are large enough to
overshoot repeatedly, and the run ends up in a sharper, less
generalizable minimum.

## Effect on the project's headline number

Adam at $lr = 0.01$ raises test accuracy from **89.2%** (the
original `train.py` configuration, SGD at $lr = 0.1$) to **95.7%**.

Two honest qualifications:

1. Roughly 5 of those points come from the learning rate rather than
   from Adam. SGD at $lr = 0.3$ reaches 94.6%. The original 89.2%
   partly reflected an undertuned baseline.
2. With 93 test images, one image is worth about 1.1 percentage
   points. The gap between 94.6% and 95.7% is a single image, which
   is well inside noise. The defensible claim is that Adam and a
   tuned SGD perform comparably, not that Adam is better.

The improvement is real, but the sweep is what established its size
honestly — and the size is smaller than the first uncorrected
comparison suggested.

## The general point

Both files 11 and 14 involve checking work rather than trusting it.
File 11 verified a hand-derived gradient against numerical
differentiation before any data existed. This file caught a
benchmark that had been unintentionally rigged in favour of the
conclusion it was testing for.

The pattern in both cases: a result that looked good was treated as
a reason for suspicion rather than satisfaction. The gradient check
could have been skipped on the grounds that the derivation seemed
right; the sweep could have been skipped on the grounds that the
first comparison already showed what it was supposed to show.

The corrected conclusion is less impressive than the original. That
is the usual outcome of checking, and it is the reason checking is
worth doing.