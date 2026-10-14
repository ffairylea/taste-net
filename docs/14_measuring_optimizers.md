# 14 — Measuring the optimizers, and getting it wrong first

File 13 derived momentum and Adam. This file is about testing
whether they actually help on this dataset — and about a mistake in
the first attempt at that test, which is the more useful half.

Every table in this file was re-measured after the dataset rebuild
described in file 16. The numbers below come from the current
`compare_optimizers.py` on the clean 176/176 dataset: 282 training
images, 70 test images, so **one test image is worth 1.43 percentage
points.** That figure matters for reading every comparison here.

## The first comparison, and why it was invalid

The initial version of `compare_optimizers.py` trained the same
network three times, changing only the update rule, and reported
(these numbers predate the rebuild and are kept only to show the
shape of the error):

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

Sweeping SGD's learning rate, on the same pre-rebuild data, showed
what that concealed: SGD at $lr = 2.0$ reached the loss target in
**10 epochs** against Adam's 38. The reported speedup was an
artifact of comparing Adam at a reasonable setting against SGD at a
poor one.

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

**SGD**

| $lr$ | final loss | epochs to 0.3 | train | test |
|---|---|---|---|---|
| 0.01 | 0.6317 | never | 63.8% | 65.7% |
| 0.05 | 0.5810 | never | 68.4% | 72.9% |
| 0.1 | 0.5329 | never | 72.3% | 72.9% |
| 0.3 | 0.3094 | never | 85.5% | 71.4% |
| 0.5 | 0.3104 | 395 | 88.3% | 70.0% |
| 1.0 | 0.1846 | 308 | 93.3% | 64.3% |
| 2.0 | 0.1903 | 156 | 93.3% | 75.7% |

**Momentum**

| $lr$ | final loss | epochs to 0.3 | train | test |
|---|---|---|---|---|
| 0.01 | 0.6321 | never | 63.8% | 65.7% |
| 0.05 | 0.5826 | never | 68.1% | 72.9% |
| 0.1 | 0.5381 | never | 72.3% | 72.9% |
| 0.3 | 0.3158 | never | 86.9% | 71.4% |
| 0.5 | 0.2135 | 336 | 91.1% | 65.7% |
| 1.0 | 0.0984 | 174 | 97.2% | 68.6% |

**Adam**

| $lr$ | final loss | epochs to 0.3 | train | test |
|---|---|---|---|---|
| 0.001 | 0.4501 | never | 76.6% | 68.6% |
| 0.005 | 0.1244 | 235 | 97.9% | 65.7% |
| 0.01 | 0.0547 | 131 | 98.9% | 65.7% |
| 0.05 | 0.0163 | 59 | 99.6% | 68.6% |

A note on the "epochs to 0.3" column: the 0.3 target was chosen
when this benchmark ran on the contaminated dataset, where losses
fell much further. On clean data most settings never reach it, so
the column now says more about how hard the target is than about
the optimizers. It is kept because the original comparison was
built on it, and because a threshold calibrated to bad data
quietly becoming uninformative is itself worth noticing.

### What survives

| optimizer | best test accuracy | spread across $lr$ |
|---|---|---|
| SGD | 75.7% | 11.4 points |
| Momentum | 72.9% | 7.1 points |
| Adam | 68.6% | 2.9 points |

**The accuracy column cannot be read as a ranking.** The 7.1-point
range between best-SGD and best-Adam is five test images out of 70.
Nothing here supports "SGD beats Adam on this problem" any more than
the first version supported the reverse — the correct statement is
that peak accuracy is indistinguishable across the three at this
sample size.

What is large enough to survive is the **spread**. SGD's result
swings 11.4 points depending on a number that has to be guessed in
advance — eight test images — while Adam's swings 2.9, which is two.
Every Adam setting from 0.001 to 0.05 lands in the 65.7–68.6% band.

That matches the mechanism derived in file 13. Dividing each step by
$\sqrt{\hat v}$ gives every weight an effective learning rate scaled
to its own gradient magnitude, which is precisely a reduction in
sensitivity to the global $lr$. The measurement confirms the
mechanism rather than a speed claim.

The honest summary is therefore **"needs less tuning"** rather than
**"converges faster"** — a smaller claim, and the one the evidence
supports. It is also the one claim in this file that survived the
dataset rebuild unchanged, which is some evidence it was the right
claim to make.

### What Adam's training loss says once the duplicates are gone

The sharpest number in the three tables is not an accuracy. Adam at
$lr = 0.05$ reaches a training loss of 0.0163 and a train accuracy
of **99.6%**, against a test accuracy of 68.6% — a 31-point gap. At
$lr = 0.01$ it is 98.9% against 65.7%. Adam fits this training set
essentially perfectly and gains nothing for it.

On the contaminated dataset that same behavior reported as
*accuracy*, because near-duplicates of the training images were
sitting in the test set, so memorizing them scored. Adam was the
best optimizer in the original benchmark and is the worst-looking
one here, and nothing about Adam changed — only whether memorization
was being rewarded.

<!-- TODO (Lea): this subsection is the one genuinely new finding in
     the rewrite and it is currently in my words, not yours. It is
     also good interview material, so it is worth writing yourself:
     the claim is just "an optimizer that fits the training set
     faster looks better the more contaminated your split is." -->

A note on `docs/evidence/optimizer_comparison.png`, because an
earlier version of this file described it wrongly. The plot plots
*training loss*, and the spread result above is about *test
accuracy*, which does not appear on it. Read literally, the figure
shows Adam's curves fanning out more than SGD's, not less — because
Adam reaches a far wider range of final training losses.

What the figure does show, clearly, is two things worth having.
SGD at $lr = 1.0$ and $lr = 2.0$ is violently jagged where every
other curve is smooth: overshooting made visual. And Adam drives
training loss roughly an order of magnitude below anything SGD
reaches, which — given that it bought no test accuracy — is the
overfitting result in picture form.

## Two things that fell out of the sweep

### Momentum does essentially nothing here

Momentum tracks plain SGD closely at every shared learning rate —
within one or two test images at 0.01, 0.05, 0.1 and 0.3, and with
near-identical loss curves throughout.

This is not a bug, and the reason follows from file 13. Momentum
exists to average out *noise* in the gradient. But training here is
**full-batch**: every gradient is computed over all 282 training
images at once, so successive gradients are near-identical rather
than noisy samples. Averaging near-identical values changes almost
nothing.

Momentum's benefit appears with mini-batches, where each gradient
comes from a small random subset and is genuinely noisy. The
training setup in this project removes the problem momentum solves.

Worth stating plainly rather than omitting: an implemented technique
that produced no measurable improvement, with a mechanistic
explanation for why.

### Overfitting is visible in the numbers

Sorting every run by final training loss, regardless of optimizer,
produces a clean monotone pattern — and the thing that grows is the
train/test gap, not the test accuracy:

| run | final loss | train | test | gap |
|---|---|---|---|---|
| SGD $lr = 0.1$ | 0.5329 | 72.3% | 72.9% | −0.6 |
| SGD $lr = 0.3$ | 0.3094 | 85.5% | 71.4% | +14.1 |
| Momentum $lr = 1.0$ | 0.0984 | 97.2% | 68.6% | +28.6 |
| Adam $lr = 0.01$ | 0.0547 | 98.9% | 65.7% | +33.2 |
| Adam $lr = 0.05$ | 0.0163 | 99.6% | 68.6% | +31.0 |

Driving the training loss from 0.53 to 0.016 — a factor of 33 — buys
27 points of training accuracy and **nothing on the test set.** Test
accuracy is flat to slightly worse across that entire range.

That is overfitting, appearing in this project's own measurements
rather than as a textbook warning: fitting the training data more
closely stopped corresponding to performing better on unseen data.
It is also much more visible on clean data than it was before the
rebuild, because previously some of that memorization was being
scored as test accuracy.

## Effect on the project's headline number

The original version of this section claimed Adam at $lr = 0.01$
raised test accuracy from 89.2% to 95.7%. Both numbers came from the
contaminated dataset, and on clean data the direction reverses: Adam
at $lr = 0.01$ scores 65.7% against `train.py`'s SGD $lr = 0.1$ at
72.9%.

The honest version is that **optimizer choice does not move this
project's headline number at all.** The default configuration gets
72.9%; the best result from any optimizer at any swept learning rate
is 75.7%, which is two test images away. Whatever is limiting Phase 1
is not the update rule.

Of the two qualifications offered for that original claim, one was
that with 93 test images a 1.1-point gap was a single image and
therefore inside noise. That reasoning was correct and it is the
reason this section needed so little rethinking after the rebuild:
the claim had already been cut down to something the sample size
could support. What it had not been cut down to was a dataset that
deserved trusting.

The useful lesson is that the noise caveat and the contamination
were independent problems. Getting the error bars right does not
protect you from measuring the wrong thing.

## The general point

Files 11, 14 and 16 all involve checking work rather than trusting
it. File 11 verified a hand-derived gradient against numerical
differentiation before any data existed. This file caught a
benchmark that had been unintentionally rigged in favor of the
conclusion it was testing for. File 16 caught the dataset that both
of them were measured on.

The pattern in both cases: a result that looked good was treated as
a reason for suspicion rather than satisfaction. The gradient check
could have been skipped on the grounds that the derivation seemed
right; the sweep could have been skipped on the grounds that the
first comparison already showed what it was supposed to show.

The corrected conclusion is less impressive than the original. That
is the usual outcome of checking, and it is the reason checking is
worth doing.