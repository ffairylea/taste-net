# taste-net

A neural network built from raw numpy, trained on my own Pinterest
boards, used to run one experiment: **can 32 hand-computed color
statistics tell a flower from a pink bedroom?**

I predicted they couldn't, wrote the prediction into the repository
before training, and ran it. Then I found that two thirds of my
dataset were duplicates, rebuilt it from scratch, and ran everything
again.

The three most useful things in this project are a failure I
designed on purpose, a benchmark I caught myself rigging, and a data
bug that had been silently inflating every number I'd reported.

## The experiment

Three boards: `pink`, `winter_minimal`, and `fleurs`. The third was
chosen to be hard on purpose: most of the flower images *are*
pink, but they belong to a different board, so separating them
requires something other than color.

The prediction went into `docs/15` and was committed to git before
any training run. The result, on 137 test images:

| | fleurs | pink | winter_minimal |
|---|---|---|---|
| **fleurs** | **25** | 14 | 6 |
| **pink** | 9 | **29** | 8 |
| **winter_minimal** | **0** | 9 | **37** |

**66.4% three-way accuracy** against a 33.3% chance baseline, on the
single split `train_boards.py` runs by default. Across five splits
the same architecture averages **58.8% ±3.4**, so the default split is
about two standard deviations lucky, and the multi-seed figure is the
one to trust. The confusion structure below is stable across seeds;
the headline number is not.

`winter_minimal` was never once classified as `fleurs` (zero times
out of 46), while fleurs and pink were confused 23 times, the
dominant error mode.

Splitting the same run in two directions:

| question | result |
|---|---|
| pink-ish vs winter_minimal | **82.5% balanced accuracy** |
| fleurs vs pink | **59.3%**, z = 1.78, not significant |

Same features, same network, same training run. Capable on one
question, at chance on the other.

Four of five pre-registered claims held. The fifth, that the
fleurs/pink confusion would be strongly *asymmetric*, flowers
absorbed into the pink board, is not supported: 14 errors one way,
9 the other, which is inside chance.

### Why it isn't the model

- **Capacity:** a width sweep from 4 to 128 hidden units is flat:
  test accuracy stays between 54.2% and 58.8% across five seeds per
  width, while train accuracy climbs from 63.0% to 91.5%. Extra
  capacity buys memorization, not generalization.
- **Imbalance:** boards are 228 / 233 / 233.
- **Optimization:** train 65.7%, test 66.4%, a 0.7 point gap, and
  `docs/14` shows optimizer choice moves the binary headline by at
  most two test images.
- **The features alone:** best d′ separating pink-ish from winter is
  0.88 against 0.63 for fleurs vs pink, and 1.6× on the mean across
  all 32 features. This is measured on the data with no model
  involved, which makes it the strongest evidence: nothing is fitted.
- **Unlimited capacity, with a caveat:** kNN peaks at 64.1% on
  fleurs vs pink. But it only reaches 69.0% on pink vs winter, where
  the network gets 82.5%, so kNN is a weaker classifier here rather
  than a true upper bound. Consistent with the claim, not proof of
  it. See `docs/15`.

What remains is representation. Subject matter is spatial, and every
one of the 32 features averages or bins over the whole image,
discarding position before the network sees anything.

## The data bug

The first version of this experiment ran on images scraped with a
browser extension. A kNN probe returned 97.7% at k=1 but 78.8% at
k=5. That is backwards: 1-NN is the noisiest classifier there is and
should lose to k=5.

That ordering has one common cause: every test image has a
near-identical twin in training, so 1-NN retrieves rather than
generalizes.

The existing deduplication scripts compared MD5 hashes of file
bytes, which catches byte-identical files and nothing else. The same
pin at two resolutions passes as two distinct images. Perceptual
hashing (greyscale, downsample to 8×8, threshold at the mean, 64
bits, compare by Hamming distance) found the truth:

    boards      751 images  ->  67.9% near-duplicate
    yes / no    466 images  ->  43.1% near-duplicate

Pinterest virtualizes its feed, so scroll-and-scrape captures the
same pins repeatedly at whatever resolution is currently rendered.
The extension worked correctly; the method was wrong.

**What the contamination did:** it did not merely inflate numbers,
it produced a confident result pointing the wrong way. On dirty
data, three-way accuracy climbed from 70.8% to 95.2% across the
width sweep, apparently proving that capacity was the limit and the
representation claim was wrong. On clean data the sweep is flat.
Extra width had been buying memorization of twins, not learning.

The fix was a different source. Pinterest's official data export
lists each saved pin once; `src/parse_pinterest_export.py`
reconstructs image URLs from the hash field and names files by that
hash, so re-downloading overwrites rather than accumulates.
Duplicates become impossible rather than detectable. Near-duplicate
rate after the rebuild: **6.8%**. Full account in `docs/16`.

## Phase 1

**72.9% test accuracy** against an exact 50% baseline (176 / 176).
Train accuracy 72.3%, so no overfitting gap.

This number went *down* from a previously reported 89.25%, and the
drop is an improvement. The original positive class came entirely
from the pink board, which meant the model was plausibly learning
"is this pink" rather than "would this be saved." The rebuilt set
samples across all 56 boards, making it the question it was supposed
to be.

## The optimizer benchmark I got wrong

My first comparison reported Adam reaching a loss target in 38
epochs against SGD's 220, an apparent 6× speedup. It was invalid:
the learning rates were arbitrary, and SGD at $lr = 2.0$ reaches the
same target in 10 epochs, faster than Adam. Comparing two methods at
one arbitrary hyperparameter each measures the hyperparameters.

Sweeping properly, on the rebuilt dataset: peak accuracy is
indistinguishable across the three optimizers (75.7% / 72.9% / 68.6%
for SGD / Momentum / Adam, a range of five test images out of 70),
but SGD's accuracy swings 11.4 points across learning rates while
Adam's swings 2.9. Adam's advantage here is robustness to tuning,
not speed. That is a smaller claim than the first version implied,
and the only one that survived the rebuild. Details in `docs/14`.

Adam also makes the clearest overfitting case in the project: at
$lr = 0.05$ it reaches 99.6% train accuracy and 68.6% test. On the
contaminated data that behavior scored as accuracy, which is why
Adam looked best before and no better than the others now.

What made it worth checking was that the result was too flattering.
A sixfold speedup from swapping an update rule on a two-layer
network is not a plausible magnitude. The suspicion came from the
size of the claim, not from spotting the error.

## Verifying the math

`src/train_boards.py` runs a numerical gradient check when no board
data is present: nudge a weight by ±h, re-run the loss, compute
$(L(w+h) - L(w-h))/2h$, the definition of a derivative, no calculus
involved. Worst disagreement against the analytic gradients: ~7e-12,
which is float64 noise. Several weights return exactly zero from
both methods, which is a closed ReLU gate confirmed independently.

A separate bug worth recording: `train.py` originally saved trained
weights but discarded the normalization statistics. Since the
weights were learned in a scaled coordinate system, this made them
unusable on new images. The intuitive fix (normalizing a single
image against its own mean and std) produces all zeros, so every
image maps to the same input and the model returns one constant
probability. Nothing crashes. Written up in `docs/12`.

## What's under the hood

No pretrained embeddings, no CLIP, no transfer learning. Every
feature is a number computed by hand in `src/features.py`:
brightness, saturation, contrast, warmth, color variance, three
average channels, and a 24-bin color histogram. Every step of
learning is backpropagation and gradient descent written from
scratch in `src/model.py`.

The math worth knowing:

- **Sigmoid and softmax are the same function.** Two-class softmax
  reduces algebraically to sigmoid applied to the difference of the
  two scores, which is why the binary model needs one output neuron
  rather than two.
- **The gradient simplification isn't luck.** Sigmoid + binary
  cross-entropy cancels to $\hat y - y$; softmax + categorical
  cross-entropy cancels to the same thing with vectors. One identity
  appearing twice, because cross-entropy takes a log and both
  activations exponentiate.
- **Adam's bias correction is exact, not a fudge.** The EMA returns
  $(1 - \beta^t)g$ rather than $g$; unrolling the recursion gives a
  geometric series whose sum makes the correction factor fall out
  algebraically. Derived step by step in `docs/13`.

## Architecture

```
Stage 1 (TasteNet, sigmoid)     "would I save this at all?"
          |
          |  only if yes
          v
Stage 2 (BoardNet, softmax)     "which board?"
```

The gate is necessary rather than convenient. Softmax always
produces a distribution summing to 1, so a board model handed a
photo of a parking garage will confidently assign it a board. It
has no way to say "none of these." Stage 1 supplies that answer.
`src/predict.py` runs the chain on a single new image.

## Status

- [x] 32 hand-computed features, forward and backward pass by hand
- [x] SGD, Momentum, and Adam from scratch (`src/optimizers.py`)
- [x] Gradient check against numerical differentiation
- [x] Learning-rate sweep replacing an invalid single-point benchmark
- [x] Fleurs experiment, prediction committed before results
- [x] Duplicate contamination found and fixed at the source
- [x] Dataset rebuilt from Pinterest's official export
- [x] 17 documentation files, worked by hand with real numbers
- [ ] Pretrained vision model comparison: how much of the 59.3% →
      82.5% gap closes when features are learned rather than
      hand-designed?
- [ ] Blue-channel ablation: does removing those features tank
      accuracy, or does the model route around them?

## The uncomfortable question this doesn't dodge

If I train a model only on things I've already chosen, and it
"learns my taste," what has it learned?

- It might be replaying my existing biases rather than discovering
  anything about aesthetics.
- Taste might not be a stable target: my saves from six months ago
  and today may already disagree, appearing as noise the model
  can't fit.
- Predicting what I'd click is not the same as having taste in the
  sense humans mean: intentionality, context, mood, meaning. This
  measures the first.
- Board membership is my own filing decision. Six near-duplicate
  groups span two boards, which is direct evidence that some images
  legitimately belong to both. That is an unknown ceiling below 100% that
  has nothing to do with the model.
- Scaled up, a system that predicts and reinforces individual
  aesthetic preference looks like the mechanism behind algorithmic
  feeds and their documented narrowing effects. A toy version is
  worth understanding because the real version already shapes much
  of what people see.

I don't think this resolves any of that. It's small enough to let me
feel the shape of the question instead of having an opinion about it
secondhand.

## Setup

```
pip install -r requirements.txt
```

Building the dataset from a Pinterest data export (Settings →
Privacy and data → request your data):

```
cd src
python3 parse_pinterest_export.py ~/path/to/export
python3 parse_pinterest_export.py ~/path/to/export --download \
    --board-id <ID> --out ../data/boards/<name> --limit 233
```

Verifying it:

```
python3 check_near_duplicates.py      # perceptual hash
python3 check_board_overlap.py        # exact hash, across boards
```

Training and analysis:

```
python3 train.py                  # Phase 1
python3 train_boards.py           # Phase 2
python3 compare_optimizers.py     # learning-rate sweep
python3 probe_representation.py   # kNN, width sweep, d-prime
python3 predict.py ../image.jpg   # both stages on a new image
```

Board images are gitignored. The code is public; the Pinterest
source images stay local.

## Documentation

`docs/` contains 17 sequenced files building from "what is a neuron"
to the duplicate contamination post-mortem, written while learning
rather than afterward. Derivations are worked by hand with real
numbers, and places where I was confused or wrong are left visible
rather than polished away. Files 14, 15 and 16 each contain a
correction to something I had previously reported.

## Related

Part of a pair with
[chaos-organizer](https://github.com/ffairylea/chaos-organizer):
that one is the applied side, using existing ML tools to solve a
workflow problem. This one is the fundamentals side: building the
machinery to understand what those tools are doing.