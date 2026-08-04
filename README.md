# taste-net

A small neural network, built entirely from raw numpy (no PyTorch,
no scikit-learn), trained to predict whether I would save an image
— using only my own labeled Pinterest-style data.

This isn't trying to solve "aesthetics" or build a general taste
model. It's a scoped, honest question: can a simple model, trained
only on my choices, predict my choices — and where does it fail?

The most useful result in the project is a failure, and it was
predicted in writing before it was measured.

## Why this exists

I built [chaos-organizer](https://github.com/ffairylea/chaos-organizer)
using `sentence-transformers` for text embeddings — a real library
doing real work I didn't fully understand under the hood. This
project is the opposite move: build the smallest honest version of
"a model that learns preference" myself, from the math up, so I
actually understand what a library like that is doing when I use one.

No pretrained embeddings, no CLIP, no transfer learning. Every
feature the model sees is a number I computed by hand (see
`src/features.py`) — brightness, saturation, contrast, color
histograms. Every step of learning is backpropagation and gradient
descent I wrote myself (see `src/model.py`).

## What's actually happening, mathematically

- **Image → vector**: each image becomes a point in 32-dimensional
  space (5 color statistics + 3 average channel values + a 24-bin
  histogram). Same idea as a text embedding, with hand-built
  features instead of a trained encoder.
- **Learning = gradient descent**: the network guesses, measures how
  wrong it was (cross-entropy loss), and uses the chain rule applied
  layer by layer — backpropagation — to work out which direction to
  nudge every weight.
- **Sigmoid and softmax are the same function.** Softmax with two
  classes reduces algebraically to sigmoid applied to the difference
  of the two scores, which is why the binary model needs only one
  output neuron. Derived in `docs/11`.
- **The gradient simplification isn't a coincidence.** Sigmoid +
  binary cross-entropy cancels to $\hat y - y$; softmax + categorical
  cross-entropy cancels to the same thing with vectors. One identity
  appearing twice, because cross-entropy takes a log and both
  activations exponentiate.

## Results

### Phase 1 — binary: would I save this at all?

**89.25% test accuracy** (SGD, $lr = 0.1$), later **95.7%** with
Adam.

The class split is 279 yes / 187 no, so always guessing "yes" scores
60%. That's the baseline the number should be read against.

Feature importance showed the model leaning heavily on blue-channel
histogram bins — plausibly because pink-aesthetic images occupy a
narrow blue range, which makes blue an efficient discriminator.

### Phase 2 — multiclass: which board does it belong to?

Three Pinterest boards, deduplicated within and across folders
(fleurs 286, pink 275, winter_minimal 190):

**67.33% test accuracy** against a 38.0% majority-class baseline.

Rows are the true board, columns the prediction:

| | fleurs | pink | winter_minimal |
|---|---|---|---|
| **fleurs** | **35** | 18 | 4 |
| **pink** | 16 | **35** | 4 |
| **winter_minimal** | 5 | 2 | **31** |

| board | recall | precision |
|---|---|---|
| fleurs | 61.4% | 62.5% |
| pink | 63.6% | 63.6% |
| winter_minimal | 81.6% | 79.5% |

### The experiment this was actually built to run

`fleurs` was chosen deliberately to be hard. Most of the flower
images are pink, but they belong to a different board — so the model
has to separate them from the `pink` board on some basis other than
"is it pink." The question is whether 32 color statistics contain
enough information to distinguish categories that differ in *subject
matter* while sharing a *palette*.

The prediction was committed to `docs/15` before training ran. Four
of five claims held: winter_minimal separates cleanly, the largest
error cell is fleurs → pink, fleurs recall is far below
winter_minimal's, and overall accuracy lands between the baselines
and Phase 1.

The fifth was wrong, and usefully so. I predicted *absorption* —
pink flowers being swallowed by the pink board, an asymmetric error.
The measurement shows 18 fleurs → pink and 16 pink → fleurs. Nearly
symmetric, which means the model has no purchase on the distinction
rather than a biased one. Restricted to those two boards it scores
62.5%, barely above a coin flip.

**The number that isolates the cause:** collapsing fleurs and pink
into one class — asking only "pink-ish or winter_minimal" and
ignoring the distinction color cannot express — gives **90.0%**.
Phase 1 got 89.25%.

The same network, same width, same training, performs identically
well on the distinction color *can* represent and collapses on the
one it can't. That rules out capacity, optimization, data volume
(190 winter images support better recall than 286 fleurs images),
and class imbalance. What remains is representation: subject matter
is spatial, and every one of the 32 features averages or bins over
the whole image, discarding position before the network sees
anything.

This is the empirical case for learned features, measured on my own
data rather than asserted from a textbook. Full analysis and caveats
in `docs/15`.

## Verifying the math before trusting it

Two places where a result was checked rather than assumed.

**The backward pass.** `src/train_boards.py` runs a numerical
gradient check when no board data is present: nudge a weight by
$\pm h$, re-run the loss, and compute $(L(w+h) - L(w-h))/2h$ — the
definition of a derivative, no calculus involved. Worst disagreement
against the analytic gradients: ~7e-12, which is float64 noise
rather than error. Several weights return exactly zero from both
methods, which is the ReLU gate being closed and the numerical check
independently confirming what `relu_derivative` asserts. Output in
`docs/evidence/`.

**The optimizer benchmark.** My first comparison reported Adam
reaching a loss target in 38 epochs against SGD's 220 — an apparent
6× speedup. It was invalid: the learning rates were arbitrary, and
SGD at $lr = 2.0$ reaches the same target in 10 epochs, faster than
Adam. Sweeping learning rates instead gives the honest result — best
accuracy is similar across optimizers (94.6% vs 95.7%, a difference
of one test image), but SGD's accuracy swings 16.1 points across
learning rates while Adam's swings 7.5. Adam's real advantage here is
robustness to tuning, not speed. Details in `docs/14`.

A bug worth recording: `train.py` originally saved trained weights
but discarded the normalization statistics. Since the weights were
learned in a scaled coordinate system, this made them unusable on
new images — and the intuitive fix (normalizing a single image
against its own mean and std) produces a vector of all zeros, so
every image maps to the identical input and the model returns one
constant probability. Nothing crashes. Written up in `docs/12`.

## Architecture

Two stages, because "would I save this" and "which board" are
different questions:

```
Stage 1 (TasteNet, sigmoid)     "would I save this at all?"
          |
          |  only if yes
          v
Stage 2 (BoardNet, softmax)     "which board?"
```

The gate is architecturally necessary, not a convenience. Softmax
always produces a distribution summing to 1, so a board model handed
a photo of a parking garage will confidently assign it a board — it
has no way to say "none of these." Stage 1 supplies that answer.
`src/predict.py` runs the chain on a single new image.

## Status

- [x] Feature extraction pipeline (32 hand-computed features)
- [x] From-scratch neural net — forward and backward pass by hand
- [x] Phase 1: 89.25% test accuracy (60% majority baseline)
- [x] SGD, Momentum, and Adam implemented from scratch
      (`src/optimizers.py`), with bias correction derived from the
      geometric series
- [x] Learning-rate sweep replacing an invalid single-point benchmark
- [x] Phase 2 multiclass: 67.3% three-way, 90.0% collapsed
- [x] Fleurs experiment — prediction committed before results
- [x] 16 sequenced documentation files under `docs/`, worked by hand
      with real numbers
- [ ] Blue-channel ablation: does removing those features tank
      accuracy, or does the model route around them?
- [ ] Comparison against a pretrained vision model, to measure how
      much of the fleurs gap closes with learned features

## The uncomfortable question this project doesn't dodge

If I train a model only on things I've already chosen, and it "learns
my taste," what has it actually learned? A few honest possibilities,
not resolved here:

- It might just be replaying my existing biases back to me —
  confirming what I already like rather than discovering anything
  about aesthetics.
- "Taste" might not be a stable target. My saves from six months ago
  and today might already disagree, which would show up as noise the
  model can't fit.
- There's a real difference between predicting what I'd click and
  having anything resembling taste in the sense humans mean —
  intentionality, context, mood, meaning. This measures the first and
  shouldn't be oversold as the second.
- Board membership is my own filing decision. Some flower images
  might reasonably belong to the pink board, which puts an unknown
  ceiling below 100% that has nothing to do with the model.
- Scaled up, a system that reliably predicts and reinforces
  individual aesthetic preference starts to look like the mechanism
  behind algorithmic feeds and their documented narrowing effects. A
  toy version is worth understanding precisely because the real
  version already shapes much of what people see online.

I don't think this project resolves any of that. It's small enough
to let me feel the shape of the question instead of having an
opinion about it secondhand.

## Setup

```
pip install -r requirements.txt
```

Phase 1 — see `LABELING_GUIDE.md` for collecting `data/yes` and
`data/no`:

```
cd src
python3 train.py
```

Phase 2 — one folder per board under `data/boards/`, discovered
automatically by folder name:

```
cd src
python3 dedupe.py ../data/boards/<board>     # within-folder duplicates
python3 check_board_overlap.py               # cross-folder duplicates
python3 train_boards.py
```

With no `data/boards/` present, `train_boards.py` runs the gradient
check and synthetic training check instead of training.

Predicting on a new image, through both stages:

```
python3 predict.py ../path/to/image.jpg
```

Optimizer comparison:

```
python3 compare_optimizers.py
```

Board images are gitignored — the code is public, the Pinterest
source images stay local.

## Documentation

`docs/` contains 16 sequenced files building from "what is a neuron"
to the fleurs experiment, written while learning rather than
afterward. Derivations are worked by hand with real numbers, and
places where I was confused are left visible rather than polished
away.

## Related

Part of a two-project pair with
[chaos-organizer](https://github.com/ffairylea/chaos-organizer):
that one is the applied side (using existing ML tools to solve a real
workflow problem). This one is the fundamentals side (building the ML
itself from the ground up to understand what those tools are doing).