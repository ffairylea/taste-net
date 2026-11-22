# taste-net

A small neural network I wrote from scratch in NumPy and trained on
my own Pinterest boards, to test one question: **can color alone
tell apart the images I sort into different boards?**

I wrote down what I expected before running anything, and the
prediction mostly held. Along the way I found that two thirds of my
dataset were near copies of each other, which had been making every
result look better than it was. I rebuilt the dataset and ran
everything again. All numbers below are from the rebuilt data.

## In plain words

This part needs no machine learning background.

I paint, and I save a lot of images on Pinterest, sorted into boards.
I wanted to know whether what makes me save an image, and which
board I put it in, can be measured.

1. **Turning images into numbers.** Each image becomes a list of 32
   numbers that describe its color: how bright it is, how saturated,
   how warm, and how much of each range of colors it contains. None
   of the 32 numbers say *where* anything is in the image. That turns
   out to matter.
2. **A program that learns from examples.** A neural network is a
   function with many adjustable numbers in it, called weights. It
   makes a guess, measures how wrong the guess was, and nudges every
   weight slightly in the direction that makes it less wrong. After
   thousands of rounds, the guesses get better. I wrote each of these
   steps myself instead of using a machine learning library, because
   I wanted to understand them.
3. **A prediction made in advance.** I picked three boards: `pink`,
   `winter_minimal` and `fleurs` (flowers). Most of my flower images
   are also pink. I predicted that color would easily separate winter
   from pink, but would fail to separate flowers from my other pink
   images, since those differ in *what* is in the picture rather than
   in color. I committed this prediction to git before training, so
   the timestamp shows it came first.
4. **What happened.** Winter vs pink: 82.5% correct, where random
   guessing would get 50%. Flowers vs pink: 59.3%, which is not
   reliably better than guessing. The same network, from the same
   training run, can do the first and not the second.
5. **Checking that it wasn't the network.** Making the network much
   bigger did not help. The limit was the information I gave it, not
   the size of the model.
6. **The bug.** Before the rebuild, a bigger network had seemed to
   help a lot. Then one check gave a result that made no sense (a
   method that should have done worse did much better), and when I
   looked into it, 67.9% of my images turned out to be near copies:
   the same picture saved at different sizes. The network was being
   tested on pictures it had already seen. I rebuilt the dataset from
   Pinterest's own data export and reran everything.

## Results at a glance

| Question | Result | Random guessing |
|---|---|---|
| Winter vs pink-ish (pink and fleurs together) | **82.5%** | 50% |
| Flowers vs pink | **59.3%**, not statistically significant | 50% |
| All three boards at once, average of 5 splits | **58.8%** | 33.3% |
| Phase 1: would I save this image at all? | **72.9%** | 50% |

Every result is measured on test images the network never saw during
training. Terms used below are defined in
[`docs/GLOSSARY.md`](docs/GLOSSARY.md) and in the numbered docs.

## The experiment in detail

Three boards: `pink`, `winter_minimal`, and `fleurs`. The third was
chosen to be hard on purpose: most of the flower images *are* pink,
but they belong to a different board, so separating them requires
something other than color.

The prediction is in
[`docs/15`](docs/15_the_fleurs_experiment.md) and was committed to
git before any training run. The result on 137 test images, where
each row is the true board and each column is the network's guess:

| true \ guessed | fleurs | pink | winter_minimal |
|---|---|---|---|
| **fleurs** | **25** | 14 | 6 |
| **pink** | 9 | **29** | 8 |
| **winter_minimal** | **0** | 9 | **37** |

That is **66.4% three-way accuracy** against a 33.3% chance baseline,
on the single split `train_boards.py` uses by default. A split is
one random division of the images into training and test sets.
Across five different splits the same network averages **58.8%
±3.4**, so the default split happened to be about two standard
deviations lucky, and 58.8% is the number to trust. The pattern of
errors is stable across splits; the headline number is not.

`winter_minimal` was never once guessed as `fleurs` (zero times out
of 46), while fleurs and pink were confused 23 times, the most
common error.

Splitting the same run into two yes/no questions:

| question | result |
|---|---|
| pink-ish vs winter_minimal | **82.5%** balanced accuracy |
| fleurs vs pink | **59.3%**, z = 1.78, not significant |

Balanced accuracy averages the accuracy on each class, so a larger
class can't inflate it. A z-score of 1.78 is below the usual 1.96
cutoff, so 59.3% can't be distinguished from guessing on this many
test images.

Four of my five pre-registered claims held. The fifth, that flowers
would mostly be absorbed into the pink board rather than the other
way round, is not supported: 14 errors one way, 9 the other, which
is within chance.

### Why it isn't the model

- **Size of the network:** I tried hidden layers from 4 to 128
  neurons. Test accuracy stays between 54.2% and 58.8% (five splits
  per size), while training accuracy climbs from 63.0% to 91.5%. A
  bigger network only memorizes the training images better; it
  doesn't do better on new ones.
- **Unequal board sizes:** the boards have 228, 233 and 233 images,
  so none dominates.
- **Training problems:** training accuracy is 65.7% and test is
  66.4%, so the network is not overfitting, and
  [`docs/14`](docs/14_measuring_optimizers.md) shows that changing
  the optimizer moves the result by at most two test images.
- **The features on their own, with no model:** d′ measures how far
  apart two groups are on one feature, relative to how spread out
  they are. The best single feature separates pink-ish from winter
  at d′ = 0.88, but fleurs from pink at only 0.63, and averaged over
  all 32 features the gap is 1.6 times larger. Nothing is trained
  here, so this is the strongest evidence.
- **A different kind of classifier, with a caveat:** k-nearest
  neighbors (kNN) labels each test image by looking up the most
  similar training images. It reaches at most 64.1% on fleurs vs
  pink. But it also only reaches 69.0% on pink vs winter, where the
  network gets 82.5%, so kNN is simply a weaker classifier here and
  not a true upper limit. It is consistent with the claim, not proof
  of it.

What remains is the representation. Subject matter is spatial, and
every one of the 32 features averages or counts over the whole
image, throwing away position before the network sees anything.

## The data bug

The first version of this experiment used images collected with a
browser extension that downloads everything on a page. A kNN check
on fleurs vs pink gave 97.7% when looking at the single nearest image (k=1), but
78.8% when taking a vote among the five nearest (k=5). That is
backwards: one neighbor is easily fooled by one odd image, so k=1
should lose to k=5.

The usual reason for that pattern is that every test image has a
near-identical twin in the training set, so k=1 just finds the twin
and copies its label.

My duplicate check compared MD5 hashes of the file bytes, a
fingerprint that only matches when two files are byte-for-byte
identical. The same pin saved at two resolutions passes as two
different images. Perceptual hashing compares what images look
like instead (convert to grayscale, shrink to 8×8, mark each pixel
as above or below the average, giving 64 bits, then count how many
bits differ). It found:

    boards      751 images  ->  67.9% near-duplicate
    yes / no    466 images  ->  43.1% near-duplicate

Pinterest only keeps the pins near your screen loaded and re-renders
them as you scroll, often at a different size. So scrolling and
downloading captured the same pins again and again. The extension
did what it was supposed to; collecting images that way was the
mistake.

**What the duplicates did:** they didn't just inflate the numbers,
they pointed to the wrong conclusion. On the contaminated data,
three-way accuracy rose from 70.8% to 95.2% as the network got
bigger, which looked like proof that the network was too small and
that my explanation was wrong. On clean data the same sweep is flat.
The bigger networks had been memorizing twins.

The fix was a different source. Pinterest's official data export
lists each saved pin once.
[`src/parse_pinterest_export.py`](src/parse_pinterest_export.py)
rebuilds each image's URL from the export and names the file after
the pin's hash, so downloading again overwrites a file instead of
adding a copy. Near-duplicate rate after the rebuild: **6.8%**. Full
account in [`docs/16`](docs/16_the_duplicate_mishap.md).

## Phase 1: would I save this at all?

Before the board experiment, a simpler yes/no model: images I saved
(`data/yes`) against varied photos from Unsplash (`data/no`).

**72.9% test accuracy** against a 50% baseline (176 images in each
class). Training accuracy is 72.3%, so there is no overfitting gap.

This is lower than the 89.25% I reported earlier, and the drop is
an improvement. The original "yes" images all came from my pink
board, so the model could have been learning "is this pink" rather
than "would I save this." The rebuilt set samples across all 56 of
my boards, so it now asks the question it was meant to.

## The optimizer comparison I got wrong

An optimizer is the rule for how the weights get nudged after each
guess, and the learning rate is how big each nudge is. My first
comparison said Adam reached a target loss in 38 epochs (passes over
the training data) against plain gradient descent's (SGD's) 220, a
6× speedup. That comparison was invalid: I had picked the learning
rates arbitrarily, and SGD at $lr = 2.0$ reaches the same target in
10 epochs, faster than Adam. Comparing two methods at one arbitrary
setting each measures the settings, not the methods.

Testing a range of learning rates for each, on the rebuilt dataset:
the best accuracies are indistinguishable (75.7% / 72.9% / 68.6% for
SGD / Momentum / Adam, a difference of five test images out of 70).
But SGD's accuracy swings 11.4 points depending on the learning
rate, while Adam's swings only 2.9. So Adam's advantage here is that
it is less sensitive to tuning, not that it is faster. That is a
smaller claim than my first version made. Details in
[`docs/14`](docs/14_measuring_optimizers.md).

Adam also shows the clearest overfitting in the project: at
$lr = 0.05$ it reaches 99.6% on training images and 68.6% on test
images. On the contaminated data, memorizing like that also scored
well on the test set, which is why Adam looked best before and no
better than the others now.

What made me check was that the result was too good. A sixfold
speedup from changing the update rule on a two-layer network is not
a believable size of effect.

## Checking the math

When no board images are present,
[`src/train_boards.py`](src/train_boards.py) runs a numerical
gradient check instead. For a weight $w$, it computes
$(L(w+h) - L(w-h))/2h$, the slope of the loss measured directly by
nudging the weight, and compares it with the slope my
backpropagation code calculates. The largest disagreement is about
7e-12, which is rounding noise. Several weights give exactly zero
from both methods, which happens when a ReLU neuron is switched off.
The saved output is in
[`docs/evidence/gradient_check_output.txt`](docs/evidence/gradient_check_output.txt).

A separate bug worth recording: `train.py` originally saved the
trained weights but not the normalization statistics (the mean and
spread of each feature, used to rescale inputs). Without them the
weights can't be used on new images. The obvious fix, normalizing a
single image against its own mean, turns every input into zeros, so
the model returns the same answer for every image, and nothing
crashes to warn you. Written up in
[`docs/12`](docs/12_normalization_and_saved_state.md).

## How it works

No pretrained models and no machine learning libraries. Every
feature is computed by hand in [`src/features.py`](src/features.py):
brightness, saturation, contrast, warmth, color variance, three
average color channels, and a 24-bin color histogram. Every step of
learning (backpropagation and gradient descent) is written from
scratch in [`src/model.py`](src/model.py).

Three pieces of math that come up:

- **Sigmoid and softmax are the same function for two classes.**
  Softmax with two classes reduces to sigmoid applied to the
  difference of the two scores, which is why the yes/no model needs
  one output neuron rather than two.
- **The gradient simplification isn't luck.** Sigmoid with binary
  cross-entropy simplifies to $\hat y - y$, and softmax with
  categorical cross-entropy simplifies to the same thing with
  vectors. It is one identity appearing twice: cross-entropy takes a
  log and both activations use an exponential.
- **Adam's bias correction is exact.** The running average Adam
  keeps starts out too small, by a factor of $(1 - \beta^t)$.
  Writing out the recursion gives a geometric series, and the
  correction factor follows from its sum. Derived step by step in
  [`docs/13`](docs/13_momentum_and_bias_correction.md).

The full system has two stages:

```
Stage 1 (TasteNet, sigmoid)     "would I save this at all?"
          |
          |  only if yes
          v
Stage 2 (BoardNet, softmax)     "which board?"
```

The first stage is needed because softmax always produces
probabilities that add up to 1. A board model shown a photo of a
parking garage will still confidently assign it a board; it has no
way to say "none of these." Stage 1 gives that answer.
[`src/predict.py`](src/predict.py) runs both stages on a new image.

## What's in this repository

**Code** (`src/`). Run everything from inside `src/`.

| file | what it does |
|---|---|
| [`features.py`](src/features.py) | turns one image into its 32 color numbers; the only file that touches pixels |
| [`dataset.py`](src/dataset.py) | loads `data/yes` and `data/no`, normalizes, splits into train and test |
| [`dataset_boards.py`](src/dataset_boards.py) | the same for board folders, with one label per board |
| [`model.py`](src/model.py) | the yes/no network: forward pass, loss, backpropagation |
| [`model_multiclass.py`](src/model_multiclass.py) | the board network: same structure, softmax output |
| [`optimizers.py`](src/optimizers.py) | SGD, Momentum and Adam, written from scratch |
| [`train.py`](src/train.py) | trains and tests the Phase 1 yes/no model |
| [`train_boards.py`](src/train_boards.py) | trains the board model, prints the confusion matrix, runs the gradient check if no data is present |
| [`predict.py`](src/predict.py) | runs both stages on a new image or folder |
| [`compare_optimizers.py`](src/compare_optimizers.py) | the learning-rate sweep across the three optimizers |
| [`probe_representation.py`](src/probe_representation.py) | the checks on the features: kNN, network size sweep, d′ |
| [`check_near_duplicates.py`](src/check_near_duplicates.py) | perceptual hashing, which found the duplicates |
| [`check_board_overlap.py`](src/check_board_overlap.py) | finds identical images that ended up in two boards |
| [`dedupe.py`](src/dedupe.py) | removes byte-identical files from a folder |
| [`parse_pinterest_export.py`](src/parse_pinterest_export.py) | builds the clean dataset from Pinterest's data export |
| [`fetch_unsplash_no.py`](src/fetch_unsplash_no.py) | downloads the Unsplash photos used as "no" examples |

**Documentation** (`docs/`). The numbered files build from "what is
a neuron" to the duplicate post-mortem and are meant to be read in
order. I wrote them while learning, not afterward, and places where
I was confused or wrong are left in. Files 14, 15 and 16 each
correct something I had reported before.

| file | topic |
|---|---|
| [`00`](docs/00_overview.md) | overview and how to read the docs |
| [`01`](docs/01_the_data.md) | how an image becomes 32 numbers, worked on a tiny 2×2 image |
| [`02`](docs/02_neurons_and_layers.md) | neurons and layers |
| [`03`](docs/03_the_weighted_sum.md) | what one neuron computes |
| [`04`](docs/04_nonlinearity.md) | why stacking weighted sums alone gains nothing (ReLU) |
| [`05`](docs/05_sigmoid.md) | sigmoid: turning a score into a probability |
| [`06`](docs/06_loss_function.md) | cross-entropy loss: measuring how wrong a guess was |
| [`07`](docs/07_backward_pass.md) | backpropagation: how the network corrects itself |
| [`08`](docs/08_matrix_multiplication.md) | doing it for all images at once with matrices |
| [`09`](docs/09_implementation_details.md) | practical details in the code |
| [`10`](docs/10_project_files_overview.md) | longer explanation of each file in `src/` |
| [`11`](docs/11_softmax_and_multiclass.md) | softmax and choosing between several boards |
| [`12`](docs/12_normalization_and_saved_state.md) | normalization, and the bug it caused |
| [`13`](docs/13_momentum_and_bias_correction.md) | momentum and Adam, with the bias correction derived |
| [`14`](docs/14_measuring_optimizers.md) | testing the optimizers, and getting it wrong first |
| [`15`](docs/15_the_fleurs_experiment.md) | the pre-registered fleurs experiment and its results |
| [`16`](docs/16_the_duplicate_mishap.md) | how 68% of the dataset turned out to be duplicates |
| [`17`](docs/17_whats_next_and_the_math.md) | what comes next, and the math behind it |

Also in `docs/`:
[`GLOSSARY.md`](docs/GLOSSARY.md) (short definitions),
[`ETHICS_AND_PHILOSOPHY.md`](docs/ETHICS_AND_PHILOSOPHY.md),
[`LIMITATIONS_AND_IDEAS.md`](docs/LIMITATIONS_AND_IDEAS.md),
[`PHASE2_MULTIBOARD_PLAN.md`](docs/PHASE2_MULTIBOARD_PLAN.md) (the
original plan for the board model, kept as a record), and
[`NEXT_EXPERIMENT_SPATIAL_FEATURES.md`](docs/NEXT_EXPERIMENT_SPATIAL_FEATURES.md)
(the plan for the next experiment). `docs/evidence/` holds saved
output from the gradient check and the optimizer comparison.

**Other files.** `data/yes` and `data/no` are the Phase 1 images,
included so `train.py` runs straight away. The board images for
Phase 2 stay on my computer. `board_net_classes.txt` is written by
`train_boards.py` and records the board order that `predict.py`
needs. `requirements.txt` lists the two libraries needed, NumPy and
Pillow (matplotlib is optional and only draws the optimizer plot).

## Questions this raises

If I train a model only on things I've already chosen, and it
"learns my taste," what has it learned?

- It might be replaying my existing biases rather than discovering
  anything about aesthetics.
- Taste might not be a stable target: what I saved six months ago
  and what I save today may already disagree, which would show up as
  noise the model can't fit.
- Predicting what I'd click is not the same as having taste in the
  sense people usually mean: intention, context, mood, meaning. This
  only measures the first.
- Which board an image goes in is my own filing decision. Six groups
  of near-duplicates span two boards, which shows that some images
  really do belong to both. That sets a ceiling below 100% that has
  nothing to do with the model.
- Scaled up, a system that predicts and reinforces one person's
  aesthetic preferences looks like the mechanism behind recommendation
  feeds and their documented narrowing effects. A toy version is
  worth understanding because the real version already shapes much of
  what people see.

I don't think this resolves any of that. It is small enough to let me
feel the shape of the question instead of having an opinion about it
secondhand. More in
[`docs/ETHICS_AND_PHILOSOPHY.md`](docs/ETHICS_AND_PHILOSOPHY.md).

## Status

- [x] 32 hand-computed features, forward and backward pass by hand
- [x] SGD, Momentum, and Adam from scratch
- [x] Gradient check against numerical differentiation
- [x] Learning-rate sweep replacing an invalid single-point benchmark
- [x] Fleurs experiment, prediction committed before results
- [x] Duplicate contamination found and fixed at the source
- [x] Dataset rebuilt from Pinterest's official export
- [x] 18 documentation files
- [ ] Spatial features (edge direction, fractal dimension, sharpness,
      symmetry, coarse layout): do they lift fleurs vs pink above
      chance where color can't? Plan in
      [`docs/NEXT_EXPERIMENT_SPATIAL_FEATURES.md`](docs/NEXT_EXPERIMENT_SPATIAL_FEATURES.md).
- [ ] Self-consistency ceiling: re-label 100 images blind, weeks
      later. How often I agree with myself caps what any model can
      reach.
- [ ] Comparison with a pretrained vision model: how much of the gap
      between 59.3% and 82.5% closes when the features are learned
      rather than designed by hand?
- [ ] Blue-channel ablation: does removing those features hurt
      accuracy, or does the model route around them?

## Running it

```
pip install -r requirements.txt
cd src
```

Phase 1 runs on the images already in the repo:

```
python3 train.py                  # Phase 1 yes/no model
python3 compare_optimizers.py     # learning-rate sweep
```

Phase 2 needs your own boards, built from a Pinterest data export
(Settings → Privacy and data → request your data):

```
python3 parse_pinterest_export.py ~/path/to/export
python3 parse_pinterest_export.py ~/path/to/export --download \
    --board-id <ID> --out ../data/boards/<name> --limit 233
python3 check_near_duplicates.py      # perceptual hash
python3 check_board_overlap.py        # identical files across boards
python3 train_boards.py               # board model
python3 probe_representation.py       # kNN, size sweep, d-prime
python3 predict.py ../image.jpg       # both stages on a new image
```

Without board data, `train_boards.py` runs the gradient check
instead.

## How I made this

I built this to understand what is happening inside tools I had
used without knowing the math, like the sentence-transformers in
[chaos-organizer](https://github.com/ffairylea/chaos-organizer).
I worked through it with an AI tutor over many sessions: planning,
building, and asking questions until I understood each part.

## Related

[chaos-organizer](https://github.com/ffairylea/chaos-organizer) is
the applied side: using existing ML tools to solve a workflow
problem. This one is the fundamentals side: building the machinery
to understand what those tools are doing.
