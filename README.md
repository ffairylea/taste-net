# taste-net

A small neural network, built entirely from raw numpy (no PyTorch,
no scikit-learn), trained to predict whether I would save an image
— using only my own labeled Pinterest-style data.

This isn't trying to solve "aesthetics" or build a general taste
model. It's a scoped, honest question: can a simple model, trained
only on my choices, predict my choices — and where does it fail?

## Why this exists

I built [chaos-organizer](https://github.com/ffairylea/chaos-organizer)
using `sentence-transformers` for text embeddings — a real library
doing real work I didn't fully understand under the hood. This
project is the opposite move: build the smallest honest version of
"a model that learns preference" myself, from the math up, so I
actually understand what a library like that is doing when I use one.

No pretrained embeddings, no clip, no transfer learning. Every
feature the model sees is a number I computed by hand (see
`src/features.py`) — brightness, saturation, contrast, color
histograms. Every step of learning is backpropagation and gradient
descent I wrote myself (see `src/model.py`).

## What's actually happening, mathematically

- Image → vector: each image becomes a point in 32-dimensional
  space (5 color statistics + 3 average channel values + a 24-bin
  histogram). This is the same idea as a text embedding, just with
  hand-built features instead of a trained encoder.
- Learning = gradient descent: the network makes a guess, measures
  how wrong it was (cross-entropy loss), and uses calculus (the chain
  rule, applied layer by layer — backpropagation) to figure out which
  direction to nudge every weight to be less wrong next time.
- The real question: is "my taste" *linearly separable* in this
  feature space, or does it need the hidden layer's non-linearity to
  be captured at all? The project can actually answer this — try
  removing the hidden layer and compare accuracy.

## Phase 2: which board, not just yes/no

Stage 1 (built) asks "would I save this at all." Stage 2 asks the
narrower question: given that I would, *which* board does it belong
to — pink, winter-minimal, and so on.

Two stages rather than one model with N+1 classes, because those are
genuinely different questions. "Is this my vibe" is a coarse gate;
"which specific board" is a fine distinction made only after the
gate passes. Stage 2 never sees rejected images at all.

Mathematically the second stage is not new machinery. Sigmoid
becomes softmax, binary cross-entropy becomes categorical
cross-entropy, and the label goes from a single 0/1 to a one-hot
vector. Everything below the output layer — the ReLU hidden layer,
the backward pass through `W1`, the gradient descent update — is
character-for-character identical to the binary model. Reading
`src/model_multiclass.py` beside `src/model.py` makes that visible.

The reason it collapses so cleanly is that the gradient
simplification carries over. For sigmoid + binary cross-entropy the
chain rule cancels down to `dLoss/dZ2 = ŷ - y`. For softmax +
categorical cross-entropy it cancels to the same thing, with both
sides now vectors instead of scalars. Sigmoid is the two-class case
of softmax, so this isn't a second lucky coincidence — it's one
identity showing up twice. Derivation in
`PHASE2_MULTIBOARD_PLAN.md`.

### Verifying the derivation before collecting any data

Deriving a gradient by hand and *being right* are different things,
so `src/train_boards.py` checks itself. With no board images present
it runs two tests instead of training:

1. **Numerical gradient check.** Nudge one weight by ±h, re-run the
   loss both times, and take `(loss(w+h) - loss(w-h)) / 2h`. That's
   the definition of a derivative with no calculus involved. If it
   matches the analytic gradient from the chain-rule code, both the
   math and the implementation are right. Worst disagreement across
   sampled weights: ~7e-12, which is float64 arithmetic noise rather
   than error.

2. **Synthetic training check.** Three well-separated Gaussian
   clusters, one per fake board. A working network should classify
   these near-perfectly; it hits 100%. If it couldn't, the bug would
   be in the code, not the data.

Output in `docs/evidence/`. A side detail worth noting: several
sampled weights return *exactly* zero from both methods. Those are
cases where the ReLU gate was closed (`Z1 <= 0`), so the weight had
no influence on the loss at all — nudging it changes nothing, and
the numerical check confirms independently what `relu_derivative`
asserts. Dead neurons get no partial credit, unlike sigmoid's
smooth derivative.

Having this pass before any images exist means a later failure is
attributable to data rather than to broken calculus.

## Status

- [x] Feature extraction pipeline
- [x] From-scratch neural net (forward + backward pass, by hand)
- [x] Training loop with train/test split
- [x] Real labeled data collected (279 "yes" images from my pink
      Pinterest board, 187 "no" images from Unsplash — see
      `LABELING_GUIDE.md`)
- [x] First real training run: 89.25% test accuracy. The class split
      (279 yes / 187 no) means always guessing "yes" would score
      60%, so the model is ~29 points above the trivial baseline.
      Feature importance analysis showed it leans heavily on
      blue-channel histogram bins, likely reflecting my pink
      aesthetic's narrow blue-channel signature
- [x] Full math documentation, restructured into sequenced files
      under `docs/` — from-scratch derivations of weighted sums,
      non-linearity, sigmoid (including the log-odds derivation),
      cross-entropy, and the complete backward pass, worked by hand
      with real numbers
- [x] Multi-board classification (softmax): Stage 2 model
      implemented from scratch, backward pass verified against
      numerical differentiation
- [ ] Board images collected for boards beyond pink
- [ ] First real multi-board training run + confusion matrix
- [ ] Ablation test: does removing blue-channel features tank
      accuracy, or is the model using more than just that shortcut?
- [ ] Comparison against a real pretrained model (e.g. CLIP) as a
      "how much am I losing by not using a real vision model" baseline

## The uncomfortable question this project doesn't dodge

If I train a model only on things I've already chosen, and it "learns
my taste," what has it actually learned? A few honest possibilities,
not resolved here, worth sitting with:

- It might just be learning to replay my existing biases back to
  me — confirming what I already like rather than discovering
  anything about aesthetics.
- "Taste" might not be a stable target at all — my Pinterest saves
  from six months ago and today might already disagree with each
  other, which would show up as noise the model can't fit.
- There's a real difference between a model that predicts what I'd
  click and a model that has anything resembling "taste" in the sense
  humans mean it — intentionality, context, mood, meaning. This
  project measures the first thing and should not be oversold as the
  second.
- If this ever got scaled up (bigger data, real vision models),
  a system that reliably predicts and reinforces individual aesthetic
  preference starts to look like the mechanism behind algorithmic
  feeds and their well-documented narrowing effects on what people
  see. A toy version of that mechanism is worth understanding
  precisely because the real version already shapes a lot of what
  people encounter online.

I don't think this project resolves any of that. I think it's small
enough to let me feel the shape of the question instead of just
having an opinion about it secondhand.

## Setup

```
pip install -r requirements.txt
```

See `LABELING_GUIDE.md` for how to collect your own `data/yes` and
`data/no` images, then:

```
cd src
python3 train.py
```

For Stage 2, images go in `data/boards/<board_name>/` — one folder
per board, discovered automatically by folder name:

```
cd src
python3 train_boards.py
```

With no `data/boards/` present this runs the gradient check and
synthetic training check described above instead of training.

## Related

Part of a two-project pair with
[chaos-organizer](https://github.com/ffairylea/chaos-organizer):
that one is the applied/product side (using existing ML tools to
solve a real workflow problem). This one is the fundamentals side
(building the ML itself from the ground up to understand what those
tools are actually doing).