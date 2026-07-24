# taste-net

A small neural network, built entirely from raw numpy (no PyTorch,
no scikit-learn), trained to predict whether I would save an image
— using only my own labeled Pinterest-style data.

This isn't trying to solve "aesthetics" or build a general taste
model. It's a scoped, honest question: **can a simple model, trained
only on my choices, predict my choices — and where does it fail?**

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

- **Image → vector**: each image becomes a point in ~29-dimensional
  space (color stats + a coarse histogram). This is the same idea as
  a text embedding, just with hand-built features instead of a
  trained encoder.
- **Learning = gradient descent**: the network makes a guess, measures
  how wrong it was (cross-entropy loss), and uses calculus (the chain
  rule, applied layer by layer — backpropagation) to figure out which
  direction to nudge every weight to be less wrong next time.
- **The real question**: is "my taste" *linearly separable* in this
  feature space, or does it need the hidden layer's non-linearity to
  be captured at all? The project can actually answer this — try
  removing the hidden layer and compare accuracy.

## Status

- [x] Feature extraction pipeline
- [x] From-scratch neural net (forward + backward pass, by hand)
- [x] Training loop with train/test split
- [x] Real labeled data collected (279 "yes" images from my pink
      Pinterest board, 187 "no" images from Unsplash — see
      `LABELING_GUIDE.md`)
- [x] First real training run: 89.25% test accuracy. Feature
      importance analysis showed the model leans heavily on
      blue-channel histogram bins, likely reflecting my pink
      aesthetic's narrow blue-channel signature
- [x] Full math documentation, restructured into sequenced files
      under `docs/` — from-scratch derivations of weighted sums,
      non-linearity, sigmoid (including the log-odds derivation),
      cross-entropy, and the complete backward pass, worked by hand
      with real numbers
- [ ] Ablation test: does removing blue-channel features tank
      accuracy, or is the model using more than just that shortcut?
- [ ] Comparison against a real pretrained model (e.g. CLIP) as a
      "how much am I losing by not using a real vision model" baseline
- [ ] Multi-board classification (softmax) — predicting which board
      an image belongs to, not just yes/no for one board

## The uncomfortable question this project doesn't dodge

If I train a model only on things I've already chosen, and it "learns
my taste," what has it actually learned? A few honest possibilities,
not resolved here, worth sitting with:

- It might just be learning to **replay my existing biases back to
  me** — confirming what I already like rather than discovering
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
python train.py
```

## Related

Part of a two-project pair with
[chaos-organizer](https://github.com/ffairylea/chaos-organizer):
that one is the applied/product side (using existing ML tools to
solve a real workflow problem). This one is the fundamentals side
(building the ML itself from the ground up to understand what those
tools are actually doing).
