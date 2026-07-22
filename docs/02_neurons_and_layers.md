# 02 — Neurons and layers: the basic vocabulary

Before getting into the actual math, it's worth being precise about
two words that get used constantly from here on: neuron and layer.

## Neuron

A neuron is the smallest building block in the network. It takes in
some numbers, computes one weighted sum, applies one activation
function, and produces one output number. That's the entire job of
a single neuron — nothing more.

## Layer

A layer is a group of neurons sitting at the same "stage" of the
network. Every neuron in a layer receives the exact same inputs, but
each neuron has its own separate weights — so a layer produces
multiple outputs at once, one per neuron, all computed in parallel.

## In taste-net, specifically

- **Hidden layer** = 8 neurons (picked 8 as a reasonable middle-ground guess for ~450 images — not derived from a formula, worth testing other values later), side by side. Each one receives the
  same 29 features, but each has its own private set of 29 weights
  plus its own bias, so each computes its own separate raw score.
- **Output layer** = 1 neuron. It receives the 8 hidden layer
  outputs, combines them into one final score, and produces the
  model's final prediction.

The picture:

    29 features (x)
         |
         | (every one of these 29 numbers goes to EVERY neuron below)
         v
    [neuron1][neuron2][neuron3]...[neuron8]   <- hidden layer
         |       |        |            |
         v-------v--------v------------v
                [neuron]                      <- output layer
                    |
                    v
                 prediction

So: **layer** = a horizontal group, a processing stage. **Neuron** =
one individual unit inside that group. My network has 2 layers total
(hidden + output) but 9 neurons total (8 in the hidden layer, 1 in
the output layer).

## Why organize it this way?

Layers process in strict sequence — every neuron in the hidden layer
must finish before the output layer can start, since the output
layer needs the hidden layer's results as its input. This staged
structure is what makes the math tractable later on (clean, one
matrix multiplication per layer), and it's exactly what backprop
walks backward through, one layer at a time — which is where the
name "back-propagation" comes from.