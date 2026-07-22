# 06 — Cross-entropy loss: measuring how wrong a prediction was

By the end of the forward pass (files 03-05), the network produces
ŷ — a probability between 0 and 1. But a probability alone doesn't
tell us how to improve. We need a single number that measures
"how wrong was this specific guess, given what the true answer
actually was." That number is the loss.

## The formula

    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

Since y is always exactly 0 or 1 (a known fact, not a guess), only
one of the two terms is ever active per example — the other
multiplies by 0 and disappears:

- If y=1: L = -log(ŷ)
- If y=0: L = -log(1-ŷ)

## Why this specific formula, not something simpler?

A simpler-seeming option would be mean squared error: L = (ŷ-y)².
Cross-entropy is used instead because of what log(x) does as x
approaches 0: it shoots toward negative infinity. With the negative
sign out front, -log(small number) shoots toward positive infinity —
meaning this loss punishes CONFIDENT WRONG answers dramatically
harder than uncertain ones.

Concrete example: say y=1 (true answer is yes), and the model
confidently — and wrongly — predicts ŷ=0.01 (1% chance of yes):

    L = -log(0.01) ≈ 4.6

Compare to a less confident, still-wrong guess, ŷ=0.3:

    L = -log(0.3) ≈ 1.2

The loss isn't just a little bigger for the more-confident wrong
answer — it's almost 4x bigger, and the ratio gets more extreme the
closer ŷ gets to 0. This is a design choice that comes directly from the shape of the log function itself: log
near zero has an incredibly steep, punishing slope.

This steep penalty for confident mistakes is specifically what's
wanted for a yes/no classifier like this one — being very sure and
very wrong should cost more than being unsure and wrong. Mean squared
error doesn't have this property; it's the right tool for predicting
continuous numbers instead (like temperature), not for classification
problems like this one.

## What loss actually looks like on a real example

Using the running example from earlier files: ŷ ≈ 0.568, y = 1:

    L = -log(0.568) ≈ 0.566

Not near 0 (which would mean a great, confident, correct guess), not
huge either (which would mean a confident, wrong guess) — reflecting
"reasonable direction, but uncertain, plenty of room to improve."
This is the exact number the backward pass (file 07) starts from.