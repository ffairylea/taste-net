# Phase 2 plan: multi-board classification

> **Status.** This is the original plan, written before Phase 2 was
> built and kept as a record of the decisions. What was actually
> built differs in two places: three boards instead of four (`pink`,
> `winter_minimal`, `fleurs`), and the images came from Pinterest's
> official data export rather than Imageye, after file 16 found the
> scraped set was mostly duplicates. Results are in `docs/15` and
> the README.

## Architecture: two-stage (hierarchical)

Stage 1: existing binary yes/no model (already built): "does this
match my taste at all."

Stage 2: a SEPARATE model, only trained on/applied to images that
passed Stage 1 as "yes". It asks "given it's a yes, which board
specifically" (pink vs winter-minimal vs etc.), using softmax across
just the board classes. Stage 2 never sees "no" images at all.

Why two-stage over one single N+1-class model: mirrors how the
actual judgment works ("is this my vibe at all" is a coarser,
different kind of question than "which specific board"), and it's
more modular: each stage can be analyzed and improved separately.

## Data plan

4 boards total: pink (already have, 279 images) + 2-3 more,
~150-200 images each. Folder structure:

    data/
      pink/
      winter_minimal/
      [board3]/
      [board4]/

Images to be gathered later via the same Imageye method used for the
original pink board. No separate "none of these" class decided yet
for Stage 2, since Stage 1 already handles the yes/no gate.

## New math needed for Stage 2

- Softmax activation (replaces sigmoid for this stage):
  softmax(zi) = e^zi / sum(e^zj for all j)
- Categorical cross-entropy loss (generalizes binary cross-entropy
  to multiple classes)
- The backward pass derivation: softmax + categorical cross-entropy
  has its own version of the sigmoid+binary-cross-entropy
  cancellation, same elegant ŷ-y shape, but ŷ and y become whole
  vectors instead of single numbers. To be derived properly, same
  real-numbers-first style as files 01-09.

## Visualizations planned for results

1. Feature-space scatter plot: images as points (via PCA or two
   strongest features as axes), colored by predicted board. Visual,
   ties to the "taste as geometry" theme running through the docs.
2. Confusion matrix: shows where the multi-class model actually
   succeeds vs. where boards get confused with each other. The real
   evidence piece, not just aesthetic.

## Image display note

Pink board images (and other Pinterest-sourced boards) likely
shouldn't be publicly redisplayed on GitHub: unclear redistribution
rights, different from personal/private use for training. Unsplash-
sourced images (if a "none" class gets added later) are fine to
display publicly under their license. Visualizations (scatter plot,
confusion matrix) sidestep this entirely since they don't reproduce
the original images.