# Next experiment: spatial features

> **Status.** Features implemented and checked on synthetic images;
> not yet run on the real boards, and not yet pre-registered. The
> draft predictions below get finalized and committed before
> `measure_spatial_dprime.py` touches real data, the same way
> `docs/15` was.

## The question

Fleurs vs pink sits at 59.3%, not significant. Every one of the 32
current features averages or bins over the whole image, so position
is discarded before the network sees anything (`docs/15`). If that
explanation is right, adding features that keep some spatial
information should close the gap, while color features alone stay
at chance on that split.

## Features, all computable in NumPy

Each one comes with a reason it should separate flowers from pink
interiors.

1. **Edge orientation entropy.** Run a Sobel filter, make a
   histogram of edge angles, and take its entropy. Rooms and
   furniture are mostly horizontal and vertical lines; petals curve,
   so their angles spread evenly. Sobel kernels are also roughly what
   a CNN's first layer learns on its own, so this builds that layer
   by hand while keeping it interpretable.
2. **Box-counting fractal dimension** of the edge map. Organic
   branching should score higher than rooms.
3. **Center vs border sharpness.** Variance of the Laplacian in the
   middle of the image against the border. Flower close-ups tend to
   have a sharp center and a blurred background; rooms are sharp
   throughout.
4. **Mirror symmetry.** Compare the image with its left-right flip.
   Crude, but computable and easy to explain.
5. **Coarse layout.** Mean brightness per cell of a 3×3 grid, the
   cheapest way to put position back in.

That is 4 + 9 = 13 new numbers. A 3×3 grid of full RGB means would
be 27 on its own, which is too many: with roughly 700 images, the
width sweep in `docs/15` already showed that extra capacity buys
memorization, not generalization. The target is about 12 additions.

Edge orientation over a coarse grid is close to the "gist"
descriptor of Oliva and Torralba (2001), which was motivated by how
quickly people categorize scenes at a glance. Worth reading before
building this.

## Design

1. **d′ per feature before any training.** In `docs/15` this was the
   strongest evidence, because nothing is fitted. If orientation
   entropy alone separates fleurs from pink with a d′ above 0.63
   (the best color feature), the result exists before a network
   runs.
2. **Pre-register and commit.** Proposed core claim: spatial
   features raise fleurs vs pink significantly above 50%, and color
   features alone stay at chance on that split.
3. **Ablation:** color only (the current baseline), spatial only,
   color + spatial. Spatial only is the interesting one. If it beats
   color on fleurs vs pink, that is close to a double dissociation:
   each feature set succeeds where the other fails. It may not be
   clean, though. winter_minimal probably has very few edges, so
   spatial features may separate it from pink as well, and the
   prediction should say so in advance.
4. **Multi-seed numbers,** with fleurs vs pink as the main metric.
   With about 90 test images on that split, the standard error is
   $\sqrt{0.5 \times 0.5 / 90} \approx 5.3$ points. Beating chance
   needs about 60%. Beating the color-only 59.3% convincingly needs
   roughly 70%, because the difference between two such estimates
   has a standard error of about 7.5 points.
5. **Then the pretrained-CNN comparison** already in the README:
   how much of the gap a learned representation closes beyond
   hand-built spatial features.

## Smallest useful version

Orientation entropy and its d′ on fleurs vs pink. About 30 lines of
NumPy, one number, and it says whether the idea is worth the rest.

## What is implemented

- `src/features_spatial.py`: the 13 features above, NumPy and
  Pillow only. Thresholds (12 orientation bins, edges = strongest
  15% of gradients, 128×128 greyscale) are fixed in the code now, so
  they can't drift toward whatever makes fleurs vs pink look good.
- `src/check_spatial_features.py`: runs the features on three
  synthetic images built to have known answers (a room of
  rectangles, a centered eight-petal flower on a blurred background,
  a recursive branching tree):

      feature                 room   flower   branching
      orientation_entropy    0.457    0.996       0.832
      fractal_dimension      1.551    1.645       1.732
      center_sharpness       1.065    1.901       0.172
      mirror_symmetry        0.263    0.996       0.978

  All four directional checks pass. The first run failed one: in an
  image with large flat areas, the 85th percentile of gradient
  magnitude is zero, so "at least the 85th percentile" marked every
  pixel as an edge. Fixed by also requiring a nonzero gradient.
- `src/measure_spatial_dprime.py`: step 1 of the design. d′ per
  feature on fleurs vs pink and pink-ish vs winter, next to the
  color features. Runs on the local board images; trains nothing.

Two things the synthetic run already says:

- **Box counting barely separates anything at this size.** The
  straight-line room scores 1.55, not 1, because a 128-pixel image
  only allows box sizes from 2 to 32 and Sobel edges are a few
  pixels thick. The flower's petal outline is a smooth curve, which
  has dimension 1 just like a straight line. Fractal dimension
  measures branching and texture, not curvature, so on close-up
  flowers it is the feature most likely to fail.
- **Orientation entropy is the strongest candidate.** 0.46 against
  1.00 is a large gap, and it measures exactly the straight vs
  curved distinction the hypothesis is about.

## Draft predictions (edit, then commit before running)

1. `orientation_entropy` alone separates fleurs from pink with d′
   above 0.63, the best color feature.
2. `fractal_dimension` is the weakest of the four scalar features on
   fleurs vs pink.
3. Spatial features also separate pink-ish from winter_minimal,
   because winter_minimal images have few edges. So the result will
   not be a clean double dissociation.
4. Spatial only, multi-seed, fleurs vs pink: above 60% (beats
   chance). Color + spatial: above the color-only 59.3%, but whether
   it clears the roughly 70% needed to show that convincingly is
   uncertain.

## Beyond this experiment: how much of taste is visual?

Color statistics are clearly a small part of taste, and the
interesting question is how small. Each step below adds a richer
kind of feature and measures how much more of my saves it explains.
The gaps between the steps are the result.

1. **Color** (done): palette, brightness, warmth.
2. **Spatial structure** (this file): edges, shape, layout.
3. **Spatial frequency and texture.** The amplitude spectrum of
   natural images falls off roughly as 1/f (Field, 1987), and its
   slope is one number from `np.fft`. Artworks across many cultures
   share roughly that falloff (Graham and Field, 2007; Redies,
   2007). Whether my saves sit closer to it than the images I skip
   is testable.
4. **Composition.** Where the brightest or busiest region sits
   (center, thirds, edge), and how much of the frame is empty.
5. **Learned features.** A pretrained vision model's embedding,
   which carries what the image *is* (a peony, a bedroom, a ski
   slope). This is the pretrained comparison already in the README.
6. **What no image feature can see.** Mood, context, novelty, why I
   saved something on a given day. This part has no feature, but it
   has a measurable ceiling: re-label a random 100 images blind,
   weeks after the originals, and count how often I agree with
   myself. If I agree 85% of the time, no model on any features can
   be expected to beat 85%, and the distance between step 5 and that
   ceiling is the part of my taste that isn't in the pixels.

Step 6 is cheap and probably the most informative thing on this
list: it turns "taste isn't just color" from an opinion into a
number.

## A note on symmetry

The 17 wallpaper groups classify *periodic* patterns, and a flower
photo is not periodic. A single flower has *point* symmetry: the
rosette groups $C_n$ (rotation only) and $D_n$ (rotation plus
mirrors), which are crystallographic point groups. Detecting those
in a messy photo is a research problem, not a quick feature, which
is why the mirror score above is deliberately crude.
