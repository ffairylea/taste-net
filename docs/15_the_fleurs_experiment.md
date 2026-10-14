# 15 — The fleurs experiment: a prediction made before running it

This file is written **before** Stage 2 has been trained on real
board data. 

> **Status.** The prediction below stands as recorded. Everything
> from "## Result" onward was measured on the Imageye dataset that
> file 16 later found to be 67.9% near-duplicates, so those numbers
> are not usable. The clean re-run is in file 16 and the README. The
> prediction section is left exactly as committed, because editing a
> prediction after seeing the outcome would defeat the point of
> having made one.

## The setup

Three boards, collected from Pinterest via Imageye, deduplicated
within folders (`dedupe.py`) and across folders
(`check_board_overlap.py`):

| board | images |
|---|---|
| fleurs | 286 |
| pink | 275 |
| winter_minimal | 190 |

Cross-board duplicates: **zero**. This matters because an identical
image in two folders would give the model contradictory labels, and
the resulting confusion would look like a finding about the boards
when it was actually a data collection error. That failure mode has
been ruled out, so anything the confusion matrix shows is about the
features, not the file management.

Four overlapping files were found and removed during that check.
None were real pins — they were Pinterest interface assets
(`facebook_share_image.png` and a repeated placeholder) that Imageye
had scraped alongside the images.

## Why these three boards specifically

`winter_minimal` is visually distinct from `pink` — different
palette, different brightness, different saturation. Separating
those two should be well within reach of color-based features.

`fleurs` was chosen to be **hard on purpose**. Most of the flower
images are pink. But they belong to a different board, so a correct
model has to separate them from the `pink` board on some basis
other than "is it pink."

The question the experiment actually asks is therefore not "can the
model classify boards" but something sharper:

> Do the 32 hand-crafted features in `features.py` contain enough
> information to distinguish two categories that differ in *subject
> matter* while sharing a *color palette*?

## The prediction

**The model will separate winter_minimal well and fail on the
fleurs/pink distinction.**

Specifically, before running:

1. `winter_minimal` should be classified reliably — well above the
   33% chance baseline and above the 38% always-guess-fleurs
   baseline.
2. The largest off-diagonal cell in the confusion matrix should be
   **fleurs predicted as pink**. The direction matters: the claim is
   that pink flowers get absorbed into the pink board, not the
   reverse.
3. `fleurs` recall should be substantially worse than
   `winter_minimal` recall. If fleurs lands near chance, the
   prediction is strongly confirmed; if it is merely somewhat worse,
   partially.
4. Overall accuracy should sit somewhere between the baselines and
   the ~89–95% Stage 1 achieved — better than guessing, clearly
   worse than the binary task.

## Why this is expected

The 32 features (file 01) are, without exception, color statistics:
brightness, saturation, contrast, warmth, color variance, the three
mean channel values, and a 24-bin RGB histogram.

None of them encode shape, texture, edges, or spatial arrangement.
A feature vector cannot represent "there is a petal here" or "this
object is radially symmetric," because every number in it is
computed by averaging or binning pixel values over the whole image,
discarding position entirely.

So a pink flower and a pink bedroom, if their color distributions
are similar, produce **nearly identical feature vectors**. The model
is not failing to learn a distinction that is available to it — the
distinction is not present in its input at all.

This is a claim about representation, not about capacity. Adding
hidden neurons, training longer, or switching to Adam cannot fix it,
because none of those change what information reaches the network.

## What each outcome would mean

**If the prediction holds** — fleurs collapses into pink while
winter_minimal separates cleanly — the experiment demonstrates a
concrete limit of hand-crafted color features, using this project's
own data rather than a textbook assertion. It is the empirical case
for learned features: convolutional layers exist precisely because
hand-designed features cannot capture shape.

It would also make the two remaining items in the README
(comparison against a pretrained model, and the blue-channel
ablation) motivated rather than decorative. A CLIP comparison would
answer "how much of this gap closes with learned features," which is
a specific question rather than a vague one.

**If the prediction fails** — fleurs separates well — then something
in the color features does correlate with flowers, and the
interesting work becomes finding out what. Plausible candidates:
higher `color_variance` from petal-versus-background contrast, more
weight in green histogram bins from stems and leaves, or lower
`brightness` from outdoor lighting. Feature importance analysis on
the multiclass weights would identify it.

That outcome would be a genuine discovery about the data and more
interesting than the predicted one. It would also mean the reasoning
in this file is wrong in an instructive way: that "color features
cannot see flowers" was too quick, because flowers may carry a
color signature that is not the flower itself but its typical
context.

**If everything fails**, including winter_minimal, the explanation
is not about fleurs at all and something more basic is wrong —
class imbalance, a bug in the multiclass pipeline, or boards that
are less visually distinct than assumed.

## Baselines to beat

With three boards of these sizes, in the test split:

- random guessing: **33.3%**
- always predicting the largest board (fleurs): **38.1%**

Any accuracy claim gets compared against the second number, not the
first. Guessing the majority class is the trivial strategy a model
must beat to have done anything at all.

## What a model that *could* see flowers does differently

The prediction above rests on a claim about representation: shape is
absent from the input, not merely hard to learn from it. That claim
is worth making concrete by describing what a model with access to
shape actually does.

### The problem: averaging destroys position

Every one of the 32 features collapses the whole image into a single
number. `brightness` is the mean pixel value; the histogram bins
count how many pixels fall in each color range. Both discard
*where* anything is.

Once position is gone, questions like "is there an edge here" cannot
be asked, because *here* no longer exists as a concept. Two images
with identical color distributions and completely different
arrangements produce identical feature vectors. A pink flower and a
pink wall are, to this model, potentially the same input.

### Convolution: looking at small patches instead

A convolutional layer avoids that collapse by refusing to summarize
the whole image at once. It examines small patches, one position at
a time, with the same small grid of weights — a **kernel** — applied
at every location.

Take a 3×3 patch of a greyscale image sitting on a boundary between
dark and bright, such as the edge of a petal against sky:

$$\begin{bmatrix} 10 & 10 & 90 \\ 10 & 10 & 90 \\ 10 & 10 & 90 \end{bmatrix}$$

And a 3×3 kernel:

$$\begin{bmatrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{bmatrix}$$

The operation is: multiply matching positions, sum everything.

Row by row:

$$(10)(-1) + (10)(0) + (90)(+1) = -10 + 0 + 90 = 80$$
$$(10)(-1) + (10)(0) + (90)(+1) = 80$$
$$(10)(-1) + (10)(0) + (90)(+1) = 80$$

$$\text{total} = 240$$

A large output. The detector has fired: there is a vertical edge at
this position, dark on the left, bright on the right.

Now the same kernel on a flat patch with no edge:

$$\begin{bmatrix} 50 & 50 & 50 \\ 50 & 50 & 50 \\ 50 & 50 & 50 \end{bmatrix}$$

$$(50)(-1) + (50)(0) + (50)(+1) = 0 \quad\text{on every row}$$

$$\text{total} = 0$$

Nothing. And on a patch with the *opposite* edge — bright left, dark
right — the result is $-240$: same magnitude, opposite sign,
encoding direction as well as presence.

The kernel outputs a large value where the patch matches its pattern
and approximately zero where it does not. Sliding it across every
position produces a **map** of where that pattern occurs — position
preserved rather than averaged away.

### The kernel's weights are learned

The values $-1, 0, +1$ above were chosen by hand for illustration,
but in a real network they are not chosen at all. A 3×3 kernel is
nine weights, initialized randomly and trained by exactly the
machinery derived in files 07 and 13: compute $\partial L / \partial
w$ for each of the nine, step against the gradient.

Edge-detecting kernels emerge because kernels that detect edges
reduce the loss. Nobody specifies "look for edges" — that
arrangement of weights is simply what gradient descent finds.

This is the precise sense in which such a network *learns features*
rather than being handed them. The 32 features in `features.py` were
decided in advance by a person, and the model can only work with
what that person thought to compute. A convolutional layer decides
for itself what to measure.

### Stacking, and the limit of what this file can claim

One layer of edge detectors is not enough to represent a flower. The
standard account is that capability comes from composition: layer 1
detects edges, layer 2 detects combinations of edges such as corners
and curves, and later layers combine those into parts and objects.

**This is reported from the literature, not observed here.** No
convolutional network exists in this repository. Descriptions of
what intermediate layers "look for" come from published
feature-visualisation work, and are stated here as background rather
than as something this project measured. The honest claim is: this
is the mechanism I expect would address the fleurs/pink gap, and
testing it is future work.

What *is* established by the arithmetic above is narrower and does
not depend on any of that: convolution retains position, and the 32
features in `features.py` do not. Whether retaining position would
actually close this specific gap is an open question that a CLIP
comparison or a small convolutional layer could answer.

### Two properties worth noting

**The same kernel is used at every position.** Nine weights are
reused across the whole image rather than each location having its
own. This is called weight sharing, and it encodes an assumption:
an edge is an edge regardless of where it appears. A dense layer of
the kind used in `model.py` has a separate weight per input position
and must learn "edge" independently for every location.

**Convolution is still just weighted sums.** Multiply inputs by
weights, add them up — the same operation as file 03, applied to a
patch instead of a whole feature vector, and reused across
positions. It is not a different kind of mathematics; it is the same
arithmetic organized so that position survives.

## Result

Run on 2026-08-02, 500 epochs, plain gradient descent at $lr = 0.1$
(the same configuration as Phase 1 — deliberately, so the two are
comparable). 150 test images from a stratified 20% split.

**Final test accuracy: 67.33%** (train 76.21%)

Baselines: random 33.3%, always-guess-fleurs 38.0%.

### Confusion matrix

Rows are the true board, columns the prediction.

| | fleurs | pink | winter_minimal |
|---|---|---|---|
| **fleurs** | **35** | 18 | 4 |
| **pink** | 16 | **35** | 4 |
| **winter_minimal** | 5 | 2 | **31** |

Per class:

| board | recall | precision | test n |
|---|---|---|---|
| fleurs | 61.4% | 62.5% | 57 |
| pink | 63.6% | 63.6% | 55 |
| winter_minimal | 81.6% | 79.5% | 38 |

### Scoring the prediction

Four of the five specific claims held.

1. **winter_minimal classified reliably** — correct. 81.6% recall,
   far above both baselines.
2. **Largest off-diagonal cell is fleurs → pink** — correct. 18, the
   biggest single error cell.
3. **fleurs recall substantially worse than winter_minimal** —
   correct. 61.4% against 81.6%. Though "near chance" was too
   pessimistic; fleurs is clearly better than the 33% floor.
4. **Overall accuracy between the baselines and Phase 1's ~89%** —
   correct. 67.3% sits between 38.0% and 89.25%.
5. **The direction of the confusion** — **wrong**, and this is the
   most informative part of the result.

### Where the prediction was wrong

The claim was *absorption*: pink flowers being swallowed by the pink
board. That describes an asymmetric error — many fleurs misread as
pink, few pink misread as fleurs.

The measurement shows 18 fleurs → pink and 16 pink → fleurs.

**How much can be concluded from that?** Less than the first draft
claimed. It is 18 out of 34 off-diagonal errors, and a proportion of
18/34 = 0.53 carries a 95% interval of roughly [0.36, 0.70]. That
interval contains 0.5, so the data is consistent with symmetry — but
it also contains 0.70, so it is consistent with substantial
asymmetry. **The result cannot distinguish the two.**

What the data does support is the narrower claim: **no strong
absorption effect.** If pink flowers were being systematically
swallowed by the pink board, the errors would run heavily one way,
and they do not. The prediction of a strong directional effect is
not supported.

What it does not support is the stronger reading that the model has
"no purchase at all" on the distinction. Restricting attention to
the 112 fleurs and pink test images, the model gets 62.5% right
against 50% for a coin flip — which is $z \approx 2.65$,
significantly above chance. Some color signal separating these
boards does exist; it is simply weak.

(That 62.5% counts as errors the 8 images routed to winter_minimal,
which were not offered as options in a two-way framing. Restricted
to the 104 images the model actually placed in {fleurs, pink}, it is
70/104 = 67.3%. This document uses the more conservative 62.5%
throughout.)

The corrected claim is therefore: **fleurs and pink are weakly
separable by color statistics — significantly above chance, but far
below what the same features achieve on distinctions they can
express — with no strong directional bias in either direction.**

This is narrower than the first draft's "not separable in either
direction," which overshot the project's own measurement.

A note on the standard being applied here. This document states
later that a one-image difference between fleurs and pink recall
"should not be interpreted." The 18-versus-16 split is a
two-image difference, and the first draft built its central
narrative on it. The rigor was applied selectively, in the
direction that favored the conclusion — which is the same failure
file 14 documents, appearing again in a different form.

### The number that isolates the variable

Collapsing `fleurs` and `pink` into a single class — asking only
whether an image is pink-ish or winter_minimal, and ignoring the
distinction the features cannot express — gives:

**90.0% accuracy.**

That figure needs its baseline stated, and the first draft of this
file omitted it — which mattered, because the omission flattered the
result. Collapsed, the classes are 112 pink-ish against 38
winter_minimal, so **always guessing "pink-ish" scores 74.7%**. The
90.0% is therefore +15.3 points over trivial, while Phase 1's 89.25%
was +29.25 over its own 60% baseline.

So the two numbers are *not* "essentially identical" in any
meaningful sense, and an earlier version of this document claimed
they were. Measured against their respective baselines, the
collapsed task is the less impressive of the two.

The figure that survives this objection is **balanced accuracy** —
the mean of the per-class recalls, which is unaffected by class
sizes:

$$\frac{92.9\% + 81.6\%}{2} = 87.2\%$$

Both classes are recognized well above chance, and the model is not
simply riding the larger class. **87.2% is the number this section
should be read on.**

With that correction, the comparison still does the work it was
meant to do: the model separates pink-ish from winter_minimal
strongly while performing near chance on fleurs versus pink. The
following alternative explanations are argued against below, with
varying degrees of success — two of them are ruled out, and two are
not yet ruled out at all.

**Class imbalance — ruled out.** fleurs has 286 images and pink 275;
in the test split that is 57 against 55. There is no meaningful
imbalance between the two confused classes for imbalance to explain.
(Listing this as an eliminated alternative in the first draft was
close to vacuous, and inflated the apparent number of explanations
ruled out.)

**Data volume, in the crude sense — argued against, not ruled out.**
190 winter_minimal images support 81.6% recall while 286 fleurs
images support 61.4%, so the board with *fewer* examples is
classified *better*. That shows raw count is not the whole story,
but it confounds volume with difficulty and does not answer the
actual question: would more fleurs and pink images improve the
fleurs/pink boundary specifically? A learning curve — training on
25%, 50%, 75% and 100% of that data and checking whether the 62.5%
is rising or flat — would answer it. **Not yet run.**

**Model capacity — ruled out, but only after file 16.** The first
draft argued that the same network reaching 90% on a
color-expressible question shows width is sufficient. That is a non
sequitur: showing an easy question is answerable at a given width
says nothing about whether a harder one needs more.

The sweep this called for now exists in `probe_representation.py`.
On the contaminated data it appeared to *refute* this file — three-way
accuracy climbed to 95.2% at 128 hidden units. On clean data, five
seeds per width:

    width   train    test
        4   63.0%   54.2% +/- 2.0%
        8   66.3%   58.8% +/- 3.4%
       16   71.4%   58.5% +/- 2.0%
       32   77.7%   55.9% +/- 3.0%
       64   85.1%   56.9% +/- 3.8%
      128   91.5%   56.5% +/- 3.5%

Train accuracy climbs 28 points, test accuracy does not move outside
noise. Extra capacity buys memorization and nothing else, which is
what "the information is not in the input" predicts.

**Optimization — argued against, still not directly measured.** The
run above uses $lr = 0.1$, and the first draft argued that the loss
was still falling at epoch 500 as evidence optimization would not
help — which is an argument that it *would*. This is also
structurally the same error file 14 exists to document: a single
arbitrary hyperparameter per condition measures the hyperparameter,
not the method.

What file 14 now shows on clean data is that optimizer and learning
rate move the *binary* headline by at most 2.8 points (72.9% default
against a 75.7% best, which is two test images), and that every
setting which drives the training loss lower buys training accuracy
without buying test accuracy. That makes it unlikely the three-way
result is undertuning. But a swept-$lr$ three-way comparison is
still **not run**, so this is an inference from the sibling task
rather than a measurement of this one.

### What this means for the conclusion

Three of the four alternatives are now eliminated on clean data —
imbalance, capacity, and (indirectly) optimization — and the crude
data-volume argument is still an argument rather than a measurement.

There was also a deeper objection the first draft did not address at
all. Every argument here concerns the behavior of one network, while
the conclusion is about the *feature space*. Showing that this model
fails does not show that no model on these 32 features could
succeed.

`probe_representation.py` was written to close that gap, and the
honest summary of what it found is mixed. Two checks, clean data,
five splits each:

**kNN, fleurs vs pink** (chance 50.5%): k=1 64.1% ±2.5, k=5 63.9%
±2.9, k=15 55.4% ±3.8. A method with unlimited capacity peaks around
64%, in the same region as the network.

**Per-feature d′, no model involved at all:** fleurs vs pink has a
best single feature at d′ = 0.63 and a mean of 0.24; pink-ish vs
winter_minimal has a best of 0.88 and a mean of 0.38. The ratio of
means is 1.6×.

The d′ comparison is the stronger of the two, because it measures
the features directly with nothing fitted. The kNN result needs a
caveat the first version of this section would have skipped: on
`pink` vs `winter_minimal`, the pair the features *can* express, kNN
reaches only 69.0% while the network reaches 82.5% balanced
accuracy. kNN is therefore not a clean upper bound on this data — it
is simply a weaker classifier here, probably because plain Euclidean
distance over 32 z-scored color statistics is a poor metric. Its 64%
on fleurs/pink is consistent with the representation claim but
cannot establish it alone.

So: the conclusion survived the rebuild and survived a capacity
sweep designed to break it, and the model-free d′ numbers point the
same way. The thing that would settle it is still the comparison
against learned features, which is listed in the README as not done.

The mechanism, if it is right: `fleurs` and `pink` differ in subject
matter, subject matter is spatial, and all 32 features average or
bin over the whole image, discarding position before the network
sees anything.

### A detail the first draft read backwards

7.1% of fleurs and pink images were misrouted to winter_minimal,
while 18.4% of winter_minimal images were misrouted to one of the
other two. The first draft read this as the expressible boundary
being "respected in both directions."

It is the opposite. winter_minimal images are misrouted at roughly
2.6 times the rate, which is a bias toward the larger group —
exactly the class-prior effect dismissed a few paragraphs earlier.
The dismissal concerned fleurs versus pink, where the classes really
are balanced, but the collapsed comparison has 112 pink-ish against
38 winter, and that asymmetry does show up in the predictions.

### What this establishes

This is the empirical case for learned features, measured rather
than asserted. A convolutional network of the kind described above
would attack this exact gap: it retains position, composes edges
into curves and curves into parts, and can therefore represent
"radially arranged petal shapes" — a thing that has no expression in
a color histogram.

The remaining README items are now motivated by a specific number
rather than by general curiosity. A comparison against a pretrained
vision model would answer how much of the 62.5% → ceiling gap closes
when features are learned instead of hand-designed. The blue-channel
ablation asks a related question from the other side: how much of
the working 90% depends on a narrow color shortcut.

### Honest caveats

- 150 test images means one image is worth 0.67 percentage points.
  The 61.4% versus 63.6% gap between fleurs and pink recall is
  roughly one image and should not be interpreted.
- The 90.0% collapsed figure is not a model that was trained on the
  collapsed task. It is the three-way model's predictions,
  re-scored while ignoring one distinction. A model trained directly
  on two classes might do better or worse.
- Board membership is a label supplied by one person's filing
  decisions. Some flower images might reasonably belong to the pink
  board, which puts an unknown ceiling below 100% that has nothing
  to do with the model.