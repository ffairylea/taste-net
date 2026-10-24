# 15: The fleurs experiment: a prediction made before running it

The prediction in this file was written **before** Stage 2 was
trained on real board data.

> **Status.** The prediction below stands exactly as committed,
> because editing a prediction after seeing the outcome would defeat
> the point of having made one. The setup section describes the
> original Imageye dataset, which file 16 later found to be 67.9%
> near-duplicates. The Result section reports the clean re-run on
> the rebuilt dataset; the original run is kept at the end of the
> file.

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
a time, with the same small grid of weights (a **kernel**) applied
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

Nothing. And on a patch with the *opposite* edge (bright left, dark
right), the result is $-240$: same magnitude, opposite sign,
encoding direction as well as presence.

The kernel outputs a large value where the patch matches its pattern
and approximately zero where it does not. Sliding it across every
position produces a **map** of where that pattern occurs, with position
preserved rather than averaged away.

### The kernel's weights are learned

The values $-1, 0, +1$ above were chosen by hand for illustration,
but in a real network they are not chosen at all. A 3×3 kernel is
nine weights, initialized randomly and trained by exactly the
machinery derived in files 07 and 13: compute $\partial L / \partial
w$ for each of the nine, step against the gradient.

Edge-detecting kernels emerge because kernels that detect edges
reduce the loss. Nobody specifies "look for edges"; that
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
feature-visualization work, and are stated here as background rather
than as something this project measured. The defensible claim is: this
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
weights, add them up: the same operation as file 03, applied to a
patch instead of a whole feature vector, and reused across
positions. It is not a different kind of mathematics; it is the same
arithmetic organized so that position survives.

## Result

This experiment was run twice. The first run used the Imageye
dataset, which file 16 later showed was 67.9% near-duplicates. The
second run used the dataset rebuilt from Pinterest's official
export. **The clean run is the result.** The original run is kept at
the end of this file, because the way I first read it contained
several mistakes worth keeping visible.

Clean run: 500 epochs, plain gradient descent at $lr = 0.1$ (the same
configuration as Phase 1, deliberately, so the two are comparable).
Boards of 228 / 233 / 233 images, 137 test images from a stratified
20% split.

**Test accuracy: 66.4%** (train 65.7%) on the default split.

Baselines: random 33.3%, always-guess-the-largest-board 33.6%.

That 66.4% comes from the single split `train_boards.py` runs by
default. Across five splits, the same architecture averages
**58.8% ±3.4**, so the default split is about two standard
deviations lucky. The multi-seed figure is the one to quote. The
confusion structure below is stable across seeds; the headline is
not.

### Confusion matrix

Rows are the true board, columns the prediction.

| | fleurs | pink | winter_minimal |
|---|---|---|---|
| **fleurs** | **25** | 14 | 6 |
| **pink** | 9 | **29** | 8 |
| **winter_minimal** | **0** | 9 | **37** |

Per class:

| board | recall | precision | test n |
|---|---|---|---|
| fleurs | 55.6% | 73.5% | 45 |
| pink | 63.0% | 55.8% | 46 |
| winter_minimal | 80.4% | 72.5% | 46 |

`winter_minimal` was never classified as `fleurs`: zero out of 46.

### Scoring the prediction

Four of the five specific claims held.

1. **winter_minimal classified reliably.** Correct. 80.4% recall,
   far above both baselines.
2. **Largest off-diagonal cell is fleurs → pink.** Correct. 14, the
   biggest single error cell.
3. **fleurs recall substantially worse than winter_minimal.**
   Correct, but only partially by the rule I set in advance: 55.6%
   against 80.4% is clearly worse, but it is not near chance.
   Fleurs sits well above the 33% floor.
4. **Overall accuracy between the baselines and Phase 1's ~89%.**
   Correct as written: 66.4% sits between them. The comparison
   point itself did not survive, though. That ~89% was a
   contaminated number, and Phase 1 on clean data is 72.9%.
5. **The direction of the confusion.** Not supported.

### Where the prediction was wrong

The claim was *absorption*: pink flowers swallowed by the pink
board. That predicts an asymmetric error, many fleurs read as pink
and few pink read as fleurs.

The measurement: 14 fleurs → pink, 9 pink → fleurs. That is 14 of
23 off-diagonal errors, a proportion of 0.61 with a 95% interval of
roughly [0.41, 0.81]. The interval contains 0.5, so the data is
consistent with no asymmetry at all. It also reaches 0.8, so it
cannot rule out a strong one either. At 23 errors, the experiment
cannot tell the two apart, and the prediction of a strong
directional effect is not supported.

### The two numbers that isolate the variable

**fleurs vs pink.** Restricted to the 91 fleurs and pink test
images, the model gets 54 right: **59.3%** against 50% for a coin
flip. That is $z \approx 1.78$, not significant at the usual 0.05
level.

(That count treats the 14 images routed to winter_minimal as errors,
even though a two-way framing would not have offered that option.
Restricted to the 77 images the model actually placed in {fleurs,
pink}, it is 54/77 = 70.1%. This file uses the conservative 59.3%
throughout.)

**pink-ish vs winter_minimal.** Collapsing fleurs and pink into one
class, and asking only whether an image is pink-ish or
winter_minimal, gives 114 / 137 = 83.2%. That number needs its
baseline: the collapsed classes are 91 against 46, so always
guessing "pink-ish" already scores 66.4%. The figure that is not
affected by class sizes is **balanced accuracy**, the mean of the
two recalls:

$$\frac{84.6\% + 80.4\%}{2} = 82.5\%$$

So, from the same features, the same network and the same training
run: **82.5% on the distinction the features can express, 59.3% on
the one they cannot.**

### Ruling out the alternatives

**Class imbalance: ruled out.** The boards are 228 / 233 / 233, and
the test split is 45 / 46 / 46. There is no imbalance for imbalance
to explain.

**Data volume: argued against, not ruled out.** With the boards
this close in size, raw image count cannot explain why one board is
recognized better than another. It still does not answer the real
question: would *more* fleurs and pink images improve the
fleurs/pink boundary specifically? A learning curve (training on
25%, 50%, 75% and 100% of the data and checking whether the 59.3% is
rising or flat) would answer it. **Not yet run.**

**Model capacity: ruled out.** `probe_representation.py`, five
seeds per width:

    width   train    test
        4   63.0%   54.2% +/- 2.0%
        8   66.3%   58.8% +/- 3.4%
       16   71.4%   58.5% +/- 2.0%
       32   77.7%   55.9% +/- 3.0%
       64   85.1%   56.9% +/- 3.8%
      128   91.5%   56.5% +/- 3.5%

Train accuracy climbs 28 points and test accuracy does not move
outside noise. Extra capacity buys memorization and nothing else,
which is what "the information is not in the input" predicts.

**Optimization: argued against, not directly measured.** File 14
shows that optimizer and learning rate move the *binary* headline by
at most 2.8 points (72.9% default against a 75.7% best, two test
images), and that every setting which drives training loss lower
buys training accuracy without buying test accuracy. That makes
undertuning an unlikely explanation for the three-way result. But a
swept-$lr$ three-way comparison has **not been run**, so this is an
inference from the sibling task, not a measurement of this one.

### The features, with no model involved

Every argument above is about one network, while the conclusion is
about the *feature space*. Showing that this model fails does not
show that no model on these 32 features could succeed.
`probe_representation.py` was written to close that gap. The result
is mixed.

**Per-feature d′, nothing fitted:** fleurs vs pink has a best single
feature at d′ = 0.63 and a mean of 0.24. pink-ish vs winter_minimal
has a best of 0.88 and a mean of 0.38. The ratio of means is 1.6×.

**kNN, fleurs vs pink** (chance 50.5%): k=1 64.1% ±2.5, k=5 63.9%
±2.9, k=15 55.4% ±3.8. A method with unlimited capacity peaks around
64%, in the same region as the network.

The d′ comparison is the stronger evidence, because it measures the
features directly. The kNN result needs a caveat: on `pink` vs
`winter_minimal`, the pair the features *can* express, kNN reaches
only 69.0% while the network reaches 82.5%. So kNN is not a clean
upper bound on this data. It is a weaker classifier here, probably
because plain Euclidean distance over 32 z-scored color statistics
is a poor metric. Its 64% on fleurs/pink is consistent with the
representation claim but cannot establish it on its own.

### What this means for the conclusion

Imbalance and capacity are ruled out by measurement, optimization by
inference, and data volume is still an argument rather than a
measurement. The model-free d′ numbers point the same way. The
comparison that would settle it, against learned features, is
listed in the README as not done.

The mechanism, if it is right: `fleurs` and `pink` differ in subject
matter, subject matter is spatial, and all 32 features average or
bin over the whole image, discarding position before the network
sees anything.

One smaller detail. 15.4% of pink-ish images were misrouted to
winter_minimal, against 19.6% of winter_minimal images misrouted to
pink-ish. In the original run that gap was 2.6×, which I first read
the wrong way round (see below). On balanced clean data it shrinks
to 1.3×, which at these counts is inside noise.

### What this establishes

This is the empirical case for learned features, measured rather
than asserted. A convolutional network of the kind described above
would attack this exact gap: it keeps position, composes edges into
curves and curves into parts, and can therefore represent "radially
arranged petal shapes," which has no expression in a color
histogram.

The remaining README items are now motivated by a specific number
rather than general curiosity. A comparison against a pretrained
vision model would show how much of the 59.3% → 82.5% gap closes
when features are learned instead of hand-designed. The
blue-channel ablation asks a related question from the other side:
how much of the working 82.5% depends on a narrow color shortcut.

### Caveats

- 137 test images means one image is worth 0.73 percentage points,
  and on the 91-image fleurs/pink subset, 1.1 points. The 55.6%
  versus 63.0% gap between fleurs and pink recall is about three
  images and should not be interpreted.
- The single default split is about two standard deviations above
  the five-split mean. Every number in this section comes from that
  split, so treat them as describing structure, not as precise
  estimates.
- The 82.5% collapsed figure is not a model trained on the collapsed
  task. It is the three-way model's predictions, re-scored while
  ignoring one distinction. A model trained directly on two classes
  might do better or worse.
- Board membership is a label supplied by one person's filing
  decisions. Six near-duplicate groups span two boards, so some
  images really do belong to both. That puts an unknown ceiling
  below 100% that has nothing to do with the model.

## The original run, on contaminated data

Run on 2026-08-02, same configuration, on the Imageye boards
(286 / 275 / 190 images, 150 test images). Kept because how I first
read it went wrong in ways that are worth more than the numbers.

| | fleurs | pink | winter_minimal |
|---|---|---|---|
| **fleurs** | **35** | 18 | 4 |
| **pink** | 16 | **35** | 4 |
| **winter_minimal** | 5 | 2 | **31** |

Test accuracy 67.3% (train 76.2%). Restricted to fleurs and pink,
62.5%, with $z \approx 2.65$: apparently significant. On clean data
the same comparison is 59.3% and not significant, so some of that
apparent signal was duplicates leaking across the split.

The mistakes in the first draft's reading of this run:

- **A story built on a two-image difference.** The first draft
  built its central narrative on 18 fleurs → pink against 16 pink →
  fleurs, while a later paragraph of the same file said a one-image
  difference "should not be interpreted." The rigor was applied
  selectively, in the direction that favored the conclusion. That
  is the same failure file 14 documents, in a different form.
- **A headline without its baseline.** The collapsed pink-ish vs
  winter figure was reported as 90.0% and called "essentially
  identical" to Phase 1's 89.25%. Always guessing "pink-ish" scored
  74.7% on that split, so 90.0% was only +15.3 over trivial, against
  Phase 1's +29.25 over its own 60%. Measured against their
  baselines, the two were not comparable. Balanced accuracy (87.2%
  then, 82.5% now) is what I should have reported from the start.
- **A non sequitur about capacity.** The first draft argued that
  because the same network reached 90% on an easy question, width
  was sufficient for a hard one. Showing an easy question is
  answerable at some width says nothing about whether a harder one
  needs more. That is why the width sweep exists.
- **An argument that pointed the other way.** The first draft cited
  the loss still falling at epoch 500 as evidence that more
  optimization would not help. A loss still falling is evidence
  that it *might*.
- **A detail read backwards.** 7.1% of fleurs and pink images went
  to winter_minimal, against 18.4% of winter_minimal images going
  the other way. The first draft read this as the boundary being
  "respected in both directions." It was the opposite: a 2.6× bias
  toward the larger group (112 pink-ish against 38 winter), the
  same class-prior effect the draft had dismissed a few paragraphs
  earlier. On the balanced clean data it largely disappears, which
  fits that reading.

The width sweep on this data also climbed from 70.8% to 95.2%,
apparently proving that capacity *was* the limit. File 16 explains
why: extra width was memorizing duplicate twins.
