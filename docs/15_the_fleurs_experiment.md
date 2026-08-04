# 15 — The fleurs experiment: a prediction made before running it

This file is written **before** Stage 2 has been trained on real
board data. Nothing in it is informed by the result. 

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
None were real pins; they were Pinterest interface assets
(`facebook_share_image.png` and a repeated placeholder) that Imageye
had scraped alongside the images.

## Why these three boards specifically

`winter_minimal` is visually distinct from `pink` — different
palette, different brightness, different saturation. Separating
those two should be well within reach of colour-based features.

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

The 32 features (file 01) are, without exception, colour statistics:
brightness, saturation, contrast, warmth, colour variance, the three
mean channel values, and a 24-bin RGB histogram.

None of them encode shape, texture, edges, or spatial arrangement.
A feature vector cannot represent "there is a petal here" or "this
object is radially symmetric," because every number in it is
computed by averaging or binning pixel values over the whole image,
discarding position entirely.

So a pink flower and a pink bedroom, if their colour distributions
are similar, produce nearly identical feature vectors. The model
is not failing to learn a distinction that is available to it — the
distinction is not present in its input at all.

This is a claim about representation, not about capacity. Adding
hidden neurons, training longer, or switching to Adam cannot fix it,
because none of those change what information reaches the network.

## What each outcome would mean

**If the prediction holds** — fleurs collapses into pink while
winter_minimal separates cleanly — the experiment demonstrates a
concrete limit of hand-crafted colour features, using this project's
own data rather than a textbook assertion. It is the empirical case
for learned features: convolutional layers exist precisely because
hand-designed features cannot capture shape.

It would also make the two remaining items in the README
(comparison against a pretrained model, and the blue-channel
ablation) motivated. A CLIP comparison would
answer "how much of this gap closes with learned features."

**If the prediction fails** — fleurs separates well — then something
in the colour features does correlate with flowers, and the
interesting work becomes finding out what. Plausible candidates:
higher `color_variance` from petal-versus-background contrast, more
weight in green histogram bins from stems and leaves, or lower
`brightness` from outdoor lighting. Feature importance analysis on
the multiclass weights would identify it.

That outcome would be a genuine discovery about the data and more
interesting than the predicted one. It would also mean the reasoning
in this file is wrong in an instructive way: that "colour features
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




## What a model that could see fleurs does differently

The prediction above rests on a claim about representation: shape is
absent from the input, not merely hard to learn from it. That claim
is worth making concrete by describing what a model with access to
shape actually does.

### The problem: averaging destroys position

Every one of the 32 features collapses the whole image into a single
number. `brightness` is the mean pixel value; the histogram bins
count how many pixels fall in each colour range. Both discard
where anything is.

Once position is gone, questions like "is there an edge here" cannot
be asked, because 'here' no longer exists as a concept. Two images
with identical colour distributions and completely different
arrangements produce identical feature vectors. A pink flower and a
pink wall are, to this model, potentially the same input.

### Convolution: looking at small patches instead

A convolutional layer avoids that collapse by refusing to summarise
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
nine weights, initialised randomly and trained by exactly the
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

### Stacking is where shape comes from

One layer of edge detectors is not enough to see a flower. The
capability comes from composition:

- **Layer 1** takes pixels, outputs maps of edges at various
  orientations.
- **Layer 2** takes those edge maps as its input and detects
  combinations of edges — corners, junctions, curves. A curve is
  literally a particular arrangement of oriented edges.
- **Layer 3** combines curves into parts: a rounded lobe, a radial
  spoke pattern.
- **Later layers** combine parts into objects. A unit that fires on
  petal-like arrangements arranged radially around a centre is,
  functionally, a flower detector.

At no point does anyone write down what a flower is. The
representation is built by composing detectors, each layer treating
the previous layer's output as its input — the same layered
structure as this project's network, but with position retained at
every stage instead of discarded at the first.

That composition is why the prediction in this file is about
representation rather than capacity. A flower is not a colour. It is
a spatial arrangement, and spatial arrangement is exactly the thing
that averaging over pixels removes. No amount of extra hidden
neurons, longer training, or a better optimizer recovers information
that was destroyed before the network saw it.

### Two properties worth noting

**The same kernel is used at every position.** Nine weights are
reused across the whole image rather than each location having its
own. This is called weight sharing, and it encodes an assumption: an
edge is an edge regardless of where it appears. A dense layer of the
kind used in `model.py` has a separate weight per input position and
must therefore learn "edge" independently for every location.

**Convolution is still just weighted sums.** Multiply inputs by
weights, add them up — the same operation as file 03, applied to a
patch instead of a whole feature vector, and reused across
positions. It is not a different kind of mathematics; it is the same
arithmetic organised so that position survives.

## Result

Not yet run. To be filled in after `train_boards.py` executes on
this data, with the confusion matrix and an honest assessment of
which parts of the above prediction survived.