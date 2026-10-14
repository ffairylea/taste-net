# 16 — How 68% of the dataset turned out to be duplicates

Every accuracy figure this project reported before August 2026 was
inflated. The cause was not a bug in the model, the math, or the
training loop. It was in how the images were collected, and it went
undetected through fifteen documentation files, two training phases,
and an optimizer benchmark.

This file records how it was found, because the method generalizes
and the mistake does not.

## The trail

The claim in file 15 is that fleurs and pink cannot be separated by
color statistics. A reviewer made a fair objection: every argument
offered was about *one network's behavior*, while the conclusion was
about the *feature space*. Showing that a particular model fails
does not show that nothing could succeed.

The direct test is k-nearest-neighbours. kNN fits no parameters — it
memorizes the entire training set and classifies by proximity — so
it cannot be accused of insufficient capacity. If kNN also fails,
the information genuinely is not there.

`probe_representation.py` was written to run it. The result:

    kNN, fleurs vs pink
      k=1    97.7%
      k=5    78.8%
      k=15   73.2%

At first glance this refutes file 15 outright: 97.7% on a
distinction the network managed 62.5% on. The obvious reading is
that the features are fine and the network was too small.

## Why the pattern was wrong

1-NN classifies a test point by copying the label of the single
closest training point. It is the highest-variance classifier
available — one unlucky neighbor and the answer flips. Averaging
over 5 neighbors normally *helps*, because it smooths out exactly
that noise.

Here k=1 beat k=5 by 19 points. That ordering is backwards.

There is one common way it happens: **every test point has a
near-identical twin sitting in the training set.** Then 1-NN doesn't
generalize at all, it retrieves. The nearest neighbor is the same
image, so the label is guaranteed correct. Adding more neighbors
brings in genuinely different images and accuracy falls back toward
the honest value.

The 19-point gap was not a result. It was a symptom.

## The gap the existing checks left

Two deduplication scripts were already in the project, and both
missed it.

`dedupe.py` and `check_board_overlap.py` compare MD5 hashes of raw
file bytes. That catches files that are byte-for-byte identical and
nothing else. The same Pinterest pin downloaded twice at different
resolutions, or recompressed at a different JPEG quality, produces a
completely different byte stream and a completely different hash.

Both scripts said so in their own documentation. Neither acted on
it, and neither did I. The limitation was written down, read, and
left as a note.

## Perceptual hashing

Comparing images by content rather than bytes:

1. Convert to greyscale — discard color
2. Resize to 8×8 = 64 pixels — discard fine detail
3. Compare each pixel to the mean of all 64 → one bit each
4. Two images are near-identical if their 64-bit strings differ in
   only a few positions (Hamming distance ≤ 5)

The reasoning: resizing or recompressing an image barely disturbs
its coarse light/dark layout. Shrinking to 64 pixels throws away
precisely the detail that resolution changes affect, while keeping
the structure that identifies the image. Two genuinely different
photographs produce bit strings differing in roughly 32 of 64
positions — half, as expected by chance.

`check_near_duplicates.py` implements this. Validated on six
deliberately resized copies that MD5 had classified as distinct: all
six caught.

## The result

    boards/     751 images  ->  510 redundant  =  67.9%
    yes + no    466 images  ->  201 redundant  =  43.1%

Roughly two thirds of the board dataset was copies. The effective
size was 241 unique images, not 751.

One reassurance in the numbers: **zero groups spanned more than one
board.** The duplicates were redundancy within each class rather
than the same image carrying two different labels. The boards were
correctly filed; there were simply far fewer distinct images in them
than the file count suggested.

## Why duplicates inflate everything

A random train/test split assumes the two sets are independent. With
copies of the same image on both sides, that assumption fails:
memorizing a training image scores a free correct answer on the test
set.

Two consequences visible in the original numbers.

**Apparent overfitting was partly leakage.** The three-way run
showed 78.9% train against 48.9% test. That looked like a small
model overfitting a small dataset. On clean data the same
configuration gives 65.7% train and 66.4% test — a gap of 0.7
points. The gap was contamination, not overfitting.

**The width sweep reversed.** On contaminated data, three-way
accuracy climbed from 70.8% at 8 hidden units to 95.2% at 128 —
apparently proving that file 15 was wrong and capacity was the
binding constraint. On clean data the same sweep is flat: test
accuracy runs 54.2% to 58.8% across widths 4 to 128, every value
inside every other value's error bar, while train accuracy climbs
from 63.0% to 91.5%.

The mechanism is clear in hindsight. Extra capacity was not learning
better features; it was memorizing more twins. The 95.2% measured
the model's ability to recognize images it had already seen.

That is the most alarming part of this episode. The contamination
did not merely inflate a number, it produced a confident,
plausible-looking result pointing the opposite direction from the
truth.

## Where the duplicates came from

The images were collected with Imageye, a browser extension that
downloads every image on a page. Pinterest virtualizes its feed:
pins far from the viewport are unloaded from the DOM and re-rendered
on scroll, often at whatever resolution suits the current layout.

Collection therefore worked by scrolling a chunk, running the
extension, scrolling further, running it again. Each pass captured
whatever was currently rendered, which heavily overlapped the
previous pass — at a different resolution, producing different
bytes, defeating hash-based deduplication.

The extension worked correctly. The method was wrong.

## The fix: a source without the problem

Pinterest's Subject Access Request export contains every saved pin
as a database record, once. `parse_pinterest_export.py` reads it and
reconstructs each image URL from the `Image` hash field:

    https://i.pinimg.com/736x/<h[0:2]>/<h[2:4]>/<h[4:6]>/<h>.jpg

(The pattern was confirmed against the single record in the export
that happened to include a full canonical link.)

Downloaded files are named by their image hash, so re-downloading
the same pin overwrites rather than accumulates. **Duplicates become
structurally impossible rather than something to detect and remove.**

Near-duplicate rate on the rebuilt dataset: **6.8%**, down from
67.9%. The residual comes from pins that are genuinely different
uploads of visually similar images — a real property of the boards
rather than an artifact of collection.

The rebuild also fixed a separate problem. The original `data/yes`
came entirely from the pink board, which meant Phase 1 was arguably
learning "is this pink" rather than "would this be saved." The
export allows sampling across all 56 boards, making it the question
it was supposed to be. Phase 1 accuracy fell from 89.25% to 72.9%
against a 50% baseline — a lower number measuring a harder and more
honest question.

## What this changes about the conclusions

File 15's central claim survived. On clean data, fleurs versus pink
runs at 59.3% against a 50% baseline ($z = 1.78$, not statistically
significant) while pink-ish versus winter_minimal reaches 82.5%
balanced accuracy — with the same features, same network, same run.
The width sweep is flat, and the per-feature d′ separation is 1.6×
larger for the pair that works than for the pair that doesn't.
Representation, not capacity.

Two honest limits on that. The kNN probe is weaker evidence than it
first appears: on `pink` vs `winter_minimal` it reaches only 69.0%
where the network reaches 82.5%, so kNN is not a capacity-unlimited
*upper bound* on this data, just a differently-shaped classifier
that happens to be worse at the easy pair too. And the headline
three-way figure deserves a note of its own — `train_boards.py` at
its default split reports 66.4%, while the same architecture across
five splits in `probe_representation.py` averages 58.8% ±3.4. The
single-split number is roughly two standard deviations high. The
multi-seed figure is the one to quote.

The second of those is the same mistake as the duplicates in a
milder form: a number that looked better than it should have, left
unquestioned because it was the number already written down.

But the claim had to be re-established, because the evidence
originally offered for it was measured on unusable data. Being right
for the wrong reasons is not the same as being right.

## What generalizes

**A result that is too good is evidence about the measurement, not
the phenomenon.** k=1 outperforming k=5 by 19 points is not a
finding about features; it is a statement that something is wrong
with the data. The same instinct caught the rigged optimizer
benchmark in file 14: a sixfold speedup from swapping an update rule
on a two-layer network was not plausible, and it wasn't real.

**Documented limitations still need acting on.** Both existing
deduplication scripts stated in their own comments that they only
caught byte-identical files. Writing a limitation down is not the
same as addressing it, and a known gap left open is functionally
identical to an unknown one.

**Prevention beats detection.** Perceptual hashing found the
duplicates, but the real fix was changing the data source so they
could not occur. Hash-named files from a database export make the
failure mode structurally unavailable. That is a stronger guarantee
than any check run afterward.

**This class of bug is not a beginner's mistake.** Train/test
contamination through duplicate or near-duplicate images has
affected published benchmark datasets used by thousands of
researchers, in some cases for years before discovery. Finding it in
one's own data before publishing anything is the good outcome, not
the embarrassing one.