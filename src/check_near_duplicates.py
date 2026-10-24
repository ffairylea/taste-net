"""
check_near_duplicates.py

Tests whether the dataset contains NEAR-duplicates - the same image
at different resolutions, crops, or compression levels.

WHY THIS MATTERS

`dedupe.py` and `check_board_overlap.py` both compare MD5 hashes of
raw file bytes, which catches only byte-identical files. The same
Pinterest pin downloaded twice at different resolutions produces two
different byte streams and survives both checks. Both scripts say so
in their own documentation; this one acts on it.

The reason to care showed up in `probe_representation.py`:

    kNN fleurs vs pink:   k=1  97.7%
                          k=5  78.8%
                          k=15 73.2%

k=1 beating k=5 by 19 points is backwards. 1-NN is the highest
variance classifier there is - normally it loses to k=5. The only
common way it wins by this margin is if nearly every test point has
an almost identical twin sitting in the training set, in which case
1-NN just copies that twin's label.

If near-duplicates are split across train and test, every result
measured on that split is inflated, because the model can memorize
rather than generalize. That would affect the width sweep too: 95.2%
at 128 hidden units with 99.3% train accuracy is consistent with
memorization.

HOW PERCEPTUAL HASHING WORKS

Instead of hashing bytes, reduce the image to its coarse visual
structure and hash that:

  1. Convert to greyscale (throw away color)
  2. Resize to 8x8 = 64 pixels (throw away fine detail)
  3. Compare each pixel to the mean -> 64 bits, one per pixel
  4. Two images are near-identical if their bit strings differ in
     only a few positions (Hamming distance)

Resizing an image or recompressing it as JPEG barely changes the
coarse light/dark pattern, so the hash survives. A genuinely
different image produces a completely different pattern.

Usage, from inside src/:
    python3 check_near_duplicates.py
    python3 check_near_duplicates.py --threshold 8
"""

import os
import sys
import numpy as np
from PIL import Image
from collections import defaultdict

BOARDS_DIR = "../data/boards"
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_THRESHOLD = 5   # bits of difference out of 64


def phash(path, size=8):
    """Perceptual hash: 64 bits describing the coarse light/dark
    layout of the image."""
    img = Image.open(path).convert("L").resize(
        (size, size), Image.LANCZOS
    )
    a = np.asarray(img, dtype=np.float64)
    return (a > a.mean()).flatten()


def hamming(h1, h2):
    """How many of the 64 bits differ. 0 = visually identical,
    ~32 = unrelated (half the bits differ by chance)."""
    return int(np.sum(h1 != h2))


def main():
    threshold = DEFAULT_THRESHOLD
    if "--threshold" in sys.argv:
        threshold = int(sys.argv[sys.argv.index("--threshold") + 1])

    # --dir lets this run on any parent folder of image folders,
    # e.g. ../data (which contains yes/ and no/) rather than only
    # ../data/boards. Phase 1's data came from the same Imageye
    # process, so it needs the same check.
    base = BOARDS_DIR
    if "--dir" in sys.argv:
        base = sys.argv[sys.argv.index("--dir") + 1]

    if not os.path.isdir(base):
        raise SystemExit(f"No {base}/")

    boards = sorted(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    )

    entries = []   # (board, filename, hash)
    for b in boards:
        folder = os.path.join(base, b)
        n = 0
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(VALID_EXT):
                continue
            try:
                entries.append((b, f, phash(os.path.join(folder, f))))
                n += 1
            except Exception as e:
                print(f"  skipped {f}: {e}")
        print(f"{b}: {n} images")

    print(f"\nComparing {len(entries)} images pairwise "
          f"({len(entries)*(len(entries)-1)//2:,} comparisons), "
          f"threshold {threshold}/64 bits...")

    H = np.array([e[2] for e in entries])
    groups = defaultdict(list)   # representative index -> members
    assigned = {}

    for i in range(len(entries)):
        if i in assigned:
            continue
        groups[i].append(i)
        assigned[i] = i
        # vectorised Hamming against all later entries
        dists = (H[i + 1:] != H[i]).sum(axis=1)
        for offset, d in enumerate(dists):
            j = i + 1 + offset
            if j not in assigned and d <= threshold:
                groups[i].append(j)
                assigned[j] = i

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    n_extra = sum(len(v) - 1 for v in dup_groups.values())

    print(f"\n{len(dup_groups)} near-duplicate group(s), "
          f"{n_extra} redundant image(s) beyond one per group")
    print(f"That is {100*n_extra/len(entries):.1f}% of the dataset.")

    if not dup_groups:
        print("\nNo near-duplicates. The kNN k=1 result is not "
              "explained by leakage, and some other cause should be "
              "looked for.")
        return

    cross = 0
    for members in dup_groups.values():
        if len({entries[m][0] for m in members}) > 1:
            cross += 1

    print(f"{cross} group(s) span more than one board - those are "
          f"label contradictions, not just redundancy.")

    print("\nLargest groups:")
    for k, members in sorted(dup_groups.items(),
                             key=lambda kv: -len(kv[1]))[:8]:
        boards_in = [entries[m][0] for m in members]
        print(f"  {len(members)} copies  "
              f"[{', '.join(sorted(set(boards_in)))}]  "
              f"e.g. {entries[k][1][:48]}")

    print(f"""
INTERPRETATION

If a substantial fraction of the dataset is near-duplicates, then a
random train/test split places copies of the same image on both
sides. Any model with enough capacity can then score highly by
memorising rather than generalising, and every accuracy figure
measured on that split is inflated by an unknown amount.

The correct fix is a GROUP-AWARE split: assign every near-duplicate
group entirely to train or entirely to test, so no image has a twin
across the boundary. `probe_representation.py` and
`train_boards.py` both currently split at random, so their numbers
should be treated as upper bounds until this is redone.

To remove redundant copies instead (keeping one per group):
    python3 check_near_duplicates.py --threshold {threshold} --delete
""")

    if "--delete" in sys.argv:
        removed = 0
        for members in dup_groups.values():
            for m in members[1:]:
                b, f, _ = entries[m]
                os.remove(os.path.join(base, b, f))
                removed += 1
        print(f"Removed {removed} redundant image(s). "
              f"Rerun dedupe and retrain.")


if __name__ == "__main__":
    main()
