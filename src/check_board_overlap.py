"""
check_board_overlap.py

Finds images that appear in MORE THAN ONE board folder.

dedupe.py removes duplicates *within* a folder. This checks *across*
folders, which becomes important for Stage 2 in a way it wasn't for
Stage 1.

The reason: if the same pin ends up in data/boards/pink/ and also in
data/boards/winter_minimal/, the training data now contains the same
feature vector with two different labels. The model is being told
that one image belongs to two mutually exclusive classes. Softmax
cannot represent that — the outputs must sum to 1 — so the best it
can do is split its probability between them, which shows up as
apparent confusion between those two boards in the confusion matrix.

That would look like a genuine finding ("winter_minimal and pink are
hard to tell apart") when it is actually a data collection error.
Worth ruling out before drawing conclusions from the confusion
matrix.

Usage, from inside src/:
    python3 check_board_overlap.py
    python3 check_board_overlap.py --delete-from winter_minimal

Without --delete-from it only reports. With it, duplicates are
removed from the named board and kept everywhere else.
"""

import os
import sys
import hashlib
from collections import defaultdict

BOARDS_DIR = "../data/boards"
VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")


def file_hash(path):
    """MD5 of the file's bytes. Catches duplicates regardless of
    filename — but only byte-identical ones. The same pin downloaded
    at two different resolutions has different bytes and will NOT be
    caught here. See the note at the end."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    delete_from = None
    if "--delete-from" in sys.argv:
        i = sys.argv.index("--delete-from")
        if i + 1 >= len(sys.argv):
            raise SystemExit("--delete-from needs a board name")
        delete_from = sys.argv[i + 1]

    if not os.path.isdir(BOARDS_DIR):
        raise SystemExit(f"No {BOARDS_DIR}/ — nothing to check yet.")

    boards = sorted(
        d for d in os.listdir(BOARDS_DIR)
        if os.path.isdir(os.path.join(BOARDS_DIR, d))
    )
    if len(boards) < 2:
        raise SystemExit(
            f"Only {len(boards)} board folder(s) — overlap needs at least 2."
        )

    # hash -> list of (board, filename)
    locations = defaultdict(list)
    for board in boards:
        folder = os.path.join(BOARDS_DIR, board)
        n = 0
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(VALID_EXT):
                continue
            locations[file_hash(os.path.join(folder, fname))].append(
                (board, fname)
            )
            n += 1
        print(f"{board}: {n} images")

    overlaps = {h: locs for h, locs in locations.items() if len(locs) > 1}

    if not overlaps:
        print(f"\nNo cross-board duplicates found across {len(boards)} boards.")
        _resolution_note()
        return

    print(f"\n{len(overlaps)} image(s) appear in more than one board:\n")
    pair_counts = defaultdict(int)
    for locs in overlaps.values():
        names = ", ".join(f"{b}/{f}" for b, f in locs)
        print(f"  {names}")
        involved = sorted(set(b for b, _ in locs))
        for i in range(len(involved)):
            for j in range(i + 1, len(involved)):
                pair_counts[(involved[i], involved[j])] += 1

    print("\nBoard pairs affected:")
    for (a, b), c in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {a} <-> {b}: {c} shared image(s)")

    if delete_from is None:
        print("\nReporting only. To remove them from one board, rerun with:")
        print("    python3 check_board_overlap.py --delete-from <board_name>")
        print("\nDecide by hand which board each image genuinely belongs to —")
        print("an image saved to two boards may legitimately fit both, which")
        print("is itself worth knowing before treating boards as exclusive.")
        _resolution_note()
        return

    if delete_from not in boards:
        raise SystemExit(f"'{delete_from}' is not one of: {boards}")

    removed = 0
    for locs in overlaps.values():
        # Only delete if the image also survives in another board.
        others = [b for b, _ in locs if b != delete_from]
        if not others:
            continue
        for board, fname in locs:
            if board == delete_from:
                os.remove(os.path.join(BOARDS_DIR, board, fname))
                print(f"  removed {board}/{fname} (kept in {others[0]})")
                removed += 1

    print(f"\nRemoved {removed} file(s) from '{delete_from}'.")
    _resolution_note()


def _resolution_note():
    print("""
Note on what this does NOT catch: hashing compares exact bytes, so
the same pin downloaded at two different resolutions produces two
different hashes and will pass as distinct images. Catching those
needs perceptual hashing (comparing downscaled greyscale versions
rather than raw bytes), which is a reasonable extension if the
confusion matrix later shows two boards blending unexpectedly.""")


if __name__ == "__main__":
    main()
