"""
predict.py

The two-stage architecture, actually running end to end on a single
new image it has never seen.

    Stage 1 (TasteNet, sigmoid)      "would I save this at all?"
              |
              |  only if yes
              v
    Stage 2 (BoardNet, softmax)      "which board does it belong to?"

Usage, from inside src/:

    python3 predict.py ../some_image.jpg
    python3 predict.py ../folder_of_images/          (whole folder)

This is where the design decision from PHASE2_MULTIBOARD_PLAN.md
becomes visible rather than theoretical: an image rejected at Stage 1
never reaches Stage 2 at all. The board model is never asked to
assign a board to something that isn't my taste in the first place -
it would happily do so (softmax always sums to 1, so it ALWAYS
returns a confident-looking distribution, even on garbage input),
which is exactly why the gate exists.

Stage 2 is optional: if board weights don't exist yet, this runs
Stage 1 alone and says so.
"""

import os
import sys
import numpy as np

from features import extract_features
from model import TasteNet
from model_multiclass import BoardNet

VALID_EXT = (".jpg", ".jpeg", ".png", ".webp")

STAGE1_WEIGHTS = "../taste_net_weights.npz"
STAGE2_WEIGHTS = "../board_net_weights.npz"
STAGE2_CLASSES = "../board_net_classes.txt"


def apply_normalization(x, mean, std):
    """Scale a feature vector using the mean/std FROM TRAINING.

    This is the part that's easy to get quietly wrong. The model
    learned its weights in a coordinate system defined by the
    training set's mean and std. A new image has to be mapped into
    that same system to be interpretable.

    Normalizing a single image against its own mean/std would give
    all zeros - every image identical, model output constant, no
    error raised. Silent failure. Hence the hard check below."""
    std = np.where(std == 0, 1, std)  # same guard as dataset.normalize
    return (x - mean) / std


def load_stage1():
    net = TasteNet(n_features=1)  # shape gets overwritten by load()
    mean, std = net.load(STAGE1_WEIGHTS)
    if mean is None or std is None:
        raise SystemExit(
            f"{STAGE1_WEIGHTS} has no saved normalization stats.\n"
            f"It was saved by an older version of train.py. Rerun:\n"
            f"    python3 train.py\n"
            f"to regenerate it with mean/std included."
        )
    return net, mean, std


def load_stage2():
    """Returns (net, mean, std, board_names) or None if Stage 2
    hasn't been trained yet."""
    if not (os.path.exists(STAGE2_WEIGHTS) and os.path.exists(STAGE2_CLASSES)):
        return None
    net = BoardNet(n_features=1, n_classes=2)  # overwritten by load()
    mean, std = net.load(STAGE2_WEIGHTS)
    if mean is None or std is None:
        raise SystemExit(
            f"{STAGE2_WEIGHTS} has no saved normalization stats. "
            f"Rerun: python3 train_boards.py"
        )
    with open(STAGE2_CLASSES) as f:
        boards = [line.strip() for line in f if line.strip()]
    if len(boards) != net.n_classes:
        raise SystemExit(
            f"Mismatch: weights have {net.n_classes} output classes but "
            f"{STAGE2_CLASSES} lists {len(boards)} boards. Retrain Stage 2."
        )
    return net, mean, std, boards


def predict_one(path, stage1, stage2, threshold=0.5):
    """Run one image through the pipeline. Returns a result dict."""
    net1, mean1, std1 = stage1
    x = extract_features(path)
    x1 = apply_normalization(x, mean1, std1).reshape(1, -1)

    p_yes = float(net1.predict(x1)[0])
    result = {"path": path, "p_yes": p_yes, "kept": p_yes > threshold}

    # The gate. Rejected images stop here - Stage 2 is never called.
    if not result["kept"] or stage2 is None:
        return result

    net2, mean2, std2, boards = stage2
    x2 = apply_normalization(x, mean2, std2).reshape(1, -1)
    probs = net2.predict_proba(x2)[0]
    order = np.argsort(-probs)

    result["board"] = boards[order[0]]
    result["board_confidence"] = float(probs[order[0]])
    result["board_distribution"] = [
        (boards[i], float(probs[i])) for i in order
    ]
    return result


def print_result(r, n_boards=None):
    name = os.path.basename(r["path"])
    print(f"\n{name}")
    print(f"  Stage 1  p(save) = {r['p_yes']:.3f}  ->  "
          f"{'KEEP' if r['kept'] else 'REJECT'}")

    if not r["kept"]:
        print("  Stage 2  skipped (didn't pass the gate)")
        return
    if "board" not in r:
        print("  Stage 2  not trained yet")
        return

    print(f"  Stage 2  {r['board']}  ({r['board_confidence']:.1%})")
    for board, p in r["board_distribution"]:
        bar = "#" * int(round(p * 30))
        print(f"           {board:<18s} {p:6.1%} {bar}")

    # Softmax always produces a full distribution, so a near-uniform
    # one is the model's way of saying "no idea" - worth surfacing
    # rather than hiding behind the argmax.
    if n_boards and r["board_confidence"] < 1.5 / n_boards:
        print("           (barely above chance - the boards look "
              "similar in feature space for this image)")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit("Give me an image path or a folder.")

    target = sys.argv[1]
    if not os.path.exists(target):
        raise SystemExit(f"No such path: {target}")

    if not os.path.exists(STAGE1_WEIGHTS):
        raise SystemExit(
            f"No {STAGE1_WEIGHTS} - train Stage 1 first: python3 train.py"
        )

    stage1 = load_stage1()
    stage2 = load_stage2()

    if stage2 is None:
        print("Stage 2 not trained yet - running Stage 1 only.\n"
              "(Collect board images into data/boards/<name>/ and run "
              "train_boards.py to enable board prediction.)")
        n_boards = None
    else:
        n_boards = len(stage2[3])
        print(f"Stage 2 boards: {stage2[3]}")

    if os.path.isdir(target):
        paths = [
            os.path.join(target, f) for f in sorted(os.listdir(target))
            if f.lower().endswith(VALID_EXT)
        ]
        if not paths:
            raise SystemExit(f"No images found in {target}")
    else:
        paths = [target]

    results = []
    for p in paths:
        try:
            r = predict_one(p, stage1, stage2)
            results.append(r)
            print_result(r, n_boards)
        except Exception as e:
            print(f"\n{os.path.basename(p)}\n  failed: {e}")

    if len(results) > 1:
        kept = sum(r["kept"] for r in results)
        print(f"\n{'-' * 40}")
        print(f"{kept}/{len(results)} images passed Stage 1 "
              f"({kept / len(results):.0%})")
        if stage2 is not None and kept:
            counts = {}
            for r in results:
                if "board" in r:
                    counts[r["board"]] = counts.get(r["board"], 0) + 1
            for board, c in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {board:<18s} {c}")


if __name__ == "__main__":
    main()