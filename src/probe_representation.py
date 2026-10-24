"""
probe_representation.py

Tests the central claim of docs/15 - that the fleurs/pink failure is
a limit of the FEATURES rather than of the model.

The argument in docs/15 is about one network's behavior, but the
conclusion is about the feature space. Those are different claims. A
skeptical reader would say: you showed that this model, at this
width, with this optimizer, at this learning rate, can't separate
these boards. You haven't shown that nothing could.

This script runs three checks that address exactly that gap.

1. k-NEAREST NEIGHBOURS on fleurs vs pink.
   kNN is nonparametric - it fits no weights and has effectively
   unlimited capacity, memorising the entire training set. If a
   method with unlimited capacity ALSO lands near the network's
   62.5%, the ceiling is a property of where the points sit in
   feature space, not of the model. That is the difference between
   demonstrating the claim and asserting it.

2. WIDTH SWEEP on the three-way task.
   docs/15 originally argued capacity was ruled out because the same
   network reached 90% on an easier question. That is a non
   sequitur. This measures directly whether more hidden units help.

3. PER-FEATURE SEPARATION (d-prime).
   For each of the 32 features, how far apart are the fleurs and
   pink distributions, measured in pooled standard deviations?
   Compared against the same statistic for pink-ish vs winter. This
   looks at the features themselves with no model involved at all.

Run from inside src/:
    python3 probe_representation.py
"""

import numpy as np

from dataset import normalize
from dataset_boards import build_board_dataset
from features import FEATURE_NAMES
from model_multiclass import BoardNet
from train_boards import train_test_split_stratified

SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------
# 1. kNN - capacity-unlimited baseline
# ---------------------------------------------------------------

def knn_predict(X_train, y_train, X_test, k=5):
    """Standard k-nearest-neighbors, written out rather than
    imported, in keeping with the rest of the project.

    For each test point: find the k closest training points by
    Euclidean distance, take a majority vote of their labels. No
    training, no weights, no gradient descent. It simply memorises
    the training set.

    This matters for the argument: kNN cannot be accused of lacking
    capacity. If it fails too, the information isn't there."""
    preds = []
    for x in X_test:
        d = np.sqrt(((X_train - x) ** 2).sum(axis=1))
        nearest = y_train[np.argsort(d)[:k]]
        preds.append(np.bincount(nearest).argmax())
    return np.array(preds)


def run_knn_pair(X, labels, class_a, class_b, board_names, seeds=SEEDS):
    """kNN restricted to two boards, averaged over several splits."""
    mask = (labels == class_a) | (labels == class_b)
    Xp = X[mask]
    yp = (labels[mask] == class_b).astype(int)

    results = {k: [] for k in (1, 5, 15)}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xp))
        n_test = int(len(Xp) * 0.2)
        te, tr = idx[:n_test], idx[n_test:]

        Xtr, mean, std = normalize(Xp[tr])
        Xte, _, _ = normalize(Xp[te], mean, std)

        for k in results:
            acc = (knn_predict(Xtr, yp[tr], Xte, k=k) == yp[te]).mean()
            results[k].append(acc)

    print(f"\nkNN: {board_names[class_a]} vs {board_names[class_b]} "
          f"({mask.sum()} images, {len(seeds)} splits)")
    print(f"  chance (majority class): {max(yp.mean(), 1-yp.mean()):.1%}")
    for k, accs in results.items():
        accs = np.array(accs)
        print(f"  k={k:<3d} {accs.mean():.1%} +/- {accs.std():.1%}")
    return {k: np.mean(v) for k, v in results.items()}


# ---------------------------------------------------------------
# 2. Width sweep - is capacity the limit?
# ---------------------------------------------------------------

def width_sweep(X, Y, labels, widths=(4, 8, 16, 32, 64, 128), seeds=SEEDS):
    print(f"\nThree-way accuracy by hidden width ({len(seeds)} seeds each)")
    print(f"  {'width':>6s} {'train':>8s} {'test':>16s}")
    out = {}
    for h in widths:
        trs, tes = [], []
        for seed in seeds:
            (Xtr, Ytr, ltr, Xte, Yte, lte) = train_test_split_stratified(
                X, Y, labels, seed=seed
            )
            Xtr, mean, std = normalize(Xtr)
            Xte, _, _ = normalize(Xte, mean, std)

            net = BoardNet(n_features=X.shape[1], n_classes=Y.shape[1],
                           n_hidden=h, seed=seed)
            for _ in range(500):
                net.train_step(Xtr, Ytr, lr=0.1)
            trs.append((net.predict(Xtr) == ltr).mean())
            tes.append((net.predict(Xte) == lte).mean())
        trs, tes = np.array(trs), np.array(tes)
        out[h] = (tes.mean(), tes.std())
        print(f"  {h:>6d} {trs.mean():>7.1%} "
              f"{tes.mean():>10.1%} +/- {tes.std():.1%}")
    return out


# ---------------------------------------------------------------
# 3. Per-feature separation, no model involved
# ---------------------------------------------------------------

def d_prime(x_a, x_b):
    """Separation between two distributions in pooled standard
    deviations:

        d' = |mean_a - mean_b| / sqrt((var_a + var_b) / 2)

    d' near 0 means the two groups are indistinguishable on this
    feature. d' around 1 means their means differ by about one
    standard deviation - visibly separated but overlapping. This is
    a property of the DATA, computed without training anything."""
    pooled = np.sqrt((x_a.var() + x_b.var()) / 2)
    if pooled == 0:
        return 0.0
    return abs(x_a.mean() - x_b.mean()) / pooled


def separation_report(X, labels, board_names):
    fl = board_names.index("fleurs") if "fleurs" in board_names else 0
    pk = board_names.index("pink") if "pink" in board_names else 1
    wm = (board_names.index("winter_minimal")
          if "winter_minimal" in board_names else 2)

    d_hard = np.array([d_prime(X[labels == fl][:, j], X[labels == pk][:, j])
                       for j in range(X.shape[1])])
    pinkish = (labels == fl) | (labels == pk)
    d_easy = np.array([d_prime(X[pinkish][:, j], X[labels == wm][:, j])
                       for j in range(X.shape[1])])

    print("\nPer-feature separation (d-prime), no model involved")
    print(f"  fleurs vs pink      max {d_hard.max():.2f}  "
          f"mean {d_hard.mean():.2f}")
    print(f"  pink-ish vs winter  max {d_easy.max():.2f}  "
          f"mean {d_easy.mean():.2f}")
    print(f"  ratio of means: {d_easy.mean() / max(d_hard.mean(), 1e-9):.1f}x")

    print("\n  most separating features, fleurs vs pink:")
    for j in np.argsort(-d_hard)[:5]:
        print(f"    {FEATURE_NAMES[j]:16s} d'={d_hard[j]:.2f}")
    print("  most separating features, pink-ish vs winter:")
    for j in np.argsort(-d_easy)[:5]:
        print(f"    {FEATURE_NAMES[j]:16s} d'={d_easy[j]:.2f}")
    return d_hard, d_easy


def main():
    X, Y, labels, boards, paths = build_board_dataset("../data/boards")
    print(f"\nBoards: {boards}")

    fl, pk = boards.index("fleurs"), boards.index("pink")
    wm = boards.index("winter_minimal")

    knn_hard = run_knn_pair(X, labels, fl, pk, boards)
    knn_easy = run_knn_pair(X, labels, pk, wm, boards)

    widths = width_sweep(X, Y, labels)
    separation_report(X, labels, boards)

    print("\n" + "=" * 62)
    print("What this tells us")
    print("=" * 62)
    best_knn = max(knn_hard.values())
    best_net = max(v[0] for v in widths.values())
    print(f"""
kNN on fleurs vs pink peaks at {best_knn:.1%}. kNN has effectively
unlimited capacity - it memorizes every training point - so if this
is close to the network's 59.3%, the limit is in the features and
not in the model. If kNN does substantially better, the claim in
docs/15 is wrong and the network was simply too small or undertrained.

Three-way accuracy peaks at {best_net:.1%} across widths tested. If
that is close to the 58.8% the default width (8) reaches, capacity
was not the binding constraint. If it is much higher, docs/15
overstated its case and should be corrected.

The d-prime numbers are the cleanest version of the argument, since
no model is involved at all: they measure how far apart the two
groups sit in each feature dimension directly.""")


if __name__ == "__main__":
    main()
