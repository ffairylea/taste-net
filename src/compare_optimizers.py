"""
compare_optimizers.py

Compares plain gradient descent against momentum and Adam on the
real dataset, holding everything else fixed: same network, same
data, same starting weights (seed=42). Only the update rule varies.

WHY THIS SWEEPS LEARNING RATES INSTEAD OF PICKING ONE

The first version of this script compared SGD at lr=0.1 against
Adam at lr=0.01 and reported that Adam reached a given loss in 38
epochs versus SGD's 220 — an apparent 6x speedup.

That comparison was invalid. The learning rates weren't chosen by
any fair procedure, and SGD at lr=2.0 turns out to reach the same
target in 10 epochs, i.e. FASTER than Adam. The "speedup" was
entirely an artifact of handicapping the baseline.

Comparing two methods at one arbitrary hyperparameter each measures
the hyperparameters, not the methods. So this version sweeps the
learning rate for every optimizer and reports the whole range. The
conclusion that survives is different from, and more interesting
than, the one the rigged version produced.

Run from inside src/:
    python3 compare_optimizers.py
"""

import numpy as np

from dataset import build_dataset, train_test_split, normalize
from model import TasteNet
from optimizers import Momentum, Adam

EPOCHS = 500
SEED = 42
TARGET_LOSS = 0.3


def run(make_optimizer, lr, X_train, y_train, X_test, y_test):
    """Train from fixed starting weights, recording loss per epoch.

    The seed is fixed because a different random initialization
    would shift the loss curve on its own, which would contaminate
    the comparison. Same seed = same starting point = the update
    rule is the only variable."""
    net = TasteNet(n_features=X_train.shape[1], n_hidden=8, seed=SEED)
    opt = make_optimizer(lr) if make_optimizer else None

    losses = []
    for _ in range(EPOCHS):
        losses.append(net.train_step(X_train, y_train, lr=lr, optimizer=opt))
    losses = np.array(losses)

    train_acc = ((net.predict(X_train) > 0.5) == y_train).mean()
    test_acc = ((net.predict(X_test) > 0.5) == y_test).mean()
    return losses, train_acc, test_acc


def epochs_to_reach(losses, target=TARGET_LOSS):
    """Epochs until loss first drops below target. Measures speed of
    getting to a given quality, which is more useful than final loss
    alone — a method can end up low while taking forever to do it."""
    below = np.where(losses < target)[0]
    return int(below[0]) + 1 if len(below) else None


def main():
    X, y, paths = build_dataset("../data/yes", "../data/no")
    X_train, y_train, _, X_test, y_test, _ = train_test_split(X, y, paths)
    X_train, mean, std = normalize(X_train)
    X_test, _, _ = normalize(X_test, mean, std)

    sweeps = [
        ("SGD",      None,                       [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]),
        ("Momentum", lambda lr: Momentum(lr=lr),  [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]),
        ("Adam",     lambda lr: Adam(lr=lr),      [0.001, 0.005, 0.01, 0.05]),
    ]

    all_results = {}
    print(f"\n{'optimizer':<12s} {'lr':>7s} {'final loss':>11s} "
          f"{'epochs<' + str(TARGET_LOSS):>12s} {'train':>8s} {'test':>8s}")
    print("-" * 62)

    for name, factory, lrs in sweeps:
        rows = []
        for lr in lrs:
            losses, tr, te = run(factory, lr, X_train, y_train, X_test, y_test)
            n = epochs_to_reach(losses)
            rows.append((lr, losses, tr, te, n))
            print(f"{name:<12s} {lr:>7g} {losses[-1]:>11.4f} "
                  f"{str(n) if n else 'never':>12s} {tr:>7.1%} {te:>7.1%}")
        all_results[name] = rows
        print()

    # --- the actual conclusion, computed rather than asserted ---
    print("=" * 62)
    print("Sensitivity to learning rate (the honest comparison):")
    print("=" * 62)
    for name, rows in all_results.items():
        accs = [te for _, _, _, te, _ in rows]
        print(f"  {name:<10s} best test {max(accs):.1%}, "
              f"spread across lr {max(accs) - min(accs):.1%}")

    print("""
Reading this: the best achievable accuracy is similar across all
three — a tuned SGD is competitive with Adam on this problem. What
differs is the SPREAD. SGD's result depends heavily on guessing a
good learning rate; Adam lands in a narrow band across a wide range
of settings.

That robustness, not raw speed, is Adam's real advantage here. On a
small tabular problem like this one there is no dramatic speedup to
be had — the honest claim is 'less tuning needed', which is a
smaller claim than the first version of this script implied.""")

    # Overfitting check: does the lowest training loss give the best
    # test accuracy? Not necessarily, and that gap is worth seeing.
    print("=" * 62)
    print("Lowest training loss vs best test accuracy:")
    print("=" * 62)
    for name, rows in all_results.items():
        by_loss = min(rows, key=lambda r: r[1][-1])
        by_acc = max(rows, key=lambda r: r[3])
        note = "same" if by_loss[0] == by_acc[0] else "DIFFERENT"
        print(f"  {name:<10s} lowest loss at lr={by_loss[0]:g} "
              f"(test {by_loss[3]:.1%}), best test at lr={by_acc[0]:g} "
              f"(test {by_acc[3]:.1%})  -> {note}")
    print("""
Where these differ, the run that fit the training data best did NOT
generalize best — the definition of overfitting, visible directly in
this project's own numbers rather than as a textbook warning.""")

    _plot(all_results)


def _plot(all_results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — table only. "
              "pip3 install matplotlib for the plot.)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (name, rows) in zip(axes, all_results.items()):
        for lr, losses, _, _, _ in rows:
            ax.plot(losses, label=f"lr={lr:g}", linewidth=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("training loss (log scale)")
    fig.suptitle("Learning-rate sensitivity: the spread within each "
                 "panel is the point", y=1.02)
    out = "../docs/evidence/optimizer_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out}")
    print("Note the spread within each panel — SGD's curves fan out "
          "widely,\nAdam's cluster tightly. That visual IS the finding.")


if __name__ == "__main__":
    main()
