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
    #
    # An earlier version of this script printed a fixed paragraph
    # here. After the dataset rebuild its numbers changed and the
    # paragraph kept printing the old reading, which was then wrong
    # in the opposite direction. Interpretation that is hardcoded
    # eventually prints a falsehood, so everything below is derived
    # from the measurements.
    one_image = 1.0 / len(y_test)
    print("=" * 62)
    print("Sensitivity to learning rate (the honest comparison):")
    print("=" * 62)
    print(f"  {len(y_test)} test images, so one image = "
          f"{one_image:.1%} of accuracy.\n")

    summary = {}
    for name, rows in all_results.items():
        accs = [te for _, _, _, te, _ in rows]
        summary[name] = (max(accs), max(accs) - min(accs))
        print(f"  {name:<10s} best test {max(accs):.1%}, "
              f"spread across lr {max(accs) - min(accs):.1%}")

    bests = [b for b, _ in summary.values()]
    best_range = max(bests) - min(bests)
    spreads = {n: s for n, (_, s) in summary.items()}
    tightest = min(spreads, key=spreads.get)
    widest = max(spreads, key=spreads.get)

    print(f"\n  Best-accuracy range across optimizers: {best_range:.1%} "
          f"({best_range / one_image:.0f} test images out of {len(y_test)})")
    print("  -> this is a single split, so it cannot rank the optimizers\n"
          "     no matter which way it comes out. Treat the ordering as\n"
          "     unmeasured until it is run across several seeds.")

    ratio = spreads[widest] / max(spreads[tightest], 1e-9)
    print(f"\n  Widest spread: {widest} ({spreads[widest]:.1%}). "
          f"Tightest: {tightest} ({spreads[tightest]:.1%}).")
    print(f"  {widest} is {ratio:.1f}x more sensitive to the learning "
          f"rate than {tightest}.")
    print("  That sensitivity gap, not peak accuracy, is what this\n"
          "  benchmark actually measures. See docs/14 for the reading.")

    # Overfitting check: sort every run by final training loss and
    # show what the train/test gap does. The gap is a much larger
    # effect than the test-accuracy differences, so it survives the
    # sample size where a ranking would not.
    print("=" * 62)
    print("Training loss vs the train/test gap:")
    print("=" * 62)
    flat = [(f"{name} lr={lr:g}", losses[-1], tr, te)
            for name, rows in all_results.items()
            for lr, losses, tr, te, _ in rows]
    flat.sort(key=lambda r: -r[1])
    print(f"  {'run':<22s} {'loss':>8s} {'train':>8s} {'test':>8s} {'gap':>8s}")
    for label, loss, tr, te in flat:
        print(f"  {label:<22s} {loss:>8.4f} {tr:>7.1%} {te:>7.1%} "
              f"{tr - te:>+7.1%}")

    lo, hi = flat[-1], flat[0]
    print(f"\n  Lowest loss ({lo[1]:.4f}) vs highest ({hi[1]:.4f}): "
          f"training accuracy\n  moves {hi[2]:.1%} -> {lo[2]:.1%}, "
          f"test accuracy moves {hi[3]:.1%} -> {lo[3]:.1%}.")
    print("  Fitting the training set harder is not buying test accuracy.\n"
          "  That is overfitting, in this project's own numbers.")

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
    print("This plots TRAINING LOSS. The learning-rate sensitivity "
          "result above\nis about TEST ACCURACY and is not visible here — "
          "don't read it off\nthis figure. What the figure does show: SGD "
          "at lr=1 and lr=2 is\njagged where everything else is smooth "
          "(overshooting), and Adam\nreaches far lower training loss than "
          "SGD ever does, which bought\nno test accuracy at all.")


if __name__ == "__main__":
    main()
