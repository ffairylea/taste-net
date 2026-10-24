"""
optimizers.py

Gradient descent, and two upgrades to it, written from scratch.

The models in model.py and model_multiclass.py originally hardcoded
the simplest possible update rule:

    W -= lr * dW

That rule treats every gradient as complete information and every
weight as needing the same step size. Neither is true. This file
implements the two standard fixes and the combination of them that
almost everything in modern ML actually uses.

All three optimizers share one interface: .step(params, grads)
mutates the parameters in place. That means swapping optimizers is
a one-line change in the training loop, which is what makes an
honest comparison possible.
"""

import numpy as np


class SGD:
    """Plain gradient descent - the rule already used in model.py,
    wrapped in the same interface so it can be compared fairly.

        w <- w - lr * g

    Every step is taken purely on the current gradient, with no
    memory of previous steps and no per-weight adjustment."""

    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g

    def reset(self):
        pass


class Momentum:
    """Gradient descent with an exponential moving average of the
    gradient.

        m <- beta*m + (1-beta)*g
        w <- w - lr*m

    The problem this solves: gradients computed on data are noisy
    estimates of the true descent direction. Stepping directly on
    each one produces zigzagging - progress in the consistent
    direction, plus random walk in every other direction.

    An EMA averages the noise out. Components of the gradient that
    point the same way every step accumulate; components that flip
    sign cancel. With beta=0.9 the average is dominated by roughly
    the last 10 gradients (since 0.9^10 ~ 0.35, contributions decay
    to about a third over that span).

    'Momentum' because the physical analogy is exact: a ball rolling
    downhill doesn't instantly change direction when the slope
    changes, it carries velocity from where it's already been."""

    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.m = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta * self.m[i] + (1 - self.beta) * g
            p -= self.lr * self.m[i]

    def reset(self):
        self.m = None


class Adam:
    """Adaptive Moment Estimation. Momentum, plus a per-parameter
    learning rate, plus a correction for the fact that both averages
    start wrong.

        m <- b1*m + (1-b1)*g          first moment  (mean of g)
        v <- b2*v + (1-b2)*g^2        second moment (mean of g^2)

        m_hat = m / (1 - b1^t)        bias correction
        v_hat = v / (1 - b2^t)

        w <- w - lr * m_hat / (sqrt(v_hat) + eps)

    THE SECOND MOMENT. v tracks the average SQUARED gradient, which
    is a measure of how large this particular weight's gradients
    typically are, regardless of sign. Dividing the step by
    sqrt(v_hat) means a weight whose gradients are consistently
    large gets a proportionally smaller step, and a weight with
    consistently tiny gradients gets a larger one. Every parameter
    ends up with its own effective learning rate, tuned
    automatically from its own history.

    This matters directly for this project: the 32 features have
    very different gradient scales even after normalization, so a
    single global lr is always a compromise. Adam removes the need
    for that compromise.

    THE BIAS CORRECTION. Both m and v start at zero, which is not a
    neutral starting point - it's an assertion that the gradient is
    zero, and it takes several steps to wash out. Concretely, with
    b1=0.9 and a constant true gradient g:

        after step 1:  m = 0.1*g        (10% of the truth)
        after step 2:  m = 0.19*g       (19%)
        after step 3:  m = 0.271*g      (27%)

    The pattern is m_t = (1 - b1^t) * g. That is exactly derivable:
    unrolling the recursion gives a geometric series that sums to
    (1 - b1^t). So dividing by (1 - b1^t) recovers g exactly.

        t=1:  1 - 0.9^1 = 0.1     ->  0.1*g / 0.1 = g
        t=2:  1 - 0.9^2 = 0.19    ->  0.19*g / 0.19 = g

    And as t grows, b1^t -> 0, so the correction factor -> 1 and
    quietly switches itself off. It isn't a fudge factor; it's the
    exact algebraic inverse of a known startup bias.

    EPSILON. sqrt(v_hat) can be zero (a weight whose gradient has
    been zero throughout - a permanently dead ReLU, for instance).
    eps prevents division by zero. It's the same defensive role as
    the std==0 guard in dataset.normalize."""

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1

        # Precompute the two correction denominators once per step
        # rather than per parameter - they depend only on t.
        c1 = 1 - self.beta1 ** self.t
        c2 = 1 - self.beta2 ** self.t

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2

            m_hat = self.m[i] / c1
            v_hat = self.v[i] / c2

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0


# Why beta2 = 0.999 rather than 0.9?
#
# The second moment is estimating a magnitude, and magnitudes should
# be stable - the typical size of a weight's gradient shouldn't
# swing wildly from step to step, or the effective learning rate
# would jitter. 0.999 averages over roughly the last 1000 steps,
# which is deliberately much longer than the ~10-step window of the
# first moment.
#
# Direction should adapt quickly; scale should adapt slowly. That
# asymmetry is the reason for two different betas, and it's the kind
# of choice that looks arbitrary until the question is asked.
