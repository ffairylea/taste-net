# 13 — Better update rules: momentum and bias correction

Files 07 and 08 ended with the simplest possible way to use a
gradient:

    w <- w - lr * g

Take the gradient, multiply by a step size, subtract. That rule is
what `model.py` originally used, and it works. This file is about
two things wrong with it and the standard fixes.

The forward pass, the backward pass, and the gradients themselves
are untouched by anything here. Only what gets *done* with the
gradients changes.

## First, what the gradient is

Worth restating precisely, because everything below is about
manipulating it.

For a single weight, the gradient is one number answering: **if this
weight increases slightly, does the loss go up or down, and how
fast?**

- gradient $+2.0$ → increasing this weight raises the loss at a rate
  of 2 → move the weight *down*, and by a meaningful amount
- gradient $-0.1$ → increasing it lowers the loss slightly → move the
  weight *up*, barely
- gradient $0$ → this weight doesn't affect the loss at all → leave
  it alone

The sign gives direction, the magnitude gives steepness. The minus
sign in $w \leftarrow w - lr \cdot g$ is what turns "the loss rises
in this direction" into "so go the other way."

File 11's gradient check demonstrates this definition directly: it
computes each gradient twice, once by the chain rule and once by
literally nudging the weight by $\pm h$ and measuring the change in
loss. Those agreed to twelve decimal places. The numerical version
*is* the definition; the calculus is a faster route to the same
number.

## And what the learning rate is

The gradient supplies a direction. The **learning rate** $lr$
supplies the distance:

$$w \leftarrow w - lr \cdot g$$

With a gradient of 2.0, $lr = 0.1$ moves the weight by 0.2, while
$lr = 0.5$ moves it by 1.0. Too small and training crawls; too large
and the update overshoots the minimum and bounces around it instead
of settling.

Both failure modes are visible in this project's own learning-rate
sweep (`compare_optimizers.py`, full tables in file 14):

| lr | final loss | train | test |
|---|---|---|---|
| 0.01 | 0.6317 | 63.8% | 65.7% |
| 0.1 | 0.5329 | 72.3% | 72.9% |
| 1.0 | 0.1846 | 93.3% | 64.3% |

At 0.01 the steps are so small that 500 epochs barely move the loss,
and both accuracies stay low. At 1.0 the loss drops to a third of
its value at 0.1 and training accuracy climbs 21 points — while test
accuracy *falls* by 8.6. The large steps are fitting the training
set and not generalizing, and the loss curve for that run in
`docs/evidence/optimizer_comparison.png` is visibly jagged rather
than smooth, which is overshooting made visual.

The learning rate is a **hyperparameter**: a number chosen before
training, as opposed to the weights, which are learned during it.
There is no way to compute the correct value in advance, which is
why finding it means sweeping a range and looking at what happens.

Adam, later in this file, is largely an answer to that problem — it
gives each weight its own effective step size, so the single global
guess matters much less.

## Problem 1: the gradient is a noisy estimate

The gradient is computed from the 282 training images — not
from some underlying truth about images in general. It's a sample
estimate, so it wobbles.

Suppose the "true" descent direction for some weight corresponds to
a gradient of 0.5, but successive measurements come out as:

$$0.6,\quad 0.4,\quad 0.55,\quad 0.45$$

Plain gradient descent steps on each of those directly: too far,
too short, too far, too short. Progress happens, but with a wasteful
zigzag layered on top.

## Momentum: average the gradient over time

Keep a running average instead of using each measurement raw:

$$m \leftarrow 0.9m + 0.1g$$

Read it as: keep 90% of what you already believed, mix in 10% of the
new measurement. This is an **exponential moving average** —
"exponential" because the influence of any single past gradient
decays by a factor of 0.9 every step, so old measurements fade
smoothly rather than being dropped abruptly.

Starting from $m = 0$ with the noisy sequence above:

$$m_1 = 0.9(0) + 0.1(0.6) = 0.06$$
$$m_2 = 0.9(0.06) + 0.1(0.4) = 0.094$$
$$m_3 = 0.9(0.094) + 0.1(0.55) = 0.1396$$
$$m_4 = 0.9(0.1396) + 0.1(0.45) = 0.1706$$

Climbing toward 0.5 smoothly. The overshoots and undershoots pull
against each other and partly cancel, while the consistent
underlying direction accumulates.

The name is a fair physical analogy: a ball rolling downhill doesn't
reverse instantly when the slope wobbles. It carries velocity from
where it has already been.

With $\beta = 0.9$, the average is dominated by roughly the last ten
gradients — $0.9^{10} \approx 0.35$, so a measurement's influence
decays to about a third over that span.

## Problem 2: the average starts out too small

Look again at those numbers. After four steps the average reads
0.17, but the true gradient is 0.5. That isn't the noise — it's a
systematic undercount, in the same direction every time, and it has
an obvious cause: $m$ was initialized to **0**.

Zero is not a neutral starting value. It is an assertion that the
gradient is zero. Every early update is therefore a compromise
between that false claim and the real measurements, weighted heavily
toward the false one, because at $\beta = 0.9$ each new measurement
only gets 10% of the say.

To see the effect without noise obscuring it, take a gradient that
is constant at **7** on every step:

$$m_1 = 0.9(0) + 0.1(7) = 0.7$$
$$m_2 = 0.9(0.7) + 0.1(7) = 1.33$$
$$m_3 = 0.9(1.33) + 0.1(7) = 1.897$$
$$m_4 = 0.9(1.897) + 0.1(7) = 2.4073$$

The gradient is 7 every single time, and the average reports 0.7,
then 1.33, then 1.9, then 2.4. It is dragging itself up out of the
zero it started from, and after four steps it is still nowhere near
the correct value.

### Why the magnitude matters

The step actually taken is $lr \cdot m$ — the learning rate times
the averaged gradient. So an $m$ of 0.7 when the truth is 7 produces
a step ten times smaller than intended. Training crawls during the
early epochs for a reason that has nothing to do with the loss
surface and everything to do with an initialization artifact.

This is not fatal — the average does converge eventually — but the
first several dozen updates are wasted, and those are precisely the
updates where the weights are furthest from good values and the
largest corrections are most useful.

## The bias has an exact algebraic form

The useful observation is that the undercount is not random. Compare
each value against the truth as a *fraction*:

$$\frac{0.7}{7} = 0.1 \qquad \frac{1.33}{7} = 0.19 \qquad \frac{1.897}{7} = 0.271 \qquad \frac{2.4073}{7} = 0.3439$$

Step 1 returns 10% of the true gradient, step 2 returns 19%, step 3
returns 27.1%.

Two things about those fractions matter.

**They don't depend on the gradient.** Running the same recursion
with a constant gradient of 0.03 gives $m_1 = 0.003$, and
$0.003 / 0.03 = 0.1$ — the same 10%. The fraction is a property of
the recursion and the step number, not of the data.

**They are exactly $1 - 0.9^t$:**

| step $t$ | $m_t$ (with $g=7$) | fraction of truth | $0.9^t$ | $1 - 0.9^t$ |
|---|---|---|---|---|
| 1 | 0.7 | 0.1 | 0.9 | 0.1 |
| 2 | 1.33 | 0.19 | 0.81 | 0.19 |
| 3 | 1.897 | 0.271 | 0.729 | 0.271 |
| 4 | 2.4073 | 0.3439 | 0.6561 | 0.3439 |

The third and fifth columns are identical. That is exact, not
approximate:

$$m_t = (1 - \beta^t)\,g$$

### Where that formula comes from

Spotting that two columns of a table agree is not a derivation. The
formula can be obtained directly from the update rule by
substitution.

Start with the rule, holding $g$ constant and using $\beta = 0.9$,
with $m_0 = 0$:

$$m_t = 0.9\,m_{t-1} + 0.1g$$

Substitute repeatedly, without simplifying, so the structure stays
visible:

$$m_1 = 0.9(0) + 0.1g = 0.1g$$

$$m_2 = 0.9(m_1) + 0.1g = 0.9(0.1g) + 0.1g$$

$$m_3 = 0.9(m_2) + 0.1g = 0.9\big(0.9(0.1g) + 0.1g\big) + 0.1g$$

Expanding that last one:

$$m_3 = 0.81(0.1g) + 0.9(0.1g) + 0.1g$$

Reordered smallest power first:

$$m_3 = 0.1g + 0.9(0.1g) + 0.81(0.1g)$$

Three terms, each being $0.1g$ multiplied by a successive power of
0.9 — namely $0.9^0$, $0.9^1$, $0.9^2$. Factoring $0.1g$ out:

$$m_3 = 0.1g\left(1 + 0.9 + 0.81\right)$$

The same structure holds at every step, with $t$ terms at step $t$:

$$m_t = 0.1g\left(1 + 0.9 + 0.9^2 + \ldots + 0.9^{t-1}\right)$$

Each term in that bracket is the previous one times 0.9, so it is a
**geometric series**, and the standard sum applies:

$$1 + r + r^2 + \ldots + r^{t-1} = \frac{1 - r^t}{1 - r}$$

With $r = 0.9$ the denominator is $1 - 0.9 = 0.1$:

$$1 + 0.9 + \ldots + 0.9^{t-1} = \frac{1 - 0.9^t}{0.1}$$

Substituting back:

$$m_t = 0.1g \cdot \frac{1 - 0.9^t}{0.1} = (1 - 0.9^t)\,g$$

The $0.1$ cancels — and that cancellation isn't luck. The $0.1$ in
the update rule *is* $1 - \beta$, and $1 - \beta$ is exactly what the
geometric sum divides by. The two are the same quantity, so they
always cancel, for any $\beta$.

Written generally:

$$m_t = (1-\beta)g\sum_{k=0}^{t-1}\beta^k = (1-\beta)g \cdot \frac{1-\beta^t}{1-\beta} = (1 - \beta^t)\,g$$

Checking at $t = 3$: the bracket gives
$1 + 0.9 + 0.81 = 2.71$, so $m_3 = 0.1g(2.71) = 0.271g$, and
$1 - 0.9^3 = 1 - 0.729 = 0.271$. They agree.

### Interpreting the result

So the EMA doesn't return the gradient. It returns the gradient
scaled down by a factor that can be written down precisely.

## Bias correction: divide the known factor back out

$$\hat m = \frac{m_t}{1 - \beta^t}$$

Checking against the table above, where the true gradient is 7:

$$t=1: \quad \frac{0.7}{0.1} = 7 \qquad t=2: \quad \frac{1.33}{0.19} = 7 \qquad t=4: \quad \frac{2.4073}{0.3439} = 7$$

Exact recovery of the true gradient, from the very first step.

Nothing about this is specific to 7. The $g$ cancels through the
algebra:

$$\frac{m_t}{1-\beta^t} = \frac{(1-\beta^t)\,g}{1-\beta^t} = g$$

Whatever the gradient is — 7, or 0.03, or different for every one of
the hundreds of weights in the network — the correction returns it
at full size. There is no target value being aimed at; the operation
simply undoes a known shrinkage.

And it disables itself automatically. As $t$ grows, $\beta^t \to 0$,
so the divisor approaches 1 and the correction stops changing
anything — correctly, since the startup bias has washed out by then.
With $\beta = 0.9$ it is essentially gone within a few dozen steps.

This is the part worth appreciating: it isn't a tuned fudge factor.
The startup error had an exact closed form, so it gets exactly
undone.

## Adam's second ingredient

## Adam's second ingredient: a per-weight step size

Momentum fixes the noise problem. It does nothing about a second
one: **a single learning rate has to serve every weight in the
network.**

Different weights can have gradients of very different sizes, so
whatever value of $lr$ gets chosen may be simultaneously too large
for some weights and too small for others.

**How much does this actually matter in this project?** Measured
after 500 epochs of training, across the 32 input features:

    gradient magnitude spread across features:  3.3x
    correlation between |W1| and |dW1|:         0.10

Only 3.3x, and weight magnitude is essentially *uncorrelated* with
gradient magnitude. An earlier draft of this file claimed that
heavily-weighted features — the blue-channel histogram bins the
model leans on, per the feature importance output in the README —
produce correspondingly large gradients. That claim was false, and
the reason it has to be false follows from file 12: every input
column is z-scored to unit variance before training, and since
$dW_1 = X^\top dZ_1 / n$, the input columns entering that product
all have comparable scale. Normalization has already removed most
of the disparity Adam's per-weight step size exists to handle.

Two things follow. First, large weight does not imply large
gradient — at convergence it often implies the opposite, since a
weight that has stopped moving has a small gradient by definition.
Second, the honest motivation for the second moment in *this*
project is not input feature scale. It is that gradient scales still
differ **across layers** ($W_1$ versus $W_2$ receive gradients
through different numbers of chain-rule factors) and across training
time, neither of which normalization addresses.

The general argument for adaptive step sizes stands. Its magnitude
here is modest, and file 14 measures what it is worth in practice
rather than assuming.

Adam's answer is to give every weight its own effective step size,
derived from its own history.

### Tracking magnitude requires squaring

To scale a weight's step by how large its gradients typically are,
that size has to be measured. The average gradient $m$ cannot supply
it, because $m$ tracks direction and direction cancels.

Consider a weight whose gradients alternate:

$$+3,\quad -3,\quad +3,\quad -3$$

These are large gradients. But averaged, they come to approximately
zero — the positives and negatives destroy each other. An optimizer
reading $m$ alone would conclude this weight has tiny gradients,
which is exactly wrong.

Squaring first:

$$9,\quad 9,\quad 9,\quad 9$$

Average 9. The sign is gone; the size survives.

This is the same maneuver used for variance in file 12, for the
same reason. Distances from a mean sum to zero by construction, so
they get squared before averaging. Here, gradients around a
direction cancel, so they get squared before averaging. Identical
problem, identical fix.

### The second moment

$$v \leftarrow \beta_2 v + (1-\beta_2)g^2$$

Structurally identical to the momentum update — an exponential
moving average — but fed $g^2$ instead of $g$, with its own decay
constant $\beta_2 = 0.999$.

The two are named by analogy with statistics: $m$ estimates the
first moment (the mean of the gradient), $v$ the second moment (the
mean of the squared gradient). Hence *Adaptive Moment Estimation*.

### Dividing by it

$$w \leftarrow w - lr \cdot \frac{\hat m}{\sqrt{\hat v} + \varepsilon}$$

Symbol by symbol:

- $w$ — a single weight; every weight is updated independently
- $lr$ — the global learning rate, as before
- $\hat m$ — the bias-corrected average gradient: *which direction*
- $\hat v$ — the bias-corrected average squared gradient: *typical
  size*
- $\sqrt{\phantom{v}}$ — converts $\hat v$ back from squared units
  into gradient units, so the division cancels properly
- $\varepsilon$ — $10^{-8}$, purely to prevent division by zero

Compare against the original rule:

$$w \leftarrow w - lr \cdot g$$

The only change is that $g$ has been replaced by
$\hat m / (\sqrt{\hat v} + \varepsilon)$: the direction, smoothed
and de-biased, divided by its own typical magnitude.

### Why this equalizes the step sizes

Two weights with very different gradient scales, both at
$lr = 0.01$:

**Weight A**, gradients around 10:

$$\hat v \approx 100 \quad\Rightarrow\quad \sqrt{\hat v} = 10 \quad\Rightarrow\quad \text{step} = 0.01 \cdot \frac{10}{10} = 0.01$$

**Weight B**, gradients around 0.1:

$$\hat v \approx 0.01 \quad\Rightarrow\quad \sqrt{\hat v} = 0.1 \quad\Rightarrow\quad \text{step} = 0.01 \cdot \frac{0.1}{0.1} = 0.01$$

Both move by 0.01, despite gradients differing by a factor of 100.
Each weight's step has been normalized by its own characteristic
gradient size.

This is why $lr$ matters much less under Adam than under plain
gradient descent — and it is the mechanism behind the reduced
learning-rate sensitivity measured in file 14.

### Why $\beta_2 = 0.999$ rather than $0.9$

$\beta_1 = 0.9$ averages over roughly the last 10 gradients;
$\beta_2 = 0.999$ over roughly the last 1000.

**Where "roughly 10" and "roughly 1000" come from.** The weight the
EMA assigns to the gradient from $k$ steps ago is
$(1-\beta)\beta^k$, so the total weight across all past steps is a
geometric series summing to 1 — the same series used for bias
correction above. The average *age* of the contributions is:

$$\sum_{k=0}^{\infty} k\,(1-\beta)\beta^k = \frac{\beta}{1-\beta}$$

For $\beta = 0.9$ that gives $0.9/0.1 = 9$ steps; for
$\beta = 0.999$, $0.999/0.001 = 999$. Hence the usual shorthand of
$1/(1-\beta)$ — 10 and 1000 respectively. The figures are not
arbitrary conventions; they are the mean lag of the average being
computed.

The asymmetry is deliberate. Direction should adapt quickly — when
the descent direction genuinely changes, the optimizer needs to
follow it. Magnitude should adapt slowly, because $\sqrt{\hat v}$
divides the step, so if it jittered from step to step, the effective
learning rate would jitter with it and updates would become erratic.

Fast for direction, slow for scale. That is the entire reason for
two separate decay constants.

### $v$ needs bias correction too

$v$ starts at zero, exactly like $m$, so the same undercount applies
and the same fix works:

$$\hat v = \frac{v}{1 - \beta_2^t}$$

It matters *more* here. With $\beta_2 = 0.999$:

$$1 - 0.999^1 = 0.001$$

Step one returns one thousandth of the true squared gradient.
Without correction, $\sqrt{\hat v}$ would be far too small, the
division would produce an enormous step, and training could diverge
in the first few updates. The correction that looked like a nicety
for $m$ is load-bearing for $v$.

### $\varepsilon$

$\sqrt{\hat v}$ can be exactly zero — a ReLU whose gate stays closed
produces a gradient of zero on every step it stays closed, so its
squared average stays at zero. Weights in that state appeared in the
gradient check output as entries reading exactly `0.00000000` from
both the analytic and numerical methods.

Worth being precise about what that output does and does not show:
it was produced on synthetic data at initialization, before any
training. It demonstrates that those weights were dead *at that
step*, not that they were permanently dead. Establishing permanence
would require tracking them across training, which has not been
done.

Adding $\varepsilon = 10^{-8}$ prevents the division from failing.
It plays the same defensive role as the `std == 0` guard in
`dataset.normalize` (file 12): a small constant that never affects
normal operation and prevents one specific degenerate case.

## What actually happened when this was measured

See `compare_optimizers.py` and
`docs/evidence/optimizer_comparison.png`. Results and the
methodology problem that surfaced are covered in file 14.