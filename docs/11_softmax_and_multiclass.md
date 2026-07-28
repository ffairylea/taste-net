# 11 — Softmax and multi-class classification

Files 01-09 covered binary classification: yes/no, one output
neuron, sigmoid. This file covers what changes for Phase 2 —
predicting WHICH board an image belongs to, among several options.

## The problem softmax solves

Instead of one output neuron, Stage 2 has $N$ output neurons — one
raw score per board. Example with 3 boards, one image:

$$z = [z_1, z_2, z_3] = [2.0, 0.5, -1.0]$$

We want to turn these 3 raw numbers into 3 probabilities that sum to
exactly 1 — "70% pink, 25% winter-minimal, 5% board 3." Sigmoid can't
do this directly — it treats each number independently, with no
guarantee multiple sigmoid outputs would sum to 1 together.

## The formula, worked with real numbers

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

**Step 1 — exponentiate every score:**

$$e^{2.0} \approx 7.389, \quad e^{0.5} \approx 1.649, \quad e^{-1.0} \approx 0.368$$

Why exponentiate at all, instead of using raw scores directly? Two
reasons: exponentials are always positive regardless of $z$'s sign,
turning even negative scores into valid positive numbers to work
with. And exponentials amplify differences — a gap of 2.0 vs 0.5 in
raw scores becomes a much bigger gap (7.389 vs 1.649) once
exponentiated, so softmax naturally makes the model's output more
decisive, letting the biggest score dominate more than plain
averaging would.

**Step 2 — sum them all up:**

$$S = 7.389 + 1.649 + 0.368 = 9.406$$

**Step 3 — divide each exponentiated score by that sum:**

$$\text{softmax}(z_1) = \frac{7.389}{9.406} \approx 0.786$$
$$\text{softmax}(z_2) = \frac{1.649}{9.406} \approx 0.175$$
$$\text{softmax}(z_3) = \frac{0.368}{9.406} \approx 0.039$$

Check: $0.786 + 0.175 + 0.039 = 1.000$ — exactly what's needed, a
genuine probability distribution across all boards. Interpretation:
78.6% confident pink, 17.5% winter-minimal, 3.9% board 3.

## Numerical stability — the max-subtraction trick

If $z$ values are very large (say 1000 vs 500), $e^{1000}$ would
overflow computationally. The standard fix: subtract the maximum $z$
value from all of them before exponentiating:

$$\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

This doesn't change the answer — subtracting the same constant from
every $z_i$ preserves the relative differences between them (which
is all softmax actually depends on), but guarantees the largest
exponent is always $e^0=1$ (never explodes), and every other exponent
is $e^{\text{negative number}}$ (always between 0 and 1). Real
numpy/PyTorch implementations of softmax do this internally by
default.

## Where this formula actually comes from

Not trial-and-error — this traces back to the **Boltzmann
distribution** in statistical mechanics (physics, 1800s), describing
the probability of a physical system occupying a given energy state:
literally the same mathematical form, $e^{-E_i/kT}$ divided by a sum
of all such terms. Machine learning borrowed this exact structure
because of its useful properties (always positive, sums to 1,
amplifies differences), not by inventing it from scratch. The name
"softmax" reflects that it's a smooth, differentiable version of
"argmax" (which would just pick the single biggest score with a hard
0/1, giving no smooth gradient to train with) — "soft" because it
produces a gradual probability distribution instead of a hard,
discontinuous winner-takes-all pick.

## Categorical cross-entropy — the natural extension

Recall binary cross-entropy: $L = -[y\log(\hat y) + (1-y)\log(1-\hat y)]$.
This worked because there were only 2 outcomes, so $y$ was a single
number.

With $N$ boards, $y$ becomes a **one-hot vector**: if the true answer
is pink (board 1):

$$y = [1, 0, 0]$$

(1 in the position of the true board, 0 everywhere else.)

The formula:

$$L = -\sum_i y_i \log(\hat y_i)$$

Plug in real numbers. Since $y = [1,0,0]$, only $y_1=1$ is nonzero —
every other term multiplies by 0 and vanishes, same simplification
trick as binary cross-entropy:

$$L = -[1\times\log(0.786) + 0 + 0] = -\log(0.786) \approx 0.241$$

This collapses to exactly the same shape as binary cross-entropy's
$-\log(\hat y)$ when $y=1$ — categorical cross-entropy isn't a
different idea, it's the same "penalize being wrong about the true
class" logic, extended to pick out whichever class actually happened
to be true, out of many possible classes instead of just 2.

If the model had been wrong instead — true board was actually board
3, $y=[0,0,1]$:

$$L = -[0+0+1\times\log(0.039)] = -\log(0.039) \approx 3.24$$

Much bigger loss — the model was 78.6% confident in the WRONG board,
and log's steep penalty near 0 punishes that confident wrongness
hard, same property as binary cross-entropy.

## The backward pass — softmax's own derivative, fully derived

Softmax's derivative is more involved than sigmoid's, because every
output $\hat y_i$ depends on ALL the $z$ values (the denominator sums
over every $z_j$), not just its own $z_i$. Two cases.

**Case A: $i=j$ (how $\hat y_i$ changes with respect to its OWN $z_i$)**

Write $\hat y_i = e^{z_i}\times S^{-1}$ where $S=\sum_k e^{z_k}$. Use
the product rule, $\frac{d}{dx}(uv)=u'v+uv'$, with $u=e^{z_i}$,
$v=S^{-1}$.

$u' = e^{z_i}$ (derivative of $e^{z_i}$ with respect to $z_i$ is
itself). $v' = -S^{-2}\times\frac{dS}{dz_i}$, and since only the
$k=i$ term inside $S$ depends on $z_i$, $\frac{dS}{dz_i}=e^{z_i}$, so
$v'=-S^{-2}\times e^{z_i}$.

Combine via product rule:

$$\frac{\partial \hat y_i}{\partial z_i} = \frac{e^{z_i}}{S} - \frac{(e^{z_i})^2}{S^2} = \hat y_i - \hat y_i^2 = \hat y_i(1-\hat y_i)$$

Exactly the same formula as sigmoid's derivative — not a coincidence.
Softmax is a generalization of sigmoid; sigmoid is literally what
softmax reduces to with only 2 classes.

**Case B: $i\ne j$ (how $\hat y_i$ changes with respect to a
DIFFERENT class's $z_j$)**

Same setup, but $e^{z_i}$ doesn't depend on $z_j$ at all, so the
first term in the product rule vanishes:

$$\frac{\partial \hat y_i}{\partial z_j} = e^{z_i}\times\left(-S^{-2}\times\frac{dS}{dz_j}\right)$$

Now $\frac{dS}{dz_j}=e^{z_j}$ (since $z_j$'s term inside $S$ is
$e^{z_j}$):

$$\frac{\partial \hat y_i}{\partial z_j} = -\frac{e^{z_i}\times e^{z_j}}{S^2} = -\frac{e^{z_i}}{S}\times\frac{e^{z_j}}{S} = -\hat y_i\hat y_j$$

Makes sense intuitively: since all probabilities must sum to 1, if
$z_j$ goes up (class $j$ becomes more likely), $\hat y_i$ must go
DOWN to compensate — less probability mass left for everyone else,
hence the negative sign.

## Combining with cross-entropy — the full cancellation, term by term

Loss is $L=-\sum_k y_k\log(\hat y_k)$. Want $\partial L/\partial z_i$
for one specific $z_i$. By chain rule, summing over every $k$:

$$\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial \hat y_k}\times\frac{\partial \hat y_k}{\partial z_i}$$

Using $\frac{\partial L}{\partial \hat y_k} = -\frac{y_k}{\hat y_k}$,
split into the $k=i$ term and every $k\ne i$ term.

**Part 1 — the $k=i$ term (using Case A):**

$$\left(-\frac{y_i}{\hat y_i}\right)\times\big(\hat y_i(1-\hat y_i)\big)$$

The $\hat y_i$ cancels top and bottom:

$$= -y_i(1-\hat y_i) = -y_i + y_i\hat y_i$$

**Part 2 — every $k\ne i$ term, summed (using Case B):**

$$\sum_{k\ne i}\left(-\frac{y_k}{\hat y_k}\right)\times(-\hat y_k\hat y_i)$$

The $\hat y_k$ cancels top and bottom, and the two negatives cancel:

$$= \sum_{k\ne i} y_k\hat y_i = \hat y_i\sum_{k\ne i}y_k$$

Since $y$ is one-hot, $\sum_k y_k = 1$, so $\sum_{k\ne i}y_k = 1-y_i$:

$$= \hat y_i(1-y_i) = \hat y_i - \hat y_i y_i$$

**Add Part 1 and Part 2:**

$$\frac{\partial L}{\partial z_i} = (-y_i + y_i\hat y_i) + (\hat y_i - \hat y_i y_i)$$

The middle two terms, $y_i\hat y_i$ and $-\hat y_i y_i$, are
identical with opposite signs — they cancel completely:

$$\frac{\partial L}{\partial z_i} = \hat y_i - y_i$$

The exact same shape as the binary case, now applied per-class,
across a whole vector at once. Softmax + categorical cross-entropy
was specifically designed so this cancellation happens — same
deliberate pairing logic as sigmoid + binary cross-entropy.

## Verify with real numbers

$\hat y = [0.786, 0.175, 0.039]$, true board was pink so
$y=[1,0,0]$:

$$\frac{\partial L}{\partial z} = [0.786-1,\ 0.175-0,\ 0.039-0] = [-0.214,\ 0.175,\ 0.039]$$

Matches exactly. Interpretation: $z_1$ (pink) should increase
(negative gradient — model wasn't confident enough), while $z_2$ and
$z_3$ should decrease (positive gradient — model gave them too much
probability mass that rightfully belonged to pink).