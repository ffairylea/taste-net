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

## Why exponentiate — testing the alternatives

The claim above is that exponentiating is needed because it's always
positive. That's true but incomplete, because plenty of operations
produce positive numbers. Working through why the obvious
alternatives fail sharpens what softmax actually requires.

**Alternative 1: just divide by the sum, no exponential.**

With $z = [2.0, 1.0, 0.1]$, the sum is 3.1, giving
$[0.645, 0.323, 0.032]$. That sums to 1 and looks fine.

Now let one score go negative — $z = [2.0, -1.0, 0.1]$. The sum is
1.1:

$$\left[\frac{2.0}{1.1},\ \frac{-1.0}{1.1},\ \frac{0.1}{1.1}\right] = [1.82,\ -0.91,\ 0.09]$$

A probability of $-0.91$, and another of $1.82$. Both meaningless.
Raw scores have no sign restriction — they're outputs of a weighted
sum, which can land anywhere — so any scheme that requires them to
be positive is broken from the start.

**Alternative 2: square them first.**

Squaring guarantees positivity. With $z = [2.0, 1.0, 0.1]$:

$$[4.0,\ 1.0,\ 0.01] \rightarrow \text{sum } 5.01 \rightarrow [0.798,\ 0.200,\ 0.002]$$

Valid. Now try $z = [-2.0, 1.0, 0.1]$:

$$[4.0,\ 1.0,\ 0.01] \rightarrow [0.798,\ 0.200,\ 0.002]$$

Identical output. The score of $-2.0$ — the model's *most rejected*
board — received the highest probability, because squaring erased
the sign. Absolute value fails the same way for the same reason.

So the requirement isn't "produce positive numbers." It's **strictly
increasing**: a larger input must always produce a larger output,
with no folding or wrapping. $e^z$ satisfies this — $e^{-2} = 0.135$,
$e^0 = 1$, $e^2 = 7.389$, always climbing, always positive, order
preserved perfectly.

**Alternative 3: some other strictly increasing positive function,
like $2^z$.**

This would actually work. Nothing breaks. The reason $e$ wins is
what happens next, in the loss function.

Take the log of a softmax output:

$$\log \hat y_i = \log\frac{e^{z_i}}{\sum_j e^{z_j}} = z_i - \log\sum_j e^{z_j}$$

The $\log$ and the $e$ annihilate each other, leaving the raw score
$z_i$ minus a term identical for every class. And cross-entropy
*takes the log* — the loss is $-\log(\hat y_{\text{correct}})$. So
the loss becomes:

$$L = -z_{\text{correct}} + \log\sum_j e^{z_j}$$

The exponential has vanished from the first term before
differentiation even begins. With $2^z$ the same cancellation
happens but drags a $\log 2$ along with it, permanently attached to
every gradient term. Still correct, just carrying an unnecessary
constant everywhere forever.

**The deeper reason, connecting to file 05.**

Sigmoid was derived in file 05 by starting from log-odds:

$$z = \log\frac{p}{1-p}$$

and solving for $p$. Solving that required exponentiating both
sides — the $e$ in sigmoid's formula wasn't chosen, it appeared
because the operation being undone was a logarithm.

Softmax is the same situation with more classes. The raw score $z$
behaves like a log-probability, so recovering an actual probability
means applying the inverse of log, which is $e^x$. The exponential
isn't a design decision anyone made for convenience. It's what
remains when a logarithm gets inverted.

And that also explains, from a different direction, why the gradient
cancels so cleanly later in this file. Cross-entropy applies a log,
softmax applies an exponential, and those operations undo each
other. The cancellation isn't a lucky algebraic accident — it's two
inverse functions meeting.

## Sigmoid is softmax with two classes — shown explicitly

Case A below notes that softmax's derivative matches sigmoid's, and
attributes it to sigmoid being softmax's two-class case. Here is
that claim actually demonstrated rather than asserted.

**With numbers first.** Two classes, scores $z_1 = 2.0$ and
$z_2 = 1.0$. Softmax on the first:

$$\hat y_1 = \frac{e^{2.0}}{e^{2.0} + e^{1.0}} = \frac{7.389}{7.389 + 2.718} = \frac{7.389}{10.107} = 0.731$$

Now sigmoid, applied to the *difference* between the two scores:

$$\sigma(2.0 - 1.0) = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} = \frac{1}{1.368} = 0.731$$

The same number — not approximately, exactly.

**Now the algebra showing it always holds.** Start with two-class
softmax and divide numerator and denominator by $e^{z_1}$:

$$\hat y_1 = \frac{e^{z_1}}{e^{z_1} + e^{z_2}} = \frac{1}{1 + \frac{e^{z_2}}{e^{z_1}}} = \frac{1}{1 + e^{z_2 - z_1}} = \frac{1}{1 + e^{-(z_1 - z_2)}}$$

That final expression is precisely sigmoid's formula, evaluated at
$z_1 - z_2$.

**What this explains about the binary model.** With two classes,
only the *difference* between the scores affects the output — adding
5 to both changes nothing, since it cancels in $z_1 - z_2$. So one
of the two scores is redundant: fix $z_2 = 0$ and let a single
output neuron represent the difference directly.

That is why `model.py` has $W_2$ of shape `(n_hidden, 1)` while
`model_multiclass.py` has `(n_hidden, n_classes)`. The binary model
isn't using a different mechanism with fewer outputs — it's using
this exact mechanism, with the redundant second output eliminated
because it carries no information.

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