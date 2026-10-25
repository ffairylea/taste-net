# 17: What's next, and the math behind it

Files 01 to 16 explain what this project does. This one is about what
it should do next, and why.

The result of file 15 is the starting point: the same 32 color
features reach 82.5% on pink-ish vs winter_minimal and 59.3% on
fleurs vs pink, which is not significantly better than a coin flip.
Every one of those 32 numbers is an average or a histogram over the
whole image, so the network never learns *where* anything is or
*what shape* it has. A pink peony and a pink bedroom can produce
nearly the same 32 numbers.

The next step is to give the model a few numbers that do carry shape
and position, and to measure, before training anything, whether they
separate flowers from rooms. The plan, with the experimental design,
is in `NEXT_EXPERIMENT_SPATIAL_FEATURES.md`. This file is the math.

Nothing here has been run on the real boards yet. Every number below
is either a worked example on tiny made-up inputs, or a calculation
about how big a result would need to be to count.

## Part 1: Finding edges with a 3×3 grid of weights

### The kernel is a weighted sum

An edge is where brightness changes quickly. To find one, slide a
3×3 grid of weights (a **kernel**) over the greyscale image, and at
each position compute the weighted sum of the 9 pixels underneath.
That is exactly the operation from file 03, applied to a small patch
instead of the whole feature vector.

The Sobel kernel for horizontal change:

$$
S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}
$$

Left column negative, right column positive: it measures "right side
minus left side." The middle row counts double because it is closest
to the center pixel. $S_y$ is the same kernel turned on its side, so
it measures "bottom minus top."

### Worked example: a vertical edge

A patch that is dark on the left and bright on the right (brightness
from 0 to 1):

$$
\begin{bmatrix} 0.2 & 0.2 & 0.8 \\ 0.2 & 0.2 & 0.8 \\ 0.2 & 0.2 & 0.8 \end{bmatrix}
$$

Multiply each pixel by the matching weight in $S_x$ and add:

$$g_x = (-1)(0.2) + (1)(0.8) + (-2)(0.2) + (2)(0.8) + (-1)(0.2) + (1)(0.8) = 2.4$$

The middle column is multiplied by 0, so it drops out. For $S_y$, the
top row and bottom row are identical, so they cancel: $g_y = 0$.

So at this spot, brightness changes strongly left to right and not at
all top to bottom.

### Strength and orientation

From $g_x$ and $g_y$ come two numbers per pixel:

- **strength** $= \sqrt{g_x^2 + g_y^2}$, how sharp the change is
- **angle** $= \arctan2(g_y, g_x)$, which way brightness increases

A diagonal patch, dark in the top-left and bright in the bottom-right:

$$
\begin{bmatrix} 0.2 & 0.2 & 0.2 \\ 0.2 & 0.2 & 0.8 \\ 0.2 & 0.8 & 0.8 \end{bmatrix}
$$

gives $g_x = 1.8$ and $g_y = 1.8$. Strength $= \sqrt{1.8^2 + 1.8^2}
\approx 2.55$, angle $= 45°$.

Two details matter for what comes next.

**The angle points across the edge, not along it.** In the vertical
edge example the angle is 0° (horizontal), because brightness
increases horizontally, while the edge line itself is vertical. That
is fine for this project: every angle is rotated by the same 90°, so
how *spread out* the angles are does not change.

**Direction does not matter, only orientation.** Dark-to-bright and
bright-to-dark along the same line are the same line. So angles are
folded into the range 0° to 180°: an angle of 200° becomes 20°.

### Why this is interesting

A trained convolutional network learns its own first-layer kernels,
and many of them end up looking like oriented edge detectors. Hubel
and Wiesel found that many simple cells in visual cortex (V1)
respond best to edges at a particular orientation. Sobel is a
hand-made version of the same idea. These are analogies, not claims
that the brain runs Sobel: the point is that "find oriented edges
first" is how biological and artificial vision both seem to start,
and here I get to build that step myself and know exactly what it
computes.

## Part 2: Orientation entropy, one number for "straight or curved"

### The histogram

Take every edge pixel (the strongest 15% of gradient strengths), sort
its angle into one of 12 bins of 15° each, and count. Stronger edges
count more: each pixel adds its strength, not just 1.

A room is mostly floor lines, wall edges and window frames, so its
histogram has two tall bars (horizontal and vertical) and little
else. A petal curves through every angle, so a flower's histogram is
spread out.

### Entropy turns the spread into one number

Turn the histogram into proportions $p_1, \dots, p_{12}$ that add up
to 1. Entropy is:

$$H = -\sum_i p_i \ln p_i$$

(bins with $p_i = 0$ contribute nothing). It is largest when every
bin is equal, and that maximum is $\ln 12$, so dividing by $\ln 12$
puts the result between 0 and 1.

Worked out for three histograms:

| histogram | calculation | normalized entropy |
|---|---|---|
| room: half at 0°, half at 90° | $-2 \times 0.5 \ln 0.5 = \ln 2$ | $\ln 2 / \ln 12 = 0.28$ |
| mixed: 0.4, 0.3, 0.2, 0.1 in four bins | $\approx 1.28$ | $0.52$ |
| flower: equal in all 12 bins | $\ln 12$ | $1.00$ |

This is the same entropy as in information theory: how surprised you
would be, on average, by the angle of a randomly picked edge. In a
room you can mostly guess it. On a flower you can't.

### Why this is the strongest candidate

It measures exactly the distinction the hypothesis is about, straight
vs curved, in one interpretable number. On a synthetic room and a
synthetic flower it scored 0.46 and 1.00.

## Part 3: Box-counting dimension, a number for "how branching"

### Dimension from counting boxes

Cover the edge map with a grid of boxes of side $s$ and count how
many boxes contain at least one edge pixel. Call that $N(s)$.

On an 8×8 image:

| box side $s$ | straight line across | completely filled |
|---|---|---|
| 4 | 2 | 4 |
| 2 | 4 | 16 |
| 1 | 8 | 64 |

Halving the box size doubles the count for a line and quadruples it
for a filled square. So $N(s) \propto (1/s)^D$, where $D = 1$ for a
line and $D = 2$ for a filled area. Taking logs:

$$\ln N(s) = D \cdot \ln(1/s) + c$$

which is a straight line, and $D$ is its slope. Count at several box
sizes, fit a line through the points with least squares, read off
the slope. Shapes that branch and wrinkle, like coastlines, fern
leaves and tree outlines, land between 1 and 2.

### Why this will probably disappoint

A smooth curve also has dimension 1, exactly like a straight line.
Zoom in far enough on a petal outline and it looks straight. So
box counting measures *branching and texture*, not *curvature*. A
close-up of a single peony has smooth outlines and may score no
higher than a room.

There is also a measurement problem. A 128-pixel image only allows
box sizes from about 2 to 32, and Sobel edges are a few pixels thick,
so even a perfectly straight line does not score exactly 1. On the
synthetic room it scored 1.55. This feature is in the plan to be
tested, and my prediction is that it will be the weakest one.

## Part 4: Center vs border sharpness

### The Laplacian measures "how sudden"

The Laplacian is a second derivative: not how fast brightness
changes, but how fast the *change* changes. Its 3×3 kernel is:

$$
\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}
$$

In one dimension it is just $[1, -2, 1]$: left neighbor, minus twice
the pixel, plus right neighbor. Run it along two rows of pixels:

| row | values | Laplacian | variance |
|---|---|---|---|
| sharp edge | 0, 0, 0, 1, 1, 1 | 0, 1, −1, 0 | 0.50 |
| blurred edge | 0, 0, 0.33, 0.67, 1, 1 | 0.33, 0, 0, −0.33 | 0.06 |

Same edge, same brightness at both ends, but the blurred version
spreads the change out, so the Laplacian sees only small bumps.
That is why the **variance of the Laplacian** is a standard focus
measure: blur lowers it.

### The feature

Compute that variance in the middle of the image (the central half
in each direction, a quarter of the area) and in the border, and take

$$\ln \frac{\text{sharpness in the middle}}{\text{sharpness at the border}}$$

A flower close-up usually has a sharp subject and a blurred
background, so it should score well above 0. A room photo is usually
sharp everywhere, so it should sit nearer 0. The log makes "twice as
sharp" and "half as sharp" equally far from 0, in opposite
directions.

## Part 5: Mirror symmetry

Flip the image left to right and measure how similar it is to the
original. Subtract the mean brightness from both, then compute the
correlation:

$$r = \frac{\sum a_i b_i}{\sqrt{\sum a_i^2 \sum b_i^2}}$$

where $a$ is the image and $b$ its flip. This is the cosine of the
angle between the two images as vectors, so it runs from −1 to 1.

On single rows of four pixels:

| row | centered | flipped | $r$ |
|---|---|---|---|
| 1, 2, 2, 1 | −0.5, 0.5, 0.5, −0.5 | identical | 1.00 |
| 1, 3, 2, 1 | −0.75, 1.25, 0.25, −0.75 | −0.75, 0.25, 1.25, −0.75 | 0.64 |
| 1, 2, 3, 4 | −1.5, −0.5, 0.5, 1.5 | reversed | −1.00 |

The last row shows a weakness worth knowing: a smooth left-to-right
brightness gradient scores −1, as "perfectly anti-symmetric." A room
lit from a window on one side could do this. So this feature is
crude on purpose, and its result should be read with that in mind.

### The real version is crystallography

A single flower has *rotational* symmetry as well as mirror
symmetry. A five-petal flower looks the same after a turn of 72°,
and also has five mirror lines. In crystallography these are point
groups: $C_n$ for rotation only, $D_n$ for rotation plus mirrors, so
that flower is $D_5$. The 17 wallpaper groups are the different
thing, for patterns that repeat across a plane, which a flower photo
does not. Detecting point groups in messy photos is a research
problem, not a quick feature, which is why the plan only uses the
left-right score.

## Part 6: Putting position back in, cheaply

Split the image into a 3×3 grid and take the mean brightness of each
cell. Nine numbers, and now "bright top, dark bottom" (a sky over a
field) is distinguishable from "dark top, bright bottom," which a
single average brightness cannot tell apart.

### Why brightness and not color

Mean red, green and blue per cell would be 27 numbers. That matters
because of the parameter count. The board network's first layer has
one weight per input per hidden neuron (file 03), so:

| features | parameters in the board network |
|---|---|
| 32 color (now) | 32 × 8 + 8 + 8 × 3 + 3 = 291 |
| 32 + 13 spatial (the plan) | 45 × 8 + 8 + 8 × 3 + 3 = 395 |
| 32 + 4 + 27 (RGB grid) | 63 × 8 + 8 + 8 × 3 + 3 = 539 |

With roughly 560 training images, file 15's width sweep already
showed that extra parameters buy memorization rather than
generalization. So the plan keeps the additions to 13.

Edge orientation pooled over a coarse grid is close to the "gist"
descriptor of Oliva and Torralba (2001), designed to capture how
quickly people recognize a scene at a glance. Part 1 plus this part
is a small version of that idea.

## Part 7: Measuring before training, d′

The first test trains nothing. For each new feature, measure how far
apart the two boards sit:

$$d' = \frac{\text{mean}_A - \text{mean}_B}{\sqrt{(\sigma_A^2 + \sigma_B^2)/2}}$$

the gap between the averages, in units of the typical spread. d′
comes from signal detection theory, where perception researchers use
it to measure how well a *person* tells two stimuli apart. Here it
measures how well a *feature* does.

A made-up example. Orientation entropy on four flowers: 0.9, 0.8,
1.0, 0.7 (mean 0.85). On four rooms: 0.4, 0.5, 0.3, 0.6 (mean 0.45).
Both groups have variance 0.0125, so

$$d' = \frac{0.85 - 0.45}{\sqrt{0.0125}} \approx 3.6$$

which would be enormous: the two groups barely overlap. Real data
will be much messier. For scale, the best of the 32 color features
on fleurs vs pink has d′ = 0.63, meaning the two distributions
overlap heavily.

**The bar:** if any spatial feature beats 0.63 on fleurs vs pink, the
representation claim from file 15 gets support from a new direction,
before any network has been trained.

The limit: d′ looks at one feature at a time, so it cannot see
features that only work in combination. That is what the training
runs are for.

## Part 8: How big a result has to be to count

With about 91 fleurs and pink images in a test split, how far from
50% does accuracy need to be before it is more than luck?

If the model were guessing, the fraction it gets right would vary
from split to split with standard error

$$SE = \sqrt{\frac{p(1-p)}{n}} = \sqrt{\frac{0.5 \times 0.5}{91}} \approx 0.052$$

about 5 percentage points. A result is conventionally called
significant when it is more than 1.96 standard errors away, so
**beating chance needs about 50 + 1.96 × 5.2 ≈ 60%.**

Beating the color-only result is a harder bar. Now two uncertain
numbers are being compared, and the uncertainty of a difference is
larger than either: treating the two as independent, the standard
error of the difference is $\sqrt{2} \times 5.1 \approx 7.3$ points,
so **convincingly beating 59.3% needs roughly 59.3 + 1.96 × 7.3 ≈
74%.** That is a conservative estimate. Both models are scored on
the same test images, and a paired test (McNemar's test) that uses
this would lower the bar somewhat. Averaging over several seeds
helps too.

This is why the plan says to decide in advance what counts as a
result: it is easy to get 64%, call it an improvement, and be wrong.

## Part 9: Beyond spatial features

Color is clearly a small part of taste, and the interesting question
is how small. The roadmap adds one richer kind of feature at a time
and measures how much more of my saves each explains.

### Spatial frequency: the 1/f slope

Any image can be rewritten as a sum of waves (a Fourier transform,
`np.fft.fft2`). Low frequencies are broad shapes, high frequencies
are fine detail. For natural photographs, the amplitude of each
frequency falls off roughly as $1/f$ (Field, 1987):

$$A(f) \propto \frac{1}{f^{\alpha}}, \qquad \ln A = -\alpha \ln f + c$$

with $\alpha$ around 1. Like box counting, it is a slope on a log-log
plot, and it summarizes how much fine detail an image has relative to
its large shapes. Studies of large art collections found paintings
across cultures share roughly this statistic (Graham and Field,
2007; Redies, 2007). Whether my saves sit closer to it than images I
skip is testable with about ten lines of NumPy.

### Learned features

A pretrained vision network's internal representation carries what
the image *is*: a peony, a bedroom, a ski slope. Comparing it against
the hand-built features measures how much of the gap learned
features close beyond what I can build and explain by hand. This is
the pretrained comparison already on the README roadmap.

### The ceiling: how often do I agree with myself?

Some of taste is in no image at all: mood, context, novelty, why I
saved something on a particular day. That part has no feature, but it
has a measurable limit.

Re-label 100 random images blind, weeks after the originals, and
count how often the two labels agree. Call that agreement $a$.

Suppose each time I label an image, I "flip" away from my real
preference with some probability $q$, independently each time. Two
labelings agree when both are right or both are flipped:

$$a = (1-q)^2 + q^2$$

Solving that quadratic for $q$:

$$q = \frac{1 - \sqrt{2a - 1}}{2}$$

A model that captured my real preference perfectly would still
disagree with my labels whenever I flipped, so its best possible
accuracy is $1 - q$:

| self-agreement $a$ | flip rate $q$ | best possible accuracy |
|---|---|---|
| 75% | 14.6% | 85.4% |
| 85% | 8.2% | 91.8% |
| 90% | 5.3% | 94.7% |

The surprising part: the ceiling is *higher* than my agreement with
myself, because when I disagree with myself, one of the two labels
was the noisy one, not both. The model of noise is simple (every
image equally likely to flip, flips independent), so treat the
ceiling as an estimate, not an exact bound.

This is the one item on the list that turns "taste is more than
color" from an opinion into a number: the gap between the best
features I can build and that ceiling is the part of my taste that
isn't in the pixels.

## Order of work

1. Write down the predictions and commit them, before any real data.
2. d′ for each spatial feature on fleurs vs pink (Part 7).
3. Train color only, spatial only, color + spatial, several seeds,
   and judge against the bars in Part 8.
4. Self-consistency labeling (Part 9), which needs no code at all.
5. Pretrained comparison, then the 1/f slope.
