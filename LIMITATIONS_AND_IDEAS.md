# Limitations and ideas (my own thinking, not settled answers)

## The "what to ignore" problem
My network gives every one of the 32 features constant importance for every image. It can't dynamically decide "ignore saturation, focus on hue" depending on context. This is a real limitation, and it maps to something close to how the brain's reticular activating system filters out irrelevant sensory input to focus attention. Real ML systems address something like this with "attention mechanisms" (the core idea behind transformers). I noticed this gap myself before knowing the term for it.

## The multi-board question
**a narrow, visually consistent aesthetic may make this an easier classification problem than 'taste' in general. Worth testing later with a harder, less visually uniform board. (Update: this turned out to be right. When the dataset was rebuilt with "yes" images sampled across all my boards instead of only the pink one, Phase 1 accuracy fell from 89.25% to 72.9%. See docs/16.)
If I expand "yes" into multiple boards (pink, winter-minimal, etc.), the network needs a way to first recognize "this is relevant to some aesthetic I care about" before deciding *which* one, closer to a triage/attention step than a flat yes/no. (Update: Phase 2 built exactly this. Stage 1 decides "would I save this at all", and only then does Stage 2 pick the board. See the README.)

## Technical limits shape what "taste" the model can even represent
The model's architecture isn't neutral. Every design choice (which
features I extracted, ReLU vs sigmoid, 8 hidden neurons vs more)
constrains what kind of pattern it's even capable of finding. This
connects to the bigger question in the README: if the model "learns
my taste," it's really learning whatever fits inside these specific
mathematical constraints, not some pure, architecture-independent
truth about my aesthetic preferences. A different architecture might
find a genuinely different notion of "my taste" in the same data.
## A bug that ran cleanly and produced nothing

`train.py` originally saved the trained weights but discarded the
normalization statistics: the 32 means and 32 standard deviations
the features were scaled by. Since the weights were learned in that
scaled coordinate system, saving them alone made them unusable on
any new image.

What makes this worth recording isn't the bug, it's the failure
mode. The intuitive repair (normalize the new image the way the
training images were normalized) produces a feature vector of all
zeros, because a single sample has zero standard deviation from
itself. Every image then maps to the identical input, the forward
pass reduces to the bias term, and the model returns one constant
probability regardless of what it's shown.

Nothing crashes. The shapes are valid, the arithmetic is legal, the
output is a plausible number between 0 and 1. The only way to catch
it is to reason about what the operations mean rather than whether
they execute, which is a general lesson about this kind of code,
not a fact about this specific mistake. Full working in
`docs/12_normalization_and_saved_state.md`.

`predict.py` now refuses to run against a weights file lacking those
statistics rather than proceeding on a guess, on the principle that
when the wrong behaviour is invisible, failing loudly is the only
safe default.