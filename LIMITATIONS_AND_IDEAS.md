# Limitations and ideas (my own thinking, not settled answers)

## The "what to ignore" problem
My network gives every one of the 29 features constant importance for every image — it can't dynamically decide "ignore saturation, focus on hue" depending on context. This is a real limitation, and it maps to something close to how the brain's reticular activating system filters out irrelevant sensory input to focus attention. Real ML systems address something like this with "attention mechanisms" (the core idea behind transformers) — I noticed this gap myself before knowing the term for it.

## The multi-board question
**a narrow, visually consistent aesthetic may make this an easier classification problem than 'taste' in general — worth testing later with a harder, less visually uniform board
If I expand "yes" into multiple boards (pink, winter-minimal, etc.), the network needs a way to first recognize "this is relevant to some aesthetic I care about" before deciding *which* one — closer to a triage/attention step than a flat yes/no. Current architecture doesn't have this. Worth exploring in a v2.

## Technical limits shape what "taste" the model can even represent
The model's architecture isn't neutral — every design choice (which
features I extracted, ReLU vs sigmoid, 8 hidden neurons vs more)
constrains what kind of pattern it's even capable of finding. This
connects to the bigger question in the README: if the model "learns
my taste," it's really learning whatever fits inside these specific
mathematical constraints — not some pure, architecture-independent
truth about my aesthetic preferences. A different architecture might
find a genuinely different notion of "my taste" in the same data.