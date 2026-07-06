# Limitations and ideas (my own thinking, not settled answers)

## The "what to ignore" problem
My network gives every one of the 29 features constant importance for every image — it can't dynamically decide "ignore saturation, focus on hue" depending on context. This is a real limitation, and it maps to something close to how the brain's reticular activating system filters out irrelevant sensory input to focus attention. Real ML systems address something like this with "attention mechanisms" (the core idea behind transformers) — I noticed this gap myself before knowing the term for it.

## The multi-board question
If I expand "yes" into multiple boards (pink, winter-minimal, etc.), the network needs a way to first recognize "this is relevant to some aesthetic I care about" before deciding *which* one — closer to a triage/attention step than a flat yes/no. Current architecture doesn't have this. Worth exploring in a v2.