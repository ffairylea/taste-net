# Collecting your data

This has to happen on your own machine/browser — I can't scrape
Pinterest for you, and honestly you shouldn't want me to: the labeling
process is where you're forced to notice *what* you're actually
choosing, which is half the point of the project.

## "Yes" set (`data/yes/`) — 150-300 images

1. Go through your real Pinterest boards (or Instagram saves, if that's
   more honest to your actual taste).
2. Right-click → save image, or use your browser's "save all images"
   for a board, then manually delete ones that don't belong.
3. Aim for images you'd *actually* re-pin today, not old saves you've
   outgrown — stale labels will just confuse the model.
4. Save them into `data/yes/` as `.jpg`, `.png`, or `.webp`.

Realistic pace: 15-20 min per 50 images if you're just scrolling and
saving. Don't overthink each one — fast, gut-reaction labeling is more
honest than deliberating.

## "No" set (`data/no/`) — 150-300 images

This needs to be a *fair* contrast set, not a strawman (e.g. don't
make "no" all ugly clip-art — that makes the task trivially easy and
the model won't learn anything real about YOUR taste specifically).

Best option: **Unsplash** (unsplash.com) — browse random/varied
categories, download a spread of genuinely decent photography that
you personally wouldn't pin. This keeps "no" images reasonably high
quality, so the model has to learn your *specific* preferences, not
just "good photo vs bad photo."

Save them into `data/no/`.

## A note on what "no" actually means

Worth sitting with, and worth writing about in your README later:
"no" doesn't mean "bad." It might mean "not my palette," "not my mood
right now," "well-made but not for me." That ambiguity is real and
it's fine — flag it as a limitation rather than pretending the labels
are cleaner than they are.

## When you're done

From inside `src/`, run:

```
pip install -r ../requirements.txt
python train.py
```
