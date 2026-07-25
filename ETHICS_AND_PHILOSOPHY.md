# Ethics and philosophy behind taste-net

This project is small and low-stakes on purpose, but the questions it
raises aren't. Writing them out here, honestly, without resolving them
neatly — I don't think they have neat answers.

## Am I just teaching a model to repeat my own biases back to me?

If "yes" is only ever things I've already chosen, the model can't
discover anything about aesthetics I don't already hold. Best case,
it learns to predict my existing preferences accurately. It cannot,
by construction, tell me anything is beautiful that I don't already
think is beautiful. That's a real ceiling on what this kind of project
can ever claim to show — it's a mirror, not a source of new judgment.

## Is "my taste" even a stable target?

My Pinterest saves from six months ago might already disagree with
saves from today. If so, the model isn't learning one fixed thing
called "my taste" — it's learning some blurred average of a moving
target, and its errors might partly just be honest noise reflecting
that I've genuinely changed, not model failure. Worth checking later:
does accuracy improve if I only use recent saves versus my full
history?

## Preference vs. taste - a distinction I think matters

Predicting what I'd click or save is predicting revealed
preference — a pattern in past behavior. That's not the same as
taste in the fuller sense: informed aesthetic judgment that can
grow, be challenged, or be deliberately cultivated (through seeing
unfamiliar work, reading criticism, being confronted with things
outside a comfort zone). A model trained only on past saves can only
ever mirror who I already was — it has no mechanism to nudge me
toward an aesthetic sense I don't yet have. If anything, optimizing
for "predict what I'd click" and actually developing taste might
pull in opposite directions.

## The data behind "no" isn't neutral

My "no" images come from Unsplash — real photographers' work, used
here as a contrast class rather than engaged with as art in its own
right. Unsplash's license permits this kind of use, so there's no
legal issue, but it's worth naming plainly: turning someone's
photograph into "an example of what Lea wouldn't pin" is still using
their work instrumentally, not neutrally.

## What this project is not, and shouldn't be mistaken for

This is a toy, transparent, single-user experiment — I know exactly
what data went in, what the model can and can't do, and I'm the only
person affected by its output. That's structurally identical, though,
to a much bigger and less comfortable thing: recommendation systems
at real scale (TikTok, Instagram, Pinterest itself) run this same
basic mechanism — predict-and-reinforce individual preference — except
opaque to the user, optimized for engagement rather than honestly
stated preference, and operating on millions of people whose sense of
"what's normal to like" gets shaped, not just detected, by what gets
shown back to them. My version is honest about being a mirror. Production
systems at scale often aren't experienced that way by the people using
them, even when the underlying mechanism is similar in shape.

## Is removing the human aspect from "having taste" even coherent?

I don't think this project answers whether AI can have taste. I think
it tests something narrower and more honest: can a small amount of my
own labeled behavior predict more of my own behavior. Whether that
constitutes "taste" in any meaningful sense — versus just curve-fitting
to a moving target — is a question I'm leaving open on purpose, not
one I think a 29-feature numpy model is positioned to settle.


## Open questions I don't have answers to, but want on record

A few bigger questions this project brushed up against, worth
writing down honestly rather than letting them evaporate. I don't
think taste-net, at its current scale, can answer most of these —
but building even this small, simple model made them feel concrete
to me instead of abstract.

### Can a model go beyond what I explicitly gave it?

Not "does it have opinions" — but genuinely: if I feed it images it
has never seen, from a source totally outside my training data, does
it correctly generalize to my taste, or does it only work on things
close to what it already saw? This is actually testable at my
project's scale (a real experiment, not just a philosophical
question) — feed it unseen images from a different source entirely
and see what happens. Partial answer, to be filled in once I run it:
[pending].

### Could something like this "turn bad" without bad input?

For a model this narrow (predicting a single number from 29
features, no actions, no persistence, no goals), the honest answer
is no — there's no mechanism here for anything beyond outputting a
probability. This isn't a deep claim about AI safety in general,
just an honest, structural fact about what this specific,
simple architecture can and can't do.

### The bigger, unresolved version of this question

What I'm actually curious about — whether more complex systems can
develop behavior or internal representations that weren't directly
specified by training, and whether "accountability" cleanly stays
with the human designer as systems get more complex — is a real,
active, unresolved area of AI research (alignment and
interpretability). It's not something resolvable at my project's
scale, and it's not something even current professional researchers
have fully settled either. I don't think it's my place to claim an
answer here. I think it's worth being honest that the question
exists, that it's bigger than this project, and that this project is
the first time it stopped feeling abstract to me.

### Can AI have "taste," in any real sense?

I don't think this project answers that (see the earlier section on
preference vs. taste). But building it made the question sharper: a
model trained only on my past choices can, at most, mirror what I
already like. It has no equivalent of the emotional, embodied
experience of encountering something beautiful — no felt sense, only
statistical correlation. Whether "taste" requires that felt sense to
count as taste at all, or whether sufficiently sophisticated pattern-
matching could someday be functionally indistinguishable from it, is
a real open question in philosophy of mind that I don't think gets
resolved by better engineering alone.