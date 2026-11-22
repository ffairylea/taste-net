# taste-net: documentation overview

This is a small neural network, built entirely from raw NumPy (no
PyTorch, no scikit-learn), trained to predict whether I would save
an image, using only my own labeled Pinterest data.

I built this to actually understand what's happening inside tools
like sentence-transformers, which I used in an earlier project
(chaos-organizer) without knowing the math underneath. Every file
here documents that understanding and explains the processes behind
the code. I worked through it with an AI tutor over many sessions:
planning, building, and asking questions until I understood
everything.

# How to read the documentation

Read the numbered files in order, from 00 to 17. Each one builds on
the ones before it.

If a term is unfamiliar, `GLOSSARY.md` in this folder has short
definitions. The other capitalized files here are side notes: the
original Phase 2 plan, the plan for the next experiment, limitations
and ideas, and the ethics questions the project raises.
