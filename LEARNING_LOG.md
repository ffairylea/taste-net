# Learning log

Dated, honest notes as I build this. Not polished — this is the
scratch work, kept on purpose. If something's confusing, I write
that down too, then come back and update it once it clicks.

---

## [date] — scaffold

Got the base structure set up: feature extraction, from-scratch
neural net (forward + backward pass), training loop. Haven't run it
on real data yet — next step is collecting my actual Pinterest saves
and a contrast set from Unsplash.

Math I'm still working through:
- Why `dZ2 = A2 - y_true` is the correct derivative of cross-entropy
  + sigmoid combined, not just a convenient shortcut. Traced it out —
  the log terms and the sigmoid derivative cancel almost completely
  when you combine them. Worth redoing by hand with real numbers to
  make sure I actually believe it, not just accept it.
- ReLU's derivative being just "did this neuron fire, yes/no" (1 or
  0) still feels almost too simple compared to sigmoid's derivative.
  Need to sit with why that's fine / actually an advantage (avoids
  vanishing gradients that sigmoid hidden layers run into).

---

## 6th July

First real training run: 279 yes / 187 no images, 89.25% test accuracy. Feature importance surprised me — mostly blue-channel histogram bins, not saturation or brightness like I expected. Makes sense in hindsight: pink images probably have a narrow, consistent blue-channel range, while my "no" set (architecture/food/nature/etc.) has blue scattered everywhere. The model found a real, explainable pattern, not noise.

<!-- Next entry: after real data is collected and first training run -->
