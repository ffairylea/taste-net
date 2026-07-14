# Glossary

**Layer** — a group of "neurons" that process the input together before passing results to the next group. My network has 2 layers: one hidden layer, one output layer.

**Weight (W)** — a number that decides how much a specific input feature matters to the decision, and in which direction (positive = pushes toward "yes," negative = pushes toward "no"). Not set by me — learned automatically during training.

**Bias (b)** — a constant number added on top of the weighted sum, like a baseline nudge before any feature is even considered.

**z (pre-activation)** — the raw weighted sum before any activation function is applied. z = (features · weights) + bias.

**ReLU** — an activation function: max(0, z). If the weighted sum is positive, keep it as-is; if negative, crush it to zero ("the neuron doesn't fire"). Loosely mirrors how biological neurons only send a signal once some threshold is crossed.

**Sigmoid** — an activation function that squashes any number into a range between 0 and 1, so it can be read as a probability. Used only at the very last layer, since we need a final "how confident" score.

**Epoch** — one full pass of the training loop over all the training data.

**Gradient** — how much the loss would change if a specific weight were nudged slightly. Tells the model which direction to adjust each weight to reduce error.

**Gradient descent** — the actual learning process: repeatedly compute gradients, then nudge every weight a small step in the direction that reduces loss.