# Glossary

**Layer**: a group of "neurons" that process the input together before passing results to the next group. My network has 2 layers: one hidden layer, one output layer.

**Weight (W)**: a number that decides how much a specific input feature matters to the decision, and in which direction (positive = pushes toward "yes," negative = pushes toward "no"). Not set by me; learned automatically during training.

**Bias (b)**: a constant number added on top of the weighted sum, like a baseline nudge before any feature is even considered.

**z (pre-activation)**: the raw weighted sum before any activation function is applied. z = (features · weights) + bias.

**ReLU**: an activation function: max(0, z). If the weighted sum is positive, keep it as-is; if negative, crush it to zero ("the neuron doesn't fire"). Loosely mirrors how biological neurons only send a signal once some threshold is crossed.

**Sigmoid**: an activation function that squashes any number into a range between 0 and 1, so it can be read as a probability. Used only at the very last layer, since we need a final "how confident" score.

**Epoch**: one full pass of the training loop over all the training data.

**Gradient**: how much the loss would change if a specific weight were nudged slightly. Tells the model which direction to adjust each weight to reduce error.

**Gradient descent**: the actual learning process: repeatedly compute gradients, then nudge every weight a small step in the direction that reduces loss.
**Feature vector**: the 32 numbers that describe one image (file 01). Each image becomes one point in a 32-dimensional space.

**Loss**: one number measuring how wrong a prediction was. This project uses cross-entropy, which punishes confident wrong answers much more than unsure ones (file 06).

**Backpropagation**: using the chain rule to work out every weight's gradient, starting from the loss and going backward through the layers (file 07).

**Softmax**: turns several raw scores into probabilities that add up to 1, one per board (file 11).

**Learning rate**: how big each weight update is.

**Optimizer**: the rule for updating the weights. SGD takes a plain step against the gradient; momentum averages recent gradients; Adam also gives each weight its own step size (file 13).

**Normalization**: rescaling each feature to mean 0 and standard deviation 1, using statistics from the training set only (file 12).

**Training set / test set**: the images the network learns from, and the images held back to check it on. A split is one random division into the two.

**Overfitting**: when the network does much better on training images than on test images, because it memorized them instead of learning a general pattern.

**Confusion matrix**: a table of true board against guessed board, showing which boards get mixed up.

**Balanced accuracy**: the average of the accuracy on each class, so a bigger class can't inflate the score.

**z-score (for a result)**: how many standard errors a result is above chance. Below about 1.96 it isn't significant at the usual 5% level.

**d′ (d-prime)**: how far apart two groups are on one feature, divided by how spread out they are. Measured with no model at all.

**kNN (k-nearest neighbors)**: labels a test image by looking at the k most similar training images and taking a vote.

**MD5 hash**: a fingerprint of a file's bytes. Two files only match if they are byte-for-byte identical.

**Perceptual hash**: a fingerprint of what an image looks like, so the same picture saved at a different size still matches (file 16).

**Pre-registration**: writing down the prediction and committing it to git before running the experiment, so the result can't quietly change the prediction (file 15).
