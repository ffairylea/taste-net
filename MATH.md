# The math behind taste-net

## Forward pass (worked example, by hand)

Given: x = [0.6, 0.2] (warmth, saturation — simplified to 2 features for this example)
W1 = [0.5, -0.3], b1 = 0.1
W2 = 0.8, b2 = 0.0
True label y = 1

z1 = (0.6)(0.5) + (0.2)(-0.3) + 0.1 = 0.34
a1 = ReLU(0.34) = 0.34   (already positive, unchanged)

z2 = (0.34)(0.8) + 0.0 = 0.272
y_hat = sigmoid(0.272) ≈ 0.568

loss = -log(0.568) ≈ 0.566
(model leaned the right direction, wasn't confident yet — expected with untrained random weights)

## Backward pass — coming next, will fill in once computed by hand