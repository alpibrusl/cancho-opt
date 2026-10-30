#!/usr/bin/env python3
"""Logistic regression as an `opt` objective: reads (w1, w2, b) on standard input,
prints the loss and its gradient on one line (the protocol of `--method bfgs`).

    opt minimize --method bfgs --x0 0,0,0 -- python3 examples/train/logistic.py

Eight points that no line separates, so the loss has a finite minimiser; a small
ridge on the weights (not the bias) keeps it strictly convex."""
import math
import sys

DATA = [(-2.0, -1.0, 0), (-1.5, 0.5, 0), (-1.0, -0.5, 0), (-0.2, 0.3, 1),
        (0.2, -0.3, 0), (1.0, 0.5, 1), (1.5, -0.5, 1), (2.0, 1.0, 1)]
L2 = 0.01


def loss_grad(w1, w2, b):
    n = len(DATA)
    loss, g = 0.0, [0.0, 0.0, 0.0]
    for x1, x2, y in DATA:
        z = w1 * x1 + w2 * x2 + b
        # log(1 + e^z) - y z, written to not overflow
        loss += max(z, 0.0) + math.log1p(math.exp(-abs(z))) - y * z
        p = 1 / (1 + math.exp(-z))
        g[0] += (p - y) * x1
        g[1] += (p - y) * x2
        g[2] += p - y
    loss = loss / n + 0.5 * L2 * (w1 * w1 + w2 * w2)
    return loss, [g[0] / n + L2 * w1, g[1] / n + L2 * w2, g[2] / n]


if __name__ == "__main__":
    loss, g = loss_grad(*map(float, sys.stdin.read().split()))
    print(repr(loss), *map(repr, g))
