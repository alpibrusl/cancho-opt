#!/usr/bin/env python3
"""The sampled logistic loss of noisy_logistic.py *with its gradient*: the loss and the gradient of the
same five drawn points, on one line (the protocol of `--method bfgs` and `--method adam`).

    opt minimize --method adam --x0 0,0,0 --lr 0.1 --maxiter 300 -- python3 examples/train/noisy_logistic_grad.py

The sample is chosen from a hash of the point, so a run is reproducible; two points that differ in the last bit
draw different samples. Adam uses the gradient of the sample it is given; BFGS's line search would not survive it."""
import hashlib
import math
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from logistic import DATA, L2  # noqa: E402

K = 5  # points drawn, with replacement, of the eight

if __name__ == "__main__":
    text = sys.stdin.read()
    w1, w2, b = map(float, text.split())
    digest = hashlib.sha256(text.encode()).digest()
    loss, g = 0.0, [0.0, 0.0, 0.0]
    for i in range(K):
        x1, x2, y = DATA[digest[i] % len(DATA)]
        z = w1 * x1 + w2 * x2 + b
        loss += max(z, 0.0) + math.log1p(math.exp(-abs(z))) - y * z
        p = 1 / (1 + math.exp(-z))
        g[0] += (p - y) * x1
        g[1] += (p - y) * x2
        g[2] += p - y
    print(repr(loss / K + 0.5 * L2 * (w1 * w1 + w2 * w2)), repr(g[0] / K + L2 * w1), repr(g[1] / K + L2 * w2), repr(g[2] / K))
