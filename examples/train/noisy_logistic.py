#!/usr/bin/env python3
"""The logistic loss of logistic.py estimated on a *sample* of the data, so it is
noisy, and with no gradient printed: one number, the objective of `--method spsa`
(or nelder-mead).

    opt minimize --method spsa --x0 0,0,0 --maxiter 300 -- python3 examples/train/noisy_logistic.py

The sample is chosen from a hash of the point, so a run is reproducible; two
points that differ in the last bit draw different samples, which is the noise."""
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
    loss = 0.0
    for i in range(K):
        x1, x2, y = DATA[digest[i] % len(DATA)]
        z = w1 * x1 + w2 * x2 + b
        loss += max(z, 0.0) + math.log1p(math.exp(-abs(z))) - y * z
    print(repr(loss / K + 0.5 * L2 * (w1 * w1 + w2 * w2)))
