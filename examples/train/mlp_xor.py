#!/usr/bin/env python3
"""A 2-2-1 sigmoid network on XOR, trained by backpropagation: nine parameters
(W1 row by row, b1, v, c), the cross-entropy loss and its gradient on one line.

    opt minimize --method bfgs --x0 0.5,-0.4,0.3,0.8,0.1,-0.1,0.7,-0.6,0 -- python3 examples/train/mlp_xor.py

The same sigmoid backward as cancho-gpu's tests/chx/bwd_sigmoid.chx:
dz = dy * (s - s*s)."""
import math
import sys

DATA = [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]


def sig(z):
    return 1 / (1 + math.exp(-z))


def forward(p, x1, x2):
    w = p[0:4]
    b1 = p[4:6]
    v = p[6:8]
    c = p[8]
    h = [sig(w[0] * x1 + w[1] * x2 + b1[0]), sig(w[2] * x1 + w[3] * x2 + b1[1])]
    return h, sig(v[0] * h[0] + v[1] * h[1] + c)


def loss_grad(p):
    n = len(DATA)
    loss, g = 0.0, [0.0] * 9
    for x1, x2, y in DATA:
        h, o = forward(p, x1, x2)
        loss += -(y * math.log(max(o, 1e-300)) + (1 - y) * math.log(max(1 - o, 1e-300)))
        d = o - y                       # d loss / d (pre-activation of the output)
        g[6] += d * h[0]
        g[7] += d * h[1]
        g[8] += d
        for j in range(2):
            dz = d * p[6 + j] * (h[j] - h[j] * h[j])
            g[2 * j] += dz * x1
            g[2 * j + 1] += dz * x2
            g[4 + j] += dz
    return loss / n, [x / n for x in g]


if __name__ == "__main__":
    loss, g = loss_grad(list(map(float, sys.stdin.read().split())))
    print(repr(loss), *map(repr, g))
