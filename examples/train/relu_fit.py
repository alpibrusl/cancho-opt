#!/usr/bin/env python3
"""A ReLU network fitted to |x|: a loss with corners, so no gradient to print and
no ReLU kernel in cancho-gpu yet (docs/training.md of cancho-gpu); `nelder-mead`
needs neither. Seven parameters: two hidden units (w, b), two output weights, a bias.

    opt minimize --x0 0.5,0.1,-0.5,0.1,0.8,0.8,0 --maxfev 4000 -- python3 examples/train/relu_fit.py"""
import sys

XS = [i / 4 - 1.0 for i in range(9)]  # -1 .. 1


def predict(p, x):
    h1 = max(p[0] * x + p[1], 0.0)
    h2 = max(p[2] * x + p[3], 0.0)
    return p[4] * h1 + p[5] * h2 + p[6]


def loss(p):
    return sum((predict(p, x) - abs(x)) ** 2 for x in XS) / len(XS)


if __name__ == "__main__":
    print(repr(loss(list(map(float, sys.stdin.read().split())))))
