#!/usr/bin/env python3
"""Central differences against a program that prints a gradient: the check to run
before trusting one in `--method bfgs`.

    python3 examples/train/gradcheck.py PROGRAM X0   e.g.  ... logistic.py 0.3,-0.2,0.1

Prints the largest relative disagreement; exit status 1 if it exceeds 1e-6."""
import subprocess
import sys


def call(prog, x):
    out = subprocess.run([sys.executable, prog], input=" ".join(map(repr, x)) + "\n",
                         capture_output=True, text=True, check=True).stdout.split()
    return float(out[0]), [float(v) for v in out[1:]]


def worst(prog, x0, h=1e-6):
    _, g = call(prog, x0)
    w = 0.0
    for i in range(len(x0)):
        up, dn = list(x0), list(x0)
        up[i] += h
        dn[i] -= h
        num = (call(prog, up)[0] - call(prog, dn)[0]) / (2 * h)
        w = max(w, abs(num - g[i]) / max(1.0, abs(num)))
    return w


if __name__ == "__main__":
    err = worst(sys.argv[1], [float(v) for v in sys.argv[2].split(",")])
    print(f"largest relative disagreement {err:.3e}")
    sys.exit(0 if err <= 1e-6 else 1)
