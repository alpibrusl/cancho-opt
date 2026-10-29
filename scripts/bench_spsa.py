#!/usr/bin/env python3
"""Where `opt minimize --method spsa` stands against Qiskit's SPSA in time (docs/design.md section 11).

    OPT=build/opt python3 scripts/bench_spsa.py [--markdown]

The built-in sphere objective, SPSA with the same gains (a = 0.2, c = 0.1, A = 0.1 * iterations) and the same number of iterations: the wall time of the whole `opt` process (start included) and of
`qiskit_algorithms`' `SPSA.minimize` alone (the import and Python's own start are not counted, which favours Qiskit), with a Python objective. The two draw their perturbations from different generators,
so their iterates differ (`scripts/check_spsa.py` makes them equal by feeding Qiskit opt's stream); the work they do is the same: two evaluations per iteration and one at the end. Each time is the minimum of 7 runs.
"""
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
from qiskit_algorithms.optimizers import SPSA

OPT = os.environ.get("OPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "opt"))


def best_of(f, reps):
    best, out = 1e9, None
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def gains(a, c, big_a):
    def eta():
        k = 1
        while True:
            yield a / (k + big_a) ** 0.602
            k += 1

    def eps():
        k = 1
        while True:
            yield c / k ** 0.101
            k += 1
    return eta, eps


def main():
    md = "--markdown" in sys.argv
    if md:
        print("| parameters | iterations | opt | Qiskit SPSA | Qiskit / opt | evaluations: opt | Qiskit |")
        print("|---|---|---|---|---|---|---|")
    for n in (2, 8, 32):
        for iterations in (100, 1000):
            x0 = [1.0] * n
            xs = ",".join(repr(v) for v in x0)
            to, jo = best_of(lambda: json.loads(subprocess.run([OPT, "minimize", "--method", "spsa", "--x0", xs, "--objective", "sphere", "--maxiter", str(iterations), "--seed", "1"], capture_output=True, text=True).stdout), 7)
            eta, eps = gains(0.2, 0.1, 0.1 * iterations)
            calls = [0]

            def f(x):
                calls[0] += 1
                return float(np.dot(x, x))
            tq, rq = best_of(lambda: SPSA(maxiter=iterations, learning_rate=eta, perturbation=eps).minimize(f, np.array(x0)), 7)
            line = (n, iterations, to, tq, tq / to, jo["nfev"], int(rq.nfev) + 1)
            if md:
                print(f"| {line[0]} | {line[1]} | {line[2] * 1000:.1f} ms | {line[3] * 1000:.1f} ms | {line[4]:.1f}x | {line[5]} | {line[6]} |")
            else:
                print(f"n={n:3d} iterations={iterations:5d}  opt {to * 1000:7.1f} ms   qiskit {tq * 1000:7.1f} ms   ({tq / to:.1f}x)   evaluations opt {jo['nfev']} qiskit {int(rq.nfev) + 1}")


main()
