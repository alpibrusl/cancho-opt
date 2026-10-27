#!/usr/bin/env python3
"""Where `opt` stands against scipy in time (docs/design.md section 10).

    OPT=build/opt python3 scripts/bench_scipy.py [--markdown]

For the built-in Rosenbrock objective, BFGS (with its gradient) and Nelder-Mead at several sizes: the wall time of the whole `opt` process (start included), and of scipy's `minimize` alone (the import of
scipy and Python's own start are not counted, which favours scipy), with scipy's numpy `rosen` and with a pure-Python objective written like the one `check_scipy.py` and `check_bfgs.py` use. Each time is the minimum of
7 runs (5 for Nelder-Mead at 5 and 10 parameters, 3 for the pure-Python objective).

A row is **comparable** only when both did the same work: the same number of iterations and evaluations. scipy's `rosen` sums with numpy's pairwise sum from 9 terms and its BLAS products differ from ours, so at 10 and 30
parameters the work can differ (the counts are printed); such rows are not a comparison of speed and are marked.
"""
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_bfgs as c  # noqa: E402

OPT = os.environ.get("OPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build", "opt"))


def best_of(f, reps):
    best, out = 1e9, None
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return best, out


def run_opt(args, reps):
    def go():
        return json.loads(subprocess.run([OPT, *args], capture_output=True, text=True).stdout)
    return best_of(go, reps)


def x_start(n):
    return [-1.2 if k % 2 == 0 else 1.0 for k in range(n)]


def main():
    md = "--markdown" in sys.argv
    rows = []
    for n in (2, 5, 10, 20, 30):
        x0 = x_start(n)
        xs = ",".join(repr(v) for v in x0)
        to, jo = run_opt(["minimize", "--method", "bfgs", "--x0", xs, "--objective", "rosenbrock", "--gtol", "1e-8"], 7)
        ts, rs = best_of(lambda: minimize(rosen, x0, jac=rosen_der, method="BFGS", options={"gtol": 1e-8}), 7)
        tp, _ = best_of(lambda: minimize(lambda x: c.rosen([float(v) for v in x])[0], x0, jac=lambda x: np.array(c.rosen([float(v) for v in x])[1]), method="BFGS", options={"gtol": 1e-8}), 3)
        same = jo["nit"] == rs.nit and jo["nfev"] == rs.nfev
        rows.append(("BFGS", n, to, ts, tp, same, f"{jo['nit']} / {jo['nfev']}", f"{rs.nit} / {rs.nfev}"))
    for n in (2, 5, 10):
        x0 = x_start(n)
        xs = ",".join(repr(v) for v in x0)
        opts = {"xatol": 1e-8, "fatol": 1e-8, "maxfev": 20000}
        to, jo = run_opt(["minimize", "--x0", xs, "--objective", "rosenbrock", "--maxfev", "20000", "--xatol", "1e-8", "--fatol", "1e-8"], 5)
        ts, rs = best_of(lambda: minimize(rosen, x0, method="Nelder-Mead", options=opts), 5)
        tp, _ = best_of(lambda: minimize(lambda x: c.rosen([float(v) for v in x])[0], x0, method="Nelder-Mead", options=opts), 3)
        same = jo["nit"] == rs.nit and jo["nfev"] == rs.nfev
        rows.append(("Nelder-Mead", n, to, ts, tp, same, f"{jo['nit']} / {jo['nfev']}", f"{rs.nit} / {rs.nfev}"))
    start, _ = best_of(lambda: subprocess.run([OPT, "minimize", "--x0", "1", "--objective", "sphere"], capture_output=True), 7)
    if md:
        print("| method | parameters | opt | scipy (numpy `rosen`) | scipy (pure-Python objective) | iterations / evaluations: opt | scipy | same work |")
        print("|---|---|---|---|---|---|---|---|")
    for m, n, to, ts, tp, same, co, cs in rows:
        if md:
            print(f"| {m} | {n} | {to * 1000:.1f} ms | {ts * 1000:.1f} ms | {tp * 1000:.1f} ms | {co} | {cs} | {'yes' if same else 'no'} |")
        else:
            print(f"{m:12} {n:3d}  opt {to * 1000:7.1f} ms   scipy {ts * 1000:7.1f} ms   scipy(py) {tp * 1000:7.1f} ms   opt {co}  scipy {cs}  {'same work' if same else 'DIFFERENT WORK'}")
    print(f"process start and a one-parameter run of opt: {start * 1000:.1f} ms")


main()
