#!/usr/bin/env python3
"""Gate 1 of docs/design.md section 6: `opt minimize` against scipy's Nelder-Mead, bit for bit.

    OPT=build/opt python3 scripts/check_scipy.py [--mutants]

Every case runs both and compares the minimiser, its value, the number of evaluations and of iterations, exactly (the doubles are printed in
the shortest form that reads back, so equal text is equal bits). `--mutants` builds `opt` with one deliberate change at a time (CANCHO=the
compiler) and requires this check to fail on every one.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = ["lib/nm/objective.cho", "lib/nm/lists.cho", "lib/nm/nm.cho", "tools/opt/main.cho"]


def sphere(x):
    t = 0.0
    for v in x:
        t = t + v * v
    return t


def rosen(x):
    t = None
    for k in range(len(x) - 1):
        a = x[k + 1] - x[k] * x[k]
        b = 1.0 - x[k]
        term = 100.0 * (a * a) + b * b
        t = term if t is None else t + term
    return t


def quad(x):
    t = 0.0
    for k, v in enumerate(x):
        d = v - float(k)
        t = t + d * d
    return t


def absolute(x):
    t = 0.0
    for k, v in enumerate(x):
        d = v - float(k)
        if d < 0.0:
            d = 0.0 - d
        t = t + float(k + 1) * d
    return t


FUNS = {"absolute": absolute, "sphere": sphere, "rosenbrock": rosen, "quadratic": quad}
INF = 1.0e300

# (objective, x0, options of the run)
CASES = [
    ("rosenbrock", [-1.2, 1.0], {}),
    ("rosenbrock", [1.3, 0.7, 0.8, 1.9, 1.2], {}),
    ("rosenbrock", [0.0, 0.0, 0.0], {}),
    ("rosenbrock", [-1.2, 1.0, 0.5], {"step": 0.5}),
    ("rosenbrock", [-1.2, 1.0], {"xatol": 1e-10, "fatol": 1e-10}),
    ("rosenbrock", [-1.2, 1.0], {"maxfev": 40}),
    ("rosenbrock", [-1.2, 1.0], {"maxiter": 20}),
    ("rosenbrock", [-1.2, 1.0], {"maxfev": 400, "step": 0.5}),
    ("rosenbrock", [3.0, 3.0], {"lower": [-INF, -INF], "upper": [2.0, 1.5]}),
    ("rosenbrock", [3.0, 3.0, 3.0], {"lower": [0.5, 0.5, 0.5], "upper": [INF, INF, 4.0]}),
    ("sphere", [1.0, 2.0, 3.0], {}),
    ("sphere", [1.0, 2.0, 3.0], {"maxfev": 5}),
    ("sphere", [1.0, 2.0, 3.0], {"maxfev": 11}),
    ("sphere", [1.0, 2.0, 3.0], {"maxfev": 4, "step": 0.5}),
    ("sphere", [0.0, 0.0], {}),
    ("sphere", [5.0], {}),
    ("quadratic", [1.0, -2.0, 0.5, 4.0, -3.0, 2.0, 1.0, -1.0], {"xatol": 1e-9, "fatol": 1e-9}),
    ("quadratic", [3.0, 3.0, 3.0, 3.0], {}),
    ("quadratic", [3.0, 3.0, 3.0, 3.0], {"lower": [0.5, 0.5, 0.5, 0.5], "upper": [INF, INF, INF, INF]}),
    ("quadratic", [-2.0, 0.0], {"step": 0.5, "maxfev": 60}),
    ("quadratic", [10.0, -10.0, 10.0, -10.0, 10.0, -10.0], {"xatol": 1e-8, "fatol": 1e-8}),
    # starts at which scipy's simplex shrinks (found by counting the evaluations of each iteration after the first)
    ("rosenbrock", [2.87, 0.54, 0.63], {}),
    ("rosenbrock", [-0.72, -2.2, 0.97], {}),
    ("rosenbrock", [-2.79, 0.3], {}),
    ("rosenbrock", [2.42, -0.99], {}),
    ("rosenbrock", [-0.69, -2.56, 0.94], {}),
    ("rosenbrock", [2.63, -1.22], {}),
    ("rosenbrock", [-1.99, -2.73], {}),
    ("rosenbrock", [-2.97, -2.66, 2.96, 0.34], {}),
    ("rosenbrock", [-2.56, 0.23, 2.66], {}),
    ("rosenbrock", [-2.98, -1.46, 0.51, -0.0], {}),
    ("rosenbrock", [-2.84, 1.18, -1.64, -0.29], {}),
    ("rosenbrock", [-2.79, 0.3], {"maxfev": 30}),
    ("rosenbrock", [-2.79, 0.3], {"maxfev": 27}),
    ("rosenbrock", [-2.79, 0.3], {"lower": [-3.0, 0.0], "upper": [3.0, 3.0]}),
]


def random_cases():
    """Seeded starts, so that the simplex does everything it can do (expand, contract both ways, shrink, clip)."""
    rng = np.random.default_rng(20261026)
    out = []
    for k in range(30):
        n = int(rng.integers(2, 7))
        name = ["rosenbrock", "absolute", "quadratic"][k % 3]
        x0 = [round(float(v), 3) for v in rng.uniform(-3.0, 3.0, n)]
        o = {}
        if k % 4 == 1:
            o["lower"] = [-1.0] * n
            o["upper"] = [1.5] * n
        if k % 5 == 2:
            o["step"] = 0.7
        out.append((name, x0, o))
    return out


def run_opt(opt, name, x0, o):
    argv = [opt, "minimize", "--x0", ",".join(repr(v) for v in x0), "--objective", name]
    for flag in ("step", "xatol", "fatol", "maxfev", "maxiter"):
        if flag in o:
            argv += ["--" + flag, repr(o[flag])]
    for flag in ("lower", "upper"):
        if flag in o:
            argv += ["--" + flag, ",".join("inf" if v >= INF and flag == "upper" else "-inf" if v <= -INF else repr(v) for v in o[flag])]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        return None, r.stderr
    return json.loads(r.stdout), ""


def run_scipy(name, x0, o):
    n = len(x0)
    opts = {"xatol": o.get("xatol", 1e-4), "fatol": o.get("fatol", 1e-4)}
    if "maxfev" in o:
        opts["maxfev"] = o["maxfev"]
    if "maxiter" in o:
        opts["maxiter"] = o["maxiter"]
    bounds = None
    base = np.array(x0)
    if "lower" in o or "upper" in o:
        lo = o.get("lower", [-INF] * n)
        hi = o.get("upper", [INF] * n)
        bounds = [(None if a <= -INF else a, None if b >= INF else b) for a, b in zip(lo, hi)]
        base = np.clip(base, lo, hi)   # `opt` clips x0 first, then builds the simplex (docs/design.md section 3)
    if "step" in o:
        opts["initial_simplex"] = np.vstack([base] + [base + o["step"] * np.eye(n)[k] for k in range(n)])
    f = FUNS[name]
    return minimize(lambda x: f([float(v) for v in x]), np.array(x0), method="Nelder-Mead", bounds=bounds, options=opts)


def check(opt):
    bad = 0
    for name, x0, o in CASES + random_cases():
        got, err = run_opt(opt, name, x0, o)
        want = run_scipy(name, x0, o)
        if got is None:
            print(f"FAIL {name} {x0} {o}: opt refused: {err.strip()[:200]}")
            bad += 1
            continue
        same = (got["x"] == [float(v) for v in want.x] and got["fun"] == float(want.fun) and got["nfev"] == int(want.nfev) and got["nit"] == int(want.nit)
                and got["converged"] == bool(want.success))
        if not same:
            print(f"FAIL {name} {x0} {o}:\n   opt   {got['x']} {got['fun']} nfev {got['nfev']} nit {got['nit']} {got['converged']}\n   scipy {list(want.x)} {float(want.fun)} nfev {want.nfev} nit {want.nit} {bool(want.success)}")
            bad += 1
    if bad == 0:
        print(f"ok   {len(CASES) + len(random_cases())} Nelder-Mead runs agree with scipy {__import__('scipy').__version__} bit for bit: minimiser, value, evaluations, iterations, success")
    return bad == 0


MUTANTS = [
    ("lib/nm/nm.cho", "(1.0 + 1.0 * 2.0) * w[xbar + j] - 1.0 * 2.0 * w[worst + j]", "(1.0 + 1.0 * 2.1) * w[xbar + j] - 1.0 * 2.1 * w[worst + j]", "expansion coefficient"),
    ("lib/nm/nm.cho", "w[r * n + j] = w[j] + 0.5 * (w[r * n + j] - w[j]);", "w[r * n + j] = w[j] + 0.6 * (w[r * n + j] - w[j]);", "shrink coefficient"),
    ("lib/nm/nm.cho", "if v > ws[hi + k] {\n            v = ws[hi + k];", "if v > ws[hi + k] + 1.0 {\n            v = ws[hi + k];", "clip to the upper bound"),
    ("lib/nm/nm.cho", "} else if fxr < w[fs + n - 1] {", "} else if fxr <= w[fs + n - 1] {", "accept test of the reflection"),
    ("lib/nm/nm.cho", "} else if fxc <= fxr {", "} else if fxc < fxr {", "outside contraction test"),
    ("lib/nm/nm.cho", "(1.0 + 0.05) * y", "(1.0 + 0.06) * y", "initial simplex"),
    ("lib/nm/nm.cho", "if spent >= maxfev {", "if spent > maxfev {", "the evaluation budget is strict"),
    ("lib/nm/nm.cho", "w[k * n + j] = 2.0 * w[hi + j] - w[k * n + j];", "w[k * n + j] = w[hi + j];", "reflection of the initial simplex into the box"),
    ("lib/nm/nm.cho", "if far <= xatol && gap <= fatol {", "if far <= xatol || gap <= fatol {", "termination"),
]


def mutants(cancho):
    ok = True
    for path, old, new, what in MUTANTS:
        d = tempfile.mkdtemp()
        try:
            for f in SRC:
                os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
                text = open(os.path.join(ROOT, f)).read()
                if f == path:
                    if old not in text:
                        print(f"FAIL mutant '{what}': the text it changes is not in {f}")
                        ok = False
                        continue
                    text = text.replace(old, new, 1)
                open(os.path.join(d, f), "w").write(text)
            out = os.path.join(d, "opt")
            r = subprocess.run([cancho, "build"] + [os.path.join(d, f) for f in SRC] + ["--std", "-o", out], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"FAIL mutant '{what}' does not build: {r.stderr[-200:]}")
                ok = False
                continue
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                survived = check(out)
            if survived:
                print(f"FAIL mutant '{what}' SURVIVED: the check does not notice it")
                ok = False
            else:
                print(f"ok   mutant killed: {what}")
        finally:
            shutil.rmtree(d)
    return ok


if __name__ == "__main__":
    opt = os.environ.get("OPT", os.path.join(ROOT, "build", "opt"))
    good = check(opt)
    if "--mutants" in sys.argv:
        good = mutants(os.environ.get("CANCHO", "cancho")) and good
    sys.exit(0 if good else 1)
