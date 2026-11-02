#!/usr/bin/env python3
"""Gate 3 of docs/design.md section 6 (added with section 9): `opt minimize --method bfgs` against scipy's BFGS with an analytic gradient.

    OPT=build/opt python3 scripts/check_bfgs.py [--mutants]

scipy's matrix and dot products are BLAS's, whose order of summation (and use of fused multiply-add) is not ours to know; `opt` sums sequentially. So the oracle is scipy with exactly those products replaced: `np.dot` and
`np.linalg.norm` of the vectors and matrices BFGS uses are patched, for the duration of this check, to sequential sums in increasing index. Nothing else of scipy is touched. Against that oracle `opt` must agree
**bit for bit**: the same minimiser, value, gradient, iterations, and counts of value and gradient evaluations, and the same status. Against scipy as it is (BLAS) the minimiser is the same but the last bits of every iterate
differ and, once in about ten runs, a marginal step-size decision goes the other way and the count differs by one or two (`--blas` measures the rate). `--mutants` builds `opt` with one deliberate bug at a time
(CANCHO=the compiler) and requires the check to fail on each.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import numpy as np
from scipy.optimize import minimize

warnings.simplefilter("ignore")

_DOT, _NORM = np.dot, np.linalg.norm


def _seqdot(a, b, out=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim == 1 and b.ndim == 1:
        s = 0.0
        for p in range(a.shape[0]):
            s = s + a[p] * b[p]
        return np.float64(s)
    if a.ndim == 2 and b.ndim == 1:
        r = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            s = 0.0
            for p in range(a.shape[1]):
                s = s + a[i, p] * b[p]
            r[i] = s
        return r
    if a.ndim == 2 and b.ndim == 2:
        r = np.zeros((a.shape[0], b.shape[1]))
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                s = 0.0
                for p in range(a.shape[1]):
                    s = s + a[i, p] * b[p, j]
                r[i, j] = s
        return r
    return _DOT(a, b)


def _seqnorm(x, ord=None, *args, **kw):
    if ord is None:
        s = 0.0
        for v in np.asarray(x, dtype=float).ravel():
            s = s + v * v
        return np.sqrt(s)
    return _NORM(x, ord, *args, **kw)


def sequential(on):
    np.dot = _seqdot if on else _DOT
    np.linalg.norm = _seqnorm if on else _NORM

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = ["lib/nm/objective.cho", "lib/nm/lists.cho", "lib/nm/nm.cho", "lib/bfgs/dcsrch.cho", "lib/bfgs/bfgs.cho", "lib/persist/persist.cho", "tools/opt/main.cho"]


def sphere(x):
    t = 0.0
    for v in x:
        t = t + v * v
    return t, [2.0 * v for v in x]


def rosen(x):
    n = len(x)
    t = None
    for k in range(n - 1):
        a = x[k + 1] - x[k] * x[k]
        b = 1.0 - x[k]
        term = 100.0 * (a * a) + b * b
        t = term if t is None else t + term
    g = [0.0] * n
    g[0] = -400.0 * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2])
    for k in range(1, n - 1):
        xm, xm1, xp1 = x[k], x[k - 1], x[k + 1]
        g[k] = 200.0 * (xm - xm1 * xm1) - 400.0 * (xp1 - xm * xm) * xm - 2.0 * (1.0 - xm)
    return t, g


def quad(x):
    t = 0.0
    for k, v in enumerate(x):
        d = v - float(k)
        t = t + d * d
    return t, [2.0 * (v - float(k)) for k, v in enumerate(x)]


FUNS = {"sphere": sphere, "rosenbrock": rosen, "quadratic": quad}

# (objective, x0, options)
CASES = [
    ("rosenbrock", [-1.2, 1.0], {}),
    ("rosenbrock", [3.0, -2.0], {}),
    ("rosenbrock", [1.3, 0.7, 0.8, 1.9, 1.2], {}),
    ("rosenbrock", [-1.2, 1.0, 0.5], {}),
    ("rosenbrock", [0.0, 0.0, 0.0, 0.0], {}),
    ("rosenbrock", [-1.2, 1.0], {"gtol": 1e-8}),
    ("rosenbrock", [-1.2, 1.0], {"gtol": 1e-12}),     # asks for more than the arithmetic can give: precision loss, the second line search
    ("rosenbrock", [-1.2, 1.0], {"maxiter": 10}),
    ("rosenbrock", [2.0, 2.0, 2.0, 2.0, 2.0, 2.0], {"gtol": 1e-7}),
    ("sphere", [1.0, 2.0, 3.0], {}),
    ("sphere", [5.0], {}),
    ("sphere", [1.0, -2.0, 0.5, 4.0, -3.0, 2.0, 1.0, -1.0], {"gtol": 1e-9}),
    ("quadratic", [3.0, 3.0, 3.0, 3.0], {}),
    ("quadratic", [-2.0, 0.0], {"maxiter": 1}),
    ("quadratic", [10.0, -10.0, 10.0, -10.0, 10.0, -10.0], {"gtol": 1e-10}),
    # starts at which scipy calls its second line search (`line_search_wolfe2`, after `dcsrch` found no step), found by counting its calls over 500 seeded random starts
    ("rosenbrock", [-1.41, 1.2, 5.56, -3.65, -1.96, 0.82, 0.48, 2.83], {"gtol": 1e-12}),
    ("rosenbrock", [5.01, 3.73, 1.38, -0.8, 0.79, 1.87, 4.6, 2.67], {"gtol": 1e-10}),
    ("rosenbrock", [-5.47, -2.84, 2.78, 3.76, 2.3, 1.28, 0.76], {"gtol": 1e-10}),
    ("rosenbrock", [-4.09, -3.97, 2.75, 4.72, -2.33, -2.43, 3.3, -0.39], {"gtol": 1e-14}),
    ("rosenbrock", [0.78, -2.49, 1.66, -2.13, 2.9, -0.13], {"gtol": 1e-08}),
    ("rosenbrock", [-0.62, -1.78, -1.23, 0.56, 5.85, -5.32], {"gtol": 1e-14}),
    ("rosenbrock", [0.51, -1.77, 5.64, -0.37], {"gtol": 1e-12}),
    ("rosenbrock", [1.46, 2.08, 0.1, -0.35, -2.77, -2.34, 4.12], {"gtol": 1e-14}),
    ("rosenbrock", [-4.0, 1.45, 1.84, -2.3, -2.06, -4.71, -4.15, -1.64], {"gtol": 1e-08}),
    ("rosenbrock", [-4.99, -0.58, 5.15, 2.99], {"gtol": 1e-10}),
]


def random_cases():
    rng = np.random.default_rng(20261027)
    out = []
    for k in range(40):
        n = int(rng.integers(2, 8))
        name = ["rosenbrock", "quadratic", "sphere"][k % 3]
        x0 = [round(float(v), 3) for v in rng.uniform(-3.0, 3.0, n)]
        out.append((name, x0, {"gtol": float(rng.choice([1e-5, 1e-6, 1e-8]))}))
    return out


def run_opt(opt, name, x0, o):
    argv = [opt, "minimize", "--method", "bfgs", "--x0", ",".join(repr(v) for v in x0), "--objective", name]
    for flag in ("gtol", "maxiter"):
        if flag in o:
            argv += ["--" + flag, repr(o[flag])]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode not in (0, 1):
        return None, r.stderr
    return json.loads(r.stdout), ""


def run_scipy(name, x0, o):
    f = FUNS[name]
    opts = {"gtol": o.get("gtol", 1e-5)}
    if "maxiter" in o:
        opts["maxiter"] = o["maxiter"]
    return minimize(lambda x: f([float(v) for v in x])[0], np.array(x0), jac=lambda x: np.array(f([float(v) for v in x])[1]), method="BFGS", options=opts)


def check(opt, quiet=False, blas=False):
    sequential(not blas)
    bad = 0
    worst_x = worst_f = 0.0
    cases = CASES + random_cases()
    for name, x0, o in cases:
        got, err = run_opt(opt, name, x0, o)
        want = run_scipy(name, x0, o)
        if got is None:
            print(f"FAIL {name} {x0} {o}: opt refused: {err.strip()[:200]}")
            bad += 1
            continue
        status = {0: 0, 1: 1, 2: 2, 3: 3}[int(want.status)]
        same_counts = got["nit"] == want.nit and got["nfev"] == want.nfev and got["njev"] == want.njev and got["converged"] == bool(want.success)
        dx = float(np.max(np.abs(np.array(got["x"]) - want.x)))
        df = abs(got["fun"] - float(want.fun))
        worst_x, worst_f = max(worst_x, dx), max(worst_f, df)
        exact = got["x"] == [float(v) for v in want.x] and got["fun"] == float(want.fun) and got["jac"] == [float(v) for v in want.jac]
        if blas:
            exact = dx <= 1e-8 + 1e-6 * float(np.max(np.abs(want.x)))
            if not exact:
                bad += 1
            continue
        if not same_counts or not exact:
            if not quiet:
                print(f"FAIL {name} {x0} {o}:\n   opt   x {got['x']} f {got['fun']} nit {got['nit']} nfev {got['nfev']} njev {got['njev']} {got['converged']}\n   scipy x {list(want.x)} f {float(want.fun)} nit {want.nit} nfev {want.nfev} njev {want.njev} {bool(want.success)} (status {want.status})")
            bad += 1
    sequential(False)
    if bad == 0 and not quiet:
        print(f"ok   {len(cases)} BFGS runs agree bit for bit with scipy {__import__('scipy').__version__} (sequential dot products): minimiser, value, gradient, iterations, evaluations, status")
    return bad == 0


MUTANTS = [
    # not listed: removing `if alpha1 < 0.0 { alpha1 = 1.0; }` of `wolfe1`. It cannot change a result: it needs phi0 > old_phi0 (the last step raised the value), which a line search that accepts only
    # sufficient decrease never produces, and the first step's old value is the value plus half the gradient's norm. An equivalent mutant.
    ("lib/bfgs/bfgs.cho", "ds[15] = c1;\n    ds[16] = c2;", "ds[15] = c2;\n    ds[16] = c1;", "c1 and c2 swapped"),
    ("lib/bfgs/bfgs.cho", "ident - all[o_sk + ri] * all[o_yk + ci] * rhok;", "ident + all[o_sk + ri] * all[o_yk + ci] * rhok;", "BFGS update sign"),
    ("lib/bfgs/bfgs.cho", "all[o_hk + ri * n + ci2] = all[o_t2 + ri * n + ci2] + rhok * all[o_sk + ri] * all[o_sk + ci2];", "all[o_hk + ri * n + ci2] = all[o_t2 + ri * n + ci2];", "BFGS update without the rho s s^T term"),
    ("lib/bfgs/bfgs.cho", "var old_old_fval = old_fval + sqrt(gg) / 2.0;", "var old_old_fval = old_fval + sqrt(gg);", "initial step guess"),
    ("lib/bfgs/bfgs.cho", "} else if k >= maxiter {\n                        status = 1;", "} else if k > maxiter {\n                        status = 1;", "maxiter status"),
    ("lib/bfgs/bfgs.cho", "ib[1] = ib[1] + 1;\n        ib[3] = 1;", "ib[3] = 1;", "gradient evaluations not counted"),
    # not listed: relaxing the curvature test of the main loop of `wolfe2` (a step accepted there without a zoom). That branch is ported as scipy has it but no case reaches it: over 1,500 seeded random Rosenbrock
    # starts that call `line_search_wolfe2` it accepted a step 19 times, every time through `_zoom`. Unexercised, and said so in docs/design.md section 9.2.
    ("lib/bfgs/bfgs.cho", "if absd <= (0.0 - c2) * derphi0 {\n                a_star = a_j;", "if absd <= (0.0 - c2) * derphi0 * 0.5 {\n                a_star = a_j;", "zoom curvature test"),
    ("lib/bfgs/bfgs.cho", "let xmin = a - cc / (2.0 * bq);", "let xmin = a + cc / (2.0 * bq);", "quadmin sign"),
    ("lib/bfgs/bfgs.cho", "let radical = bb * bb - 3.0 * aa * cc;", "let radical = bb * bb - 2.0 * aa * cc;", "cubicmin radicand"),
    ("lib/bfgs/dcsrch.cho", "stpf = stpc + (stpq - stpc) / 2.0;", "stpf = stpc;", "dcstep case 1 average"),
    ("lib/bfgs/dcsrch.cho", "stp = ds[0] + p5 * (ds[3] - ds[0]);", "stp = ds[0] + p66 * (ds[3] - ds[0]);", "dcsrch bisection"),
    ("lib/bfgs/dcsrch.cho", "ds[7] = stp + xtrapu * stp;\n        return (stp, f, g, 1);", "ds[7] = stp + 2.0 * stp;\n        return (stp, f, g, 1);", "dcsrch initial stmax"),
    ("lib/nm/objective.cho", "grad[k] = 2.0 * (point[k] - float_of(k));", "grad[k] = 2.0 * point[k];", "quadratic gradient"),
    ("lib/nm/objective.cho", "grad[n - 1] = 200.0 * (point[n - 1] - point[n - 2] * point[n - 2]);", "grad[n - 1] = 100.0 * (point[n - 1] - point[n - 2] * point[n - 2]);", "rosenbrock gradient, last component"),
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
                    if text.count(old) != 1:
                        print(f"FAIL mutant '{what}': the text it changes occurs {text.count(old)} times in {f}")
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
            if check(out, quiet=True):
                print(f"FAIL mutant '{what}' SURVIVED: the check does not notice it")
                ok = False
            else:
                print(f"ok   mutant killed: {what}")
        finally:
            shutil.rmtree(d)
    return ok


def wide(opt):
    """500 seeded random Rosenbrock problems (2 to 8 parameters, starts in [-6, 6], gtol from 1e-5 to 1e-14), bit for bit against the oracle; counts how many of them make scipy call its second line search."""
    import scipy.optimize._optimize as op
    calls = [0]
    original = op.line_search_wolfe2

    def counted(*a, **k):
        calls[0] += 1
        return original(*a, **k)

    op.line_search_wolfe2 = counted
    sequential(True)
    rng = np.random.default_rng(777)
    bad = second = 0
    for _ in range(500):
        n = int(rng.integers(2, 9))
        x0 = [round(float(v), 2) for v in rng.uniform(-6, 6, n)]
        gt = float(rng.choice([1e-5, 1e-8, 1e-10, 1e-12, 1e-14]))
        calls[0] = 0
        w = run_scipy("rosenbrock", x0, {"gtol": gt})
        second += calls[0] > 0
        g, err = run_opt(opt, "rosenbrock", x0, {"gtol": gt})
        if g is None or not (g["x"] == [float(v) for v in w.x] and g["fun"] == float(w.fun) and g["nit"] == w.nit and g["nfev"] == w.nfev and g["njev"] == w.njev and g["converged"] == bool(w.success)):
            bad += 1
            print(f"FAIL wide {x0} gtol {gt}")
    op.line_search_wolfe2 = original
    sequential(False)
    print(f"{'ok  ' if not bad else 'FAIL'} 500 random Rosenbrock problems agree bit for bit with the oracle; {second} of them make scipy call its second line search; {bad} differ")
    return bad == 0


def blas_rate(opt):
    """How often `opt` and scipy with its own BLAS agree on the counts, over seeded random Rosenbrock problems (information, not a gate)."""
    rng = np.random.default_rng(5)
    sequential(False)
    equal = total = same_min = 0
    worst = 0
    for t in range(120):
        n = int(rng.integers(2, 7))
        x0 = [round(float(v), 2) for v in rng.uniform(-5, 5, n)]
        gt = [1e-5, 1e-6, 1e-8, 1e-10][t % 4]
        g, _ = run_opt(opt, "rosenbrock", x0, {"gtol": gt})
        w = run_scipy("rosenbrock", x0, {"gtol": gt})
        total += 1
        equal += g["nit"] == w.nit and g["nfev"] == w.nfev and g["njev"] == w.njev
        same_min += abs(g["fun"] - float(w.fun)) < 1e-12 or (g["fun"] < 1e-9 and float(w.fun) < 1e-9)
        worst = max(worst, abs(g["nit"] - w.nit))
    print(f"info {equal} of {total} random Rosenbrock runs (starts in [-5, 5]) have the same counts as scipy with its own BLAS, {same_min} the same minimum, the iterations differ by at most {worst}")


if __name__ == "__main__":
    opt = os.environ.get("OPT", os.path.join(ROOT, "build", "opt"))
    good = check(opt)
    if "--wide" in sys.argv:
        good = wide(opt) and good
    if "--blas" in sys.argv:
        blas_rate(opt)
    if "--mutants" in sys.argv:
        good = mutants(os.environ.get("CANCHO", "cancho")) and good
    sys.exit(0 if good else 1)
