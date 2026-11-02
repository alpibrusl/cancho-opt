#!/usr/bin/env python3
"""Gate of docs/design.md section 15: Adam against a float64 Python implementation of the same formulas (bit for bit), and against torch.optim.Adam (a tolerance).

    OPT=build/opt python3 scripts/check_adam.py [--mutants]       # needs torch for the second part; --mutants needs CANCHO=the compiler

The reference is `adam()` below: torch's single-tensor order of operations, with beta^t as a running product (what `opt` does; `pow` is not trusted to round alike). The built-in
objectives use scipy's formulas in the order `lib/nm/objective.cho` does; the program objective is examples/train/logistic.py's `loss_grad`, so the reference calls the same function.
Equal text is equal bits (the shortest form that reads back). `--mutants` builds `opt` with one deliberate bug at a time and requires the check to fail on each."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OPT = os.environ.get("OPT", os.path.join(ROOT, "build", "opt"))
SRC = ["lib/nm/objective.cho", "lib/nm/lists.cho", "lib/nm/nm.cho", "lib/bfgs/dcsrch.cho", "lib/bfgs/bfgs.cho", "lib/spsa/spsa.cho", "lib/adam/adam.cho", "lib/persist/persist.cho", "tools/opt/main.cho"]
sys.path.insert(0, os.path.join(ROOT, "examples", "train"))
import logistic  # noqa: E402


def sphere(x):
    t = 0.0
    for v in x:
        t = t + v * v
    return t, [2.0 * v for v in x]


def rosenbrock(x):
    n = len(x)
    t = 0.0
    for k in range(n - 1):
        a = x[k + 1] - x[k] * x[k]
        u = 1.0 - x[k]
        term = 100.0 * (a * a) + u * u
        t = term if k == 0 else t + term
    g = [0.0] * n
    g[0] = (0.0 - 400.0) * x[0] * (x[1] - x[0] * x[0]) - 2.0 * (1.0 - x[0])
    g[n - 1] = 200.0 * (x[n - 1] - x[n - 2] * x[n - 2])
    for k in range(1, n - 1):
        g[k] = 200.0 * (x[k] - x[k - 1] * x[k - 1]) - 400.0 * (x[k + 1] - x[k] * x[k]) * x[k] - 2.0 * (1.0 - x[k])
    return t, g


def quadratic(x):
    t = 0.0
    for k, v in enumerate(x):
        d = v - float(k)
        t = t + d * d
    return t, [2.0 * (v - float(k)) for k, v in enumerate(x)]


def logistic_fn(x):
    return logistic.loss_grad(*x)


OBJECTIVES = {"sphere": sphere, "rosenbrock": rosenbrock, "quadratic": quadratic, "logistic": logistic_fn}


def adam(fn, x0, maxiter, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    n = len(x0)
    x, m, v = list(x0), [0.0] * n, [0.0] * n
    w1, w2 = 1.0 - beta1, 1.0 - beta2
    b1t = b2t = 1.0
    for _ in range(maxiter):
        _, g = fn(x)
        b1t = b1t * beta1
        b2t = b2t * beta2
        step = lr / (1.0 - b1t)
        root = math.sqrt(1.0 - b2t)
        for k in range(n):
            d = g[k] - m[k]
            m[k] = g[k] - d * beta1 if w1 >= 0.5 else m[k] + w1 * d
            v[k] = v[k] * beta2 + w2 * g[k] * g[k]
            x[k] = x[k] + (0.0 - step) * (m[k] / (math.sqrt(v[k]) / root + eps))
    f, g = fn(x)
    return x, f, g


# (objective, start, iterations, options)
CASES = [
    ("sphere", [1.0, -2.0, 3.0], 1, {}),
    ("sphere", [1.0, -2.0, 3.0], 100, {"lr": 0.05}),
    ("sphere", [0.5], 300, {"lr": 0.01, "beta1": 0.5, "beta2": 0.9}),
    ("sphere", [0.5, 0.25, -0.75, 2.0, 3.0, -1.0, 0.1, 0.2], 40, {"lr": 0.1, "beta1": 0.3}),
    ("rosenbrock", [-1.2, 1.0], 100, {"lr": 0.01}),
    ("rosenbrock", [-1.2, 1.0, 0.5, -0.3], 250, {"lr": 0.005, "beta2": 0.99, "eps": 1e-6}),
    ("rosenbrock", [0.0, 0.0, 0.0], 17, {"lr": 0.02}),
    ("quadratic", [5.0, 5.0, 5.0], 200, {"lr": 0.1}),
    ("quadratic", [-1.0, 4.0], 60, {"lr": 0.2, "beta1": 0.0}),
    ("logistic", [0.0, 0.0, 0.0], 150, {"lr": 0.05}),
    ("logistic", [1.0, -1.0, 0.5], 80, {"lr": 0.1, "beta1": 0.8, "beta2": 0.95, "eps": 1e-7}),
]


def run_opt(opt, name, x0, maxiter, options):
    argv = [opt, "minimize", "--method", "adam", "--x0", ",".join(repr(v) for v in x0), "--maxiter", str(maxiter)]
    for flag, key in (("--lr", "lr"), ("--beta1", "beta1"), ("--beta2", "beta2"), ("--eps", "eps")):
        if key in options:
            argv += [flag, repr(options[key])]
    if name == "logistic":
        argv += ["--", sys.executable, os.path.join(ROOT, "examples", "train", "logistic.py")]
    else:
        argv += ["--objective", name]
    return subprocess.run(argv, capture_output=True, text=True)


def check(opt, quiet=False):
    bad = 0
    for name, x0, maxiter, options in CASES:
        r = run_opt(opt, name, x0, maxiter, options)
        label = f"{name} n={len(x0)} {maxiter} iterations {options}"
        if r.returncode != 0:
            print(f"FAIL {label}: status {r.returncode} {r.stderr[:120]}")
            bad += 1
            continue
        got = json.loads(r.stdout)
        x, f, g = adam(OBJECTIVES[name], x0, maxiter, **options)
        same = got["x"] == x and got["fun"] == f and got["jac"] == g and got["nit"] == maxiter and got["nfev"] == maxiter + 1 and got["njev"] == maxiter + 1 and got["completed"] is True
        if same:
            if not quiet:
                print(f"ok   {label}: x, fun and jac equal the reference bit for bit")
        else:
            bad += 1
            print(f"FAIL {label}: got x {got['x'][:3]} fun {got['fun']}, want x {x[:3]} fun {f}")
    return bad == 0


def persistent_same(opt):
    """--persistent changes how the objective is called, not what comes out: byte for byte the same result."""
    base = [opt, "minimize", "--method", "adam", "--x0", "0,0,0", "--maxiter", "60", "--lr", "0.05"]
    prog = ["--", sys.executable, os.path.join(ROOT, "examples", "train", "logistic.py")]
    a = subprocess.run(base + prog, capture_output=True, text=True)
    server = os.path.join(tempfile.mkdtemp(), "serve.py")
    open(server, "w").write(f"import sys\nsys.path.insert(0, {os.path.join(ROOT, 'examples', 'train')!r})\nimport logistic\nfor line in sys.stdin:\n    f, g = logistic.loss_grad(*map(float, line.split()))\n    print(repr(f), *map(repr, g), flush=True)\n")
    b = subprocess.run(base + ["--persistent"] + ["--", sys.executable, server], capture_output=True, text=True)
    good = a.returncode == 0 and a.stdout == b.stdout and b.stdout != ""
    print(f"{'ok  ' if good else 'FAIL'} --persistent gives byte for byte the result of a process for each point (61 evaluations)")
    return good


def against_torch():
    try:
        import torch
    except ImportError:
        print("skip torch.optim.Adam: torch is not installed")
        return True
    ok = True
    for name, x0, maxiter, options in (("sphere", [1.0, -2.0, 3.0], 100, {"lr": 0.05}), ("quadratic", [5.0, 5.0, 5.0], 100, {"lr": 0.1}),
                                       ("logistic", [0.0, 0.0, 0.0], 100, {"lr": 0.05}), ("rosenbrock", [-1.2, 1.0], 20, {"lr": 0.01})):
        r = run_opt(OPT, name, x0, maxiter, options)
        got = json.loads(r.stdout)["x"]
        p = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        o = torch.optim.Adam([p], lr=options.get("lr", 0.001), betas=(options.get("beta1", 0.9), options.get("beta2", 0.999)), eps=options.get("eps", 1e-8))
        for _ in range(maxiter):
            p.grad = torch.tensor(OBJECTIVES[name](p.detach().tolist())[1], dtype=torch.float64)
            o.step()
        want = p.detach().tolist()
        worst = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(got, want))
        good = worst <= 1e-9
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name} {maxiter} iterations against torch.optim.Adam: worst relative difference {worst:.1e}")
    return ok


MUTANTS = [
    ("lib/adam/adam.cho", "var mk = w[mo + k] + w1 * d;", "var mk = w[mo + k] + beta1 * d;", "first moment weight"),
    ("lib/adam/adam.cho", "let vk = w[vo + k] * beta2 + w2 * g * g;", "let vk = w[vo + k] * beta2 + w2 * g;", "second moment forgets a factor of g"),
    ("lib/adam/adam.cho", "let step = lr / (1.0 - b1t);", "let step = lr;", "no bias correction of the first moment"),
    ("lib/adam/adam.cho", "let root = math.sqrt(1.0 - b2t);", "let root = 1.0 - b2t;", "bias correction of the second moment without the root"),
    ("lib/adam/adam.cho", "let denom = math.sqrt(vk) / root + eps;", "let denom = math.sqrt(vk) / root;", "no eps"),
    ("lib/adam/adam.cho", "w[k] = w[k] + (0.0 - step) * (mk / denom);", "w[k] = w[k] + step * (mk / denom);", "step sign"),
    ("lib/adam/adam.cho", "mk = g - d * beta1;", "mk = g - d * w1;", "the other branch of the first moment"),
    ("lib/adam/adam.cho", "b2t = b2t * beta2;", "b2t = b2t * beta1;", "running product of beta2"),
    ("lib/adam/adam.cho", "let (c3, f3) = objective.value_gradient(heap, io, exec, clock, link, kind, exe, argv, w[0..n], jac);", "let (c3, f3) = (0, 0.0);", "no final evaluation"),
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


if __name__ == "__main__":
    good = check(OPT)
    good &= persistent_same(OPT)
    good &= against_torch()
    if "--mutants" in sys.argv:
        good &= mutants(os.environ.get("CANCHO", "cancho"))
    print("check_adam:", "ok" if good else "FAILED")
    sys.exit(0 if good else 1)
