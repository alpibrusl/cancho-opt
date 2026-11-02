#!/usr/bin/env python3
"""Gate of docs/design.md section 11: SPSA against qiskit_algorithms' SPSA (an independent implementation of the same algorithm; scipy has none).

    OPT=build/opt python3 scripts/check_spsa.py [--mutants]       # needs qiskit-algorithms; --mutants needs CANCHO=the compiler

The oracle is run with the same gain sequences (a / (k + A)^0.602, c / k^0.101) and with its source of perturbations replaced by opt's own generator
(xorshift64*, entry i is +1 when the draw is at least 0.5). Every point `opt` asks the objective about is logged by the objective program and must equal,
in order, the point the oracle asks about, to a relative 1e-9 (not bit for bit: the gains use `pow`, which Python's C library and `std.math` may round differently in the last
place); the result must agree likewise, `nfev` must be the oracle's plus the final evaluation, and `nit` must be equal. Then the gradient estimate is checked to be unbiased
on a noisy objective, over many seeds. `--mutants` builds `opt` with one deliberate bug at a time and requires the check to fail on each."""
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from qiskit_algorithms.optimizers import SPSA
import qiskit_algorithms.optimizers.spsa as spsa_module

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OPT = os.environ.get("OPT", os.path.join(ROOT, "build", "opt"))
SRC = ["lib/nm/objective.cho", "lib/nm/lists.cho", "lib/nm/nm.cho", "lib/bfgs/dcsrch.cho", "lib/bfgs/bfgs.cho", "lib/spsa/spsa.cho", "lib/persist/persist.cho", "tools/opt/main.cho"]
M = (1 << 64) - 1

FUNCTIONS = {
    "sphere": "t = sum(v * v for v in x)",
    "rosenbrock": "t = sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))",
    "quartic": "t = sum((v - 0.5) ** 4 + 0.1 * v * v + math.sin(3.0 * v) for v in x)",
}


class Stream:
    """opt's generator: xorshift64* (cancho-quantum's qc.measure.next)."""

    def __init__(self, seed):
        self.s = seed

    def draw(self):
        s = self.s
        s ^= s >> 12
        s ^= (s << 25) & M
        s ^= s >> 27
        self.s = s
        y = (s * 2685821657736338717) & M
        return (y >> 11) / 9007199254740992.0


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


def oracle(name, x0, maxiter, a, c, big_a, seed):
    """(x, fun, nfev, nit, points) of qiskit's SPSA with opt's perturbation stream."""
    stream = Stream(seed)
    code = FUNCTIONS[name]
    points = []

    def fun(x):
        env = {"x": [float(v) for v in x], "math": math}
        exec(code, env)
        points.append([float(v) for v in x])
        return env["t"]

    def perturbation(dim, perturbation_dims=None):
        return np.array([1 if stream.draw() >= 0.5 else -1 for _ in range(dim)])

    saved = spsa_module.bernoulli_perturbation
    spsa_module.bernoulli_perturbation = perturbation
    try:
        eta, eps = gains(a, c, big_a)
        r = SPSA(maxiter=maxiter, learning_rate=eta, perturbation=eps).minimize(fun, np.array(x0, dtype=float))
    finally:
        spsa_module.bernoulli_perturbation = saved
    return [float(v) for v in r.x], float(r.fun), int(r.nfev) + 1, int(r.nit), points


def run_opt(opt, name, x0, maxiter, a, c, big_a, seed, tmp):
    log = os.path.join(tmp, "points.log")
    prog = os.path.join(tmp, "obj.py")
    open(log, "w").close()
    open(prog, "w").write(f"import sys, math\nx=[float(v) for v in sys.stdin.read().split()]\n{FUNCTIONS[name]}\nopen({log!r}, 'a').write(' '.join(repr(v) for v in x) + '\\n')\nprint(repr(t))\n")
    argv = [opt, "minimize", "--method", "spsa", "--x0", ",".join(repr(v) for v in x0), "--maxiter", str(maxiter), "--a", repr(a), "--c", repr(c), "--seed", str(seed)]
    if big_a is not None:
        argv += ["--stability", repr(big_a)]
    r = subprocess.run(argv + ["--", sys.executable, prog], capture_output=True, text=True)
    points = [[float(v) for v in line.split()] for line in open(log).read().splitlines()]
    return r, points


def close(u, v):
    return all(abs(p - q) <= 1e-9 * max(1.0, abs(p), abs(q)) for p, q in zip(u, v)) and len(u) == len(v)


CASES = [
    ("sphere", [1.0, 1.0, 1.0], 30, 0.2, 0.1, None, 1),
    ("sphere", [0.5, -0.5], 50, 0.5, 0.2, 5.0, 7),
    ("rosenbrock", [-1.2, 1.0], 60, 0.01, 0.1, None, 3),
    ("rosenbrock", [0.0, 0.0, 0.0, 0.0], 40, 0.005, 0.05, 2.0, 12345),
    ("quartic", [0.3, -0.2, 0.9, 1.5, -1.1], 80, 0.1, 0.15, None, 99),
    ("quartic", [2.0], 25, 0.3, 0.1, 0.0, 424242),
    ("sphere", [1.0] * 12, 100, 0.2, 0.1, None, 2),
]


def check(opt, quiet=False):
    ok = True
    for name, x0, maxiter, a, c, big_a, seed in CASES:
        stab = 0.1 * maxiter if big_a is None else big_a
        want_x, want_f, want_nfev, want_nit, want_pts = oracle(name, x0, maxiter, a, c, stab, seed)
        tmp = tempfile.mkdtemp()
        try:
            r, pts = run_opt(opt, name, x0, maxiter, a, c, big_a, seed, tmp)
            if r.returncode != 0:
                good, why = False, f"status {r.returncode}: {r.stderr[:120]}"
            else:
                got = json.loads(r.stdout)
                # the oracle's last point is its final evaluation, which opt also makes: the logs have the same length
                good = (len(pts) == len(want_pts) and all(close(p, q) for p, q in zip(pts, want_pts)) and close(got["x"], want_x)
                        and close([got["fun"]], [want_f]) and got["nfev"] == want_nfev and got["nit"] == want_nit and got["completed"] is True and "converged" not in got)
                why = f"points {len(pts)} vs {len(want_pts)}, nfev {got['nfev']} vs {want_nfev}, nit {got['nit']} vs {want_nit}"
        finally:
            shutil.rmtree(tmp)
        if good:
            if not quiet:
                print(f"ok   {name} n={len(x0)} maxiter {maxiter} seed {seed}: {len(want_pts)} points, x, fun, nfev, nit agree with qiskit's SPSA")
        else:
            if not quiet:
                print(f"FAIL {name} n={len(x0)} maxiter {maxiter} seed {seed}: {why}")
            ok = False
            if quiet:
                return False
    ok = unbiased(opt, quiet) and ok
    return ok


def unbiased(opt, quiet):
    """The estimate (f(x + c d) - f(x - c d)) / (2 c) * d of the gradient of f = sum of v^3 / 3 + noise: over many seeds, one iteration from x0 moves x by a * estimate (A = 0, k = 1, a_1 = a),
    so the mean step over 1000 seeds, divided by -a, is the gradient up to the O(c^2) of the cubic (and the noise, which has mean zero, averages out)."""
    tmp = tempfile.mkdtemp()
    try:
        prog = os.path.join(tmp, "noisy.py")
        # the noise is a deterministic function of the point (a hash), so that the program is repeatable and the seeds are the only randomness
        open(prog, "w").write("import sys, struct, zlib\nx=[float(v) for v in sys.stdin.read().split()]\nt=sum(v**3/3 for v in x)\n"
                              "h=zlib.crc32(struct.pack('<%dd' % len(x), *x))\nt+=0.05*((h / 4294967295.0) - 0.5)\nprint(repr(t))\n")
        x0 = [1.0, -0.5, 2.0]
        a = 0.01
        total = np.zeros(3)
        seeds = 600
        for seed in range(1, seeds + 1):
            r = subprocess.run([opt, "minimize", "--method", "spsa", "--x0", ",".join(map(repr, x0)), "--maxiter", "1", "--a", repr(a), "--c", "0.1", "--stability", "0", "--seed", str(seed), "--", sys.executable, prog], capture_output=True, text=True)
            total += np.array(json.loads(r.stdout)["x"]) - np.array(x0)
        mean_grad = -(total / seeds) / a
        want = np.array([v * v for v in x0])
        # standard error: each estimate has a variance of about (noise / c)^2 per entry; 600 seeds make the mean good to about 0.1
        good = bool(np.all(np.abs(mean_grad - want) < 0.5))
        if not quiet:
            print(f"{'ok  ' if good else 'FAIL'} gradient estimate averaged over {seeds} seeds: {np.round(mean_grad, 2)} against the gradient {want}")
        return good
    finally:
        shutil.rmtree(tmp)


MUTANTS = [
    ("lib/spsa/spsa.cho", "if draw(g) >= 0.5 {", "if draw(g) > 0.6 {", "perturbation threshold"),
    ("lib/spsa/spsa.cho", "math.pow(float_of(it) + big_a, 0.602)", "math.pow(float_of(it) + big_a, 0.6)", "gain exponent"),
    ("lib/spsa/spsa.cho", "math.pow(float_of(it), 0.101)", "math.pow(float_of(it), 0.1)", "perturbation exponent"),
    ("lib/spsa/spsa.cho", "math.pow(float_of(it) + big_a, 0.602)", "math.pow(float_of(it) + big_a + 1.0, 0.602)", "gain offset"),
    ("lib/spsa/spsa.cho", "(fplus - fminus) / (2.0 * ck)", "(fplus - fminus) / ck", "difference quotient"),
    ("lib/spsa/spsa.cho", "w[k] = w[k] - ak * estimate;", "w[k] = w[k] + ak * estimate;", "step sign"),
    ("lib/spsa/spsa.cho", "w[trial + k] = w[k] - ck * w[delta + k];", "w[trial + k] = w[k] - ck * w[delta + k] * 0.5;", "minus probe size"),
    ("lib/spsa/spsa.cho", "s = s ^ s << 25;", "s = s ^ s << 24;", "generator shift"),
    ("lib/spsa/spsa.cho", "let (c3, f) = objective.value(heap, io, exec, clock, kind, exe, argv, w[0..n]);", "let (c3, f) = objective.value(heap, io, exec, clock, kind, exe, argv, w[trial..trial + n]);", "final evaluation point"),
    ("lib/spsa/spsa.cho", "let rng = box_slice(heap, 1, seed);", "let rng = box_slice(heap, 1, seed + 1);", "seed"),
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
    if "--mutants" in sys.argv:
        good = mutants(os.environ.get("CANCHO", "cancho")) and good
    sys.exit(0 if good else 1)
