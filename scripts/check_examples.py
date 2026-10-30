#!/usr/bin/env python3
"""Gate of docs/design.md section 12: every program of examples/train/ is run
through `opt` and the claim its table row makes is checked. It can fail: the last
check feeds `gradcheck.py` a gradient that is wrong on purpose.

    OPT=build/opt python3 scripts/check_examples.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile

OPT = os.environ.get("OPT", "build/opt")
PY = sys.executable  # opt does not search PATH for the program: an absolute path
HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(os.path.dirname(HERE), "examples", "train")
sys.path.insert(0, EX)
import logistic  # noqa: E402
import mlp_xor  # noqa: E402

failed = 0


def check(name, good, detail=""):
    global failed
    print(f"{'ok  ' if good else 'FAIL'} {name} {detail}")
    failed += 0 if good else 1


def opt(*args):
    r = subprocess.run([OPT, "minimize", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"opt {args}: {r.stdout}{r.stderr}")
    return json.loads(r.stdout)


def newton_logistic():
    """The independent solve: Newton's method on the same loss, Hessian written out."""
    w = [0.0, 0.0, 0.0]
    for _ in range(50):
        _, g = logistic.loss_grad(*w)
        h = [[0.0] * 3 for _ in range(3)]
        for x1, x2, _y in logistic.DATA:
            p = 1 / (1 + math.exp(-(w[0] * x1 + w[1] * x2 + w[2])))
            for i, a in enumerate((x1, x2, 1.0)):
                for j, c in enumerate((x1, x2, 1.0)):
                    h[i][j] += p * (1 - p) * a * c / len(logistic.DATA)
        h[0][0] += logistic.L2
        h[1][1] += logistic.L2
        # solve h d = g by Gaussian elimination
        m = [row + [g[i]] for i, row in enumerate(h)]
        for i in range(3):
            piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
            m[i], m[piv] = m[piv], m[i]
            for r in range(3):
                if r != i:
                    f = m[r][i] / m[i][i]
                    m[r] = [a - f * b for a, b in zip(m[r], m[i])]
        w = [w[i] - m[i][3] / m[i][i] for i in range(3)]
    return logistic.loss_grad(*w)[0]


# 1. a convex fit
r = opt("--method", "bfgs", "--x0", "0,0,0", "--", PY, os.path.join(EX, "logistic.py"))
want = newton_logistic()
check("logistic bfgs converged", r["converged"], f"nfev {r['nfev']}")
check("logistic gradient below gtol", max(abs(g) for g in r["jac"]) < 1e-5, f"largest {max(abs(g) for g in r['jac']):.2e}")
check("logistic loss equals an independent Newton solve", abs(r["fun"] - want) < 1e-9, f"{r['fun']!r} vs {want!r}")

# 2. a network: a fixed list of starts, and how many reach the solution
STARTS = ["0.5,-0.4,0.3,0.8,0.1,-0.1,0.7,-0.6,0", "1,1,-1,-1,0.5,-0.5,1,-1,0", "2,-2,2,-2,-1,1,2,2,-1",
          "0.3,0.2,-0.3,0.4,0.1,0.2,0.5,0.5,0.1", "-1,1,1,-1,0,0,1,1,0", "3,3,-3,-3,-1,5,3,-3,0",
          "0.9,-0.8,0.7,-0.6,0.5,-0.4,0.3,-0.2,0.1", "1.5,0.5,-0.5,1.5,0.2,-0.2,1,1,-1"]
solved = stuck_converged = 0
for s in STARTS:
    r = opt("--method", "bfgs", "--x0", s, "--", PY, os.path.join(EX, "mlp_xor.py"))
    right = all((mlp_xor.forward(r["x"], a, b)[1] > 0.5) == bool(y) for a, b, y in mlp_xor.DATA)
    if r["fun"] < 1e-3 and right:
        solved += 1
    elif r["converged"]:
        stuck_converged += 1
check("mlp_xor: at least one start classifies the four points", solved >= 1, f"{solved} of {len(STARTS)} starts solve it")
check("mlp_xor: BFGS reports converged at a wrong answer", stuck_converged >= 1,
      f"{stuck_converged} of {len(STARTS)} starts converge to a local minimum that does not solve it")

# 3. noise, no gradient
r = opt("--method", "spsa", "--x0", "0,0,0", "--maxiter", "300", "--", PY, os.path.join(EX, "noisy_logistic.py"))
true_start, true_end = logistic.loss_grad(0, 0, 0)[0], logistic.loss_grad(*r["x"])[0]
check("spsa lowers the true loss", true_end < true_start - 0.05, f"{true_start:.4f} -> {true_end:.4f} (the optimum is {newton_logistic():.4f})")

# 4. a corner
r = opt("--x0", "0.5,0.1,-0.5,0.1,0.8,0.8,0", "--maxfev", "4000", "--", PY, os.path.join(EX, "relu_fit.py"))
check("nelder-mead fits |x| with a ReLU network", r["converged"] and r["fun"] < 1e-8, f"loss {r['fun']:.2e}, nfev {r['nfev']}")

# 5. the gradient check, and that it can fail
def gradcheck(prog, x0):
    return subprocess.run([PY, os.path.join(EX, "gradcheck.py"), prog, x0], capture_output=True, text=True).returncode


check("gradcheck passes logistic.py", gradcheck(os.path.join(EX, "logistic.py"), "0.3,-0.2,0.1") == 0)
check("gradcheck passes mlp_xor.py", gradcheck(os.path.join(EX, "mlp_xor.py"), "0.5,-0.4,0.3,0.8,0.1,-0.1,0.7,-0.6,0.2") == 0)
src = open(os.path.join(EX, "logistic.py")).read()
assert "g[2] += p - y" in src
with tempfile.TemporaryDirectory() as d:
    bad = os.path.join(d, "wrong.py")
    open(bad, "w").write(src.replace("g[2] += p - y", "g[2] += 2 * (p - y)"))
    check("gradcheck fails a gradient that is wrong on purpose", gradcheck(bad, "0.3,-0.2,0.1") == 1)

sys.exit(1 if failed else 0)
