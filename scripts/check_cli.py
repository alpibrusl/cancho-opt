#!/usr/bin/env python3
"""Gate 2 of docs/design.md section 6: the command line. Every refusal has its rule tag, status 2 and nothing on standard output; a program
as the objective gives the same answer as the built-in one it computes; a failing objective is reported with its point.

    OPT=build/opt python3 scripts/check_cli.py
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OPT = os.environ.get("OPT", os.path.join(ROOT, "build", "opt"))
bad = 0


def opt(*argv, stdin=""):
    return subprocess.run([OPT, *argv], capture_output=True, text=True, input=stdin)


def expect_refusal(tag, *argv):
    global bad
    r = opt(*argv)
    try:
        err = json.loads(r.stderr)
    except ValueError:
        err = {}
    if r.returncode == 2 and r.stdout == "" and err.get("error") == tag and err.get("message") and err.get("repair"):
        print(f"ok   refused as {tag}: {' '.join(argv)[:70]}")
    else:
        print(f"FAIL expected {tag} (status 2, empty stdout): status {r.returncode}, stdout {r.stdout[:80]!r}, stderr {r.stderr[:160]!r}")
        bad += 1
    return err


X0 = "--x0", "1,2"
expect_refusal("opt.usage")
expect_refusal("opt.usage", "frobnicate")
expect_refusal("opt.usage", "minimize", *X0)
expect_refusal("opt.usage", "minimize", "--objective", "sphere")
expect_refusal("opt.usage", "minimize", *X0, "--objective", "nosuch")
expect_refusal("opt.usage", "minimize", *X0, "--objective", "sphere", "--", "true")
expect_refusal("opt.usage", "minimize", "--x0")
expect_refusal("opt.unknown-flag", "minimize", *X0, "--objective", "sphere", "--tolerance", "1")
expect_refusal("opt.unknown-method", "minimize", *X0, "--objective", "sphere", "--method", "cg")
expect_refusal("opt.bad-number", "minimize", "--x0", "1,x", "--objective", "sphere")
expect_refusal("opt.bad-number", "minimize", "--x0", "1,", "--objective", "sphere")
expect_refusal("opt.bad-number", "minimize", "--x0", "1,1e999", "--objective", "sphere")
expect_refusal("opt.bad-number", *("minimize", *X0, "--objective", "sphere", "--step", "0"))
expect_refusal("opt.bad-number", *("minimize", *X0, "--objective", "sphere", "--xatol", "-1"))
expect_refusal("opt.bad-number", *("minimize", *X0, "--objective", "sphere", "--maxfev", "0"))
expect_refusal("opt.bad-number", *("minimize", *X0, "--objective", "sphere", "--maxiter", "1.5"))
expect_refusal("opt.bounds-length", *("minimize", *X0, "--objective", "sphere", "--lower", "0"))
expect_refusal("opt.bounds-order", *("minimize", *X0, "--objective", "sphere", "--lower", "3,0", "--upper", "1,1"))
expect_refusal("opt.too-many-parameters", "minimize", "--x0", ",".join(["1"] * 33), "--objective", "sphere")

# BFGS
expect_refusal("opt.no-gradient", "minimize", "--method", "bfgs", *X0, "--objective", "absolute")
expect_refusal("opt.flag-not-for-method", "minimize", "--method", "bfgs", *X0, "--objective", "sphere", "--xatol", "1e-6")
expect_refusal("opt.flag-not-for-method", "minimize", "--method", "bfgs", *X0, "--objective", "sphere", "--lower", "0,0")
expect_refusal("opt.flag-not-for-method", "minimize", *X0, "--objective", "sphere", "--gtol", "1e-6")
expect_refusal("opt.bad-number", "minimize", "--method", "bfgs", *X0, "--objective", "sphere", "--gtol", "-1")
expect_refusal("opt.bad-number", "minimize", "--method", "bfgs", *X0, "--objective", "sphere", "--gtol", "x")
expect_refusal("opt.unknown-method", "minimize", "--method", "lm", *X0, "--objective", "sphere")
expect_refusal("opt.no-gradient", "minimize", "--method", "bfgs", "--x0", "1", "--objective", "rosenbrock")

# a program objective computes the built-in function
tmp = tempfile.mkdtemp()
prog = os.path.join(tmp, "obj.py")
open(prog, "w").write("import sys\nx=[float(v) for v in sys.stdin.read().split()]\nt=0.0\nfor v in x:\n    t=t+v*v\nprint(repr(t))\n")
a = opt("minimize", "--x0", "1,2,3", "--objective", "sphere")
b = opt("minimize", "--x0", "1,2,3", "--", sys.executable, prog)
if a.returncode == 0 and b.returncode == 0 and json.loads(a.stdout) == json.loads(b.stdout):
    print("ok   a program that computes the sum of squares gives the same answer as the built-in objective (bit for bit, x, fun, nfev, nit)")
else:
    print("FAIL program objective differs:", a.stdout[:150], b.stdout[:150], b.stderr[:150])
    bad += 1

# BFGS with a program that prints the value and then the gradient
gprog = os.path.join(tmp, "grad.py")
open(gprog, "w").write("import sys\nx=[float(v) for v in sys.stdin.read().split()]\nt=0.0\nfor v in x:\n    t=t+v*v\nprint(repr(t), *[repr(2.0*v) for v in x])\n")
a = opt("minimize", "--method", "bfgs", "--x0", "1,2,3", "--objective", "sphere")
b = opt("minimize", "--method", "bfgs", "--x0", "1,2,3", "--", sys.executable, gprog)
if a.returncode == 0 and b.returncode == 0 and json.loads(a.stdout) == json.loads(b.stdout):
    print("ok   a program that prints the value and the gradient of the sum of squares gives the same BFGS answer as the built-in objective (bit for bit)")
else:
    print("FAIL BFGS program objective differs:", a.stdout[:150], b.stdout[:150], b.stderr[:150])
    bad += 1
p1 = os.path.join(tmp, "g1.py")
open(p1, "w").write("print('1.0')\n")
expect_refusal("opt.objective-output", "minimize", "--method", "bfgs", "--x0", "1,2", "--", sys.executable, p1)
p2 = os.path.join(tmp, "g2.py")
open(p2, "w").write("print('1.0 nan 2.0')\n")
expect_refusal("opt.objective-output", "minimize", "--method", "bfgs", "--x0", "1,2", "--", sys.executable, p2)
r = opt("minimize", "--method", "bfgs", "--x0", "-1.2,1", "--objective", "rosenbrock", "--maxiter", "5")
if r.returncode == 1 and json.loads(r.stdout)["converged"] is False and json.loads(r.stdout)["nit"] == 5:
    print("ok   BFGS with a limit reached: status 1, the result printed, converged false, nit exactly the limit")
else:
    print("FAIL BFGS maxiter:", r.returncode, r.stdout[:200])
    bad += 1

# failures of the program, with the point
fails = {
    "exits": ("import sys\nsys.exit(3)\n", "opt.objective-failed"),
    "two numbers": ("print('1 2')\n", "opt.objective-output"),
    "text": ("print('hello')\n", "opt.objective-output"),
    "nothing": ("pass\n", "opt.objective-output"),
    "nan": ("print('NaN')\n", "opt.objective-output"),
    "infinite": ("print('1e999')\n", "opt.objective-not-finite"),
}
for what, (code, tag) in fails.items():
    p = os.path.join(tmp, "f.py")
    open(p, "w").write(code)
    err = expect_refusal(tag, "minimize", "--x0", "0.5,2", "--", sys.executable, p)
    if err.get("point") != [0.5, 2.0]:
        print(f"FAIL {what}: the error does not name the first point: {err.get('point')}")
        bad += 1
err = expect_refusal("opt.objective-failed", "minimize", "--x0", "1", "--", "/nonexistent/program")

# a limit reached is status 1 and still prints
r = opt("minimize", "--x0", "-1.2,1", "--objective", "rosenbrock", "--maxfev", "20")
if r.returncode == 1 and json.loads(r.stdout)["converged"] is False and json.loads(r.stdout)["nfev"] == 20:
    print("ok   a limit reached: status 1, the result printed, converged false, nfev exactly the limit")
else:
    print("FAIL maxfev:", r.returncode, r.stdout[:200])
    bad += 1

# no input reaches a panic: many odd arguments
odd = [["minimize", "--x0", v, "--objective", "sphere"] for v in ["", ",", "1,,2", "-", "1e", "--", "0x10", "1" * 400, "[1]", "nan", "inf", "-inf"]]
odd += [["minimize", "--x0", "1", "--objective", "sphere", "--lower", v] for v in ["", ",,", "inf", "1", "a"]]
for argv in odd:
    r = opt(*argv)
    if r.returncode not in (0, 1, 2) or "panic" in r.stderr.lower() or "trap" in r.stderr.lower():
        print("FAIL a panic or a trap:", argv, r.returncode, r.stderr[:100])
        bad += 1
print(f"ok   {len(odd)} odd arguments: no panic" if bad == 0 else "")
sys.exit(1 if bad else 0)
