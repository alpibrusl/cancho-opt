#!/usr/bin/env python3
"""Gate of docs/design.md section 14: `opt minimize --persistent`, one process that answers a line for each line.

    OPT=build/opt python3 scripts/check_persistent.py [--mutants]       # --mutants needs CANCHO=the compiler

Checks, each of which can fail:
  1. the same answer: with --persistent, Nelder-Mead, BFGS and SPSA print byte for byte what they print when the program is started for each point;
  2. one process: the program is started once (it logs its starts), the per-point program nfev times;
  3. the end: the program sees its input close (and says so), and one that ignores that is killed after five seconds;
  4. the refusals, each with its rule tag and the point: the program exits, answers with the wrong count, answers with two lines, answers infinity (1e999),
     does not answer within --timeout-ms, or is not there; and the flags that go wrong;
  5. the time: 159 evaluations through a persistent program take under half of the time of a process each.
`--mutants` builds `opt` with one deliberate bug at a time and requires the check to fail on each."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OPT = os.environ.get("OPT", os.path.join(ROOT, "build", "opt"))
PY = sys.executable
SRC = ["lib/nm/objective.cho", "lib/nm/lists.cho", "lib/nm/nm.cho", "lib/bfgs/dcsrch.cho", "lib/bfgs/bfgs.cho", "lib/spsa/spsa.cho", "lib/persist/persist.cho",
       "tools/opt/main.cho"]

# One program, two ways to run it: `MODE once` reads a point and answers; `MODE loop` answers a line for each line until its input closes.
# argv: mode, behaviour, log file, [after]. Behaviours: ok, grad (value then gradient: Rosenbrock), exit-after N, short (value only, for BFGS),
# two (two lines for a line), nan, silent (never answers), deaf (ignores the end of its input and sleeps).
PROGRAM = r'''
import math, os, sys, time
mode, behaviour, log = sys.argv[1], sys.argv[2], sys.argv[3]
after = int(sys.argv[4]) if len(sys.argv) > 4 else 0
def note(text):
    with open(log, "a") as f:
        f.write(text + "\n")
def rosen(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))
def rosen_der(x):
    n = len(x); g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return g
def answer(x, count):
    if behaviour == "grad":
        return " ".join(repr(v) for v in [rosen(x)] + rosen_der(x))
    if behaviour == "nan":
        return "1e999"
    return repr(rosen(x))
note("start %d" % os.getpid())
if mode == "once":
    x = [float(v) for v in sys.stdin.read().split()]
    print(answer(x, 0))
else:
    count = 0
    for line in sys.stdin:
        x = [float(v) for v in line.split()]
        count += 1
        if behaviour == "silent":
            time.sleep(60)
        if behaviour == "exit-after" and count > after:
            sys.exit(3)
        print(answer(x, count), flush=True)
        if behaviour == "two":
            print(answer(x, count), flush=True)
        if behaviour == "deaf" and count == 1:
            pass
    note("eof")
    if behaviour == "deaf":
        time.sleep(30)
'''

bad = 0
QUIET = False


def say(good, text):
    global bad
    if not QUIET:
        print(f"{'ok  ' if good else 'FAIL'} {text}")
    bad += 0 if good else 1
    return good


def run(opt, *argv, limit=40):
    start = time.time()
    try:
        r = subprocess.run([opt, *argv], capture_output=True, text=True, timeout=limit)
        return r.returncode, r.stdout, r.stderr, time.time() - start
    except subprocess.TimeoutExpired:
        return None, "", "", time.time() - start


def check(opt, quiet=False):
    global bad, QUIET
    bad = 0
    QUIET = quiet
    d = tempfile.mkdtemp()
    out = print if not quiet else (lambda *a, **k: None)
    try:
        prog = os.path.join(d, "prog.py")
        open(prog, "w").write(PROGRAM)
        log = os.path.join(d, "log")

        def program(mode, behaviour, *extra):
            return ["--", PY, prog, mode, behaviour, log, *extra]

        def fresh():
            if os.path.exists(log):
                os.remove(log)

        def lines():
            return open(log).read().split("\n")[:-1] if os.path.exists(log) else []

        # 1. the same answer, three methods
        cases = [("nelder-mead", "ok", ["--x0", "-1.2,1"]), ("nelder-mead", "ok", ["--x0", "-1.2,1,0.5", "--lower", "-2,-2,-2", "--upper", "2,2,2"]),
                 ("bfgs", "grad", ["--method", "bfgs", "--x0", "-1.2,1"]), ("spsa", "ok", ["--method", "spsa", "--x0", "-1.2,1", "--maxiter", "40", "--seed", "7"])]
        for name, behaviour, flags in cases:
            a = run(opt, "minimize", *flags, *program("once", behaviour))
            b = run(opt, "minimize", "--persistent", *flags, *program("loop", behaviour))
            same = a[0] is not None and a[0] == b[0] and a[1] == b[1] and a[1] != ""
            say(same, f"the same answer with --persistent: {name} {' '.join(flags)[:50]}" + ("" if same else f"\n       once  {a[:2]}\n       loop  {b[:2]}"))

        # 2. one process
        fresh()
        a = run(opt, "minimize", "--x0", "-1.2,1", *program("once", "ok"))
        starts_once = len([l for l in lines() if l.startswith("start")])
        fresh()
        b = run(opt, "minimize", "--persistent", "--x0", "-1.2,1", *program("loop", "ok"))
        starts_loop = len([l for l in lines() if l.startswith("start")])
        nfev = json.loads(a[1])["nfev"] if a[1] else -1
        say(nfev > 10 and starts_once == nfev and starts_loop == 1, f"one process: {starts_loop} start with --persistent, {starts_once} without, for {nfev} evaluations")

        # 3. the end
        say("eof" in lines(), "the program saw its input close")
        fresh()
        a = run(opt, "minimize", "--persistent", "--x0", "1,1", "--maxfev", "3", *program("loop", "deaf"))
        pids = [int(l.split()[1]) for l in lines() if l.startswith("start")]
        time.sleep(0.3)
        alive = False
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError:
                pass
        say(a[0] in (0, 1) and 4.5 <= a[3] <= 9.5 and not alive, f"a program that ignores the end is killed after five seconds: {a[3]:.1f} s, status {a[0]}, process {'still there' if alive else 'gone'}")

        # 4. the refusals
        def refused(tag, what, *argv, limit=40, under=None):
            r = run(opt, *argv, limit=limit)
            try:
                err = json.loads(r[2])
            except ValueError:
                err = {}
            good = r[0] == 2 and r[1] == "" and err.get("error") == tag and err.get("message") and err.get("repair")
            if good and what.startswith("exit") or good and what.startswith("answers"):
                good = "point" in err
            if under is not None:
                good = good and r[3] < under
            say(good, f"refused as {tag}: {what}" + ("" if good else f" (status {r[0]}, stdout {r[1][:60]!r}, stderr {r[2][:120]!r}, {r[3]:.1f} s)"))

        fresh()
        refused("opt.objective-failed", "exits after 3 requests", "minimize", "--persistent", "--x0", "-1.2,1", *program("loop", "exit-after", "3"))
        refused("opt.objective-output", "answers with the wrong count (BFGS, value only)", "minimize", "--persistent", "--method", "bfgs", "--x0", "-1.2,1", *program("loop", "ok"))
        refused("opt.objective-output", "answers with two lines for one", "minimize", "--persistent", "--x0", "-1.2,1", *program("loop", "two"))
        refused("opt.objective-not-finite", "answers infinity (1e999)", "minimize", "--persistent", "--x0", "-1.2,1", *program("loop", "nan"))
        refused("opt.objective-failed", "does not answer within --timeout-ms 400", "minimize", "--persistent", "--timeout-ms", "400", "--x0", "-1.2,1", *program("loop", "silent"), under=8)
        refused("opt.objective-failed", "is not there", "minimize", "--persistent", "--x0", "1,1", "--", "/no/such/program")
        refused("opt.usage", "--persistent with a built-in objective", "minimize", "--persistent", "--x0", "1,1", "--objective", "sphere")
        refused("opt.usage", "--timeout-ms without --persistent", "minimize", "--timeout-ms", "400", "--x0", "1,1", *program("once", "ok"))
        refused("opt.bad-number", "--timeout-ms 0", "minimize", "--persistent", "--timeout-ms", "0", "--x0", "1,1", *program("loop", "ok"))
        refused("opt.bad-number", "--timeout-ms x", "minimize", "--persistent", "--timeout-ms", "x", "--x0", "1,1", *program("loop", "ok"))
        # no process of the refused runs is left behind
        time.sleep(0.5)
        left = subprocess.run(["pgrep", "-f", prog], capture_output=True, text=True).stdout.split()
        say(not left, f"no program is left running after the refusals ({len(left)} left)")

        # 5. the time
        fresh()
        a = run(opt, "minimize", "--x0", "-1.2,1", *program("once", "ok"))
        b = run(opt, "minimize", "--persistent", "--x0", "-1.2,1", *program("loop", "ok"))
        say(a[0] == 0 and b[0] == 0 and b[3] * 2 < a[3], f"159 evaluations: {b[3]:.2f} s persistent, {a[3]:.2f} s with a process each (machine-dependent; the check is a ratio over 2)")
    finally:
        shutil.rmtree(d)
    return bad == 0


MUTANTS = [
    ("lib/persist/persist.cho", "if nl + 1 < len(all) {", "if nl + 1 < 0 {", "a second line is not noticed"),
    ("lib/persist/persist.cho", "let deadline = clock_ms(clock) + timeout;", "let deadline = clock_ms(clock) + 100000000;", "the timeout is not applied"),
    ("lib/persist/persist.cho", [("    pipe_close(to);\n    var p = poller;", "    var p = poller;"), ("    pipe_close(pipe_in);\n", "    pipe_close(pipe_in);\n    pipe_close(to);\n")], None, "the program's input is closed only after the wait"),
    ("lib/persist/persist.cho", "    if !exited {\n        borrow child as &ch in {\n            child_kill(ch, 256);", "    if false {\n        borrow child as &ch in {\n            child_kill(ch, 256);", "a program that ignores the end is not killed"),
    ("lib/persist/persist.cho", "if nl < 0 && ended {\n                        code = 2;", "if nl < 0 && ended {\n                        code = 0;", "a closed output is not an error"),
    ("lib/nm/objective.cho", "let (c3, v3) = number_of(buffer.bytes(ar)[0..length]);", "let (c3, v3) = number_of(buffer.bytes(ar)[0..length - 1]);", "the last character of the answer is dropped"),
    ("lib/nm/objective.cho", "let wait = l.timeout;", "let wait = 100000000;", "the timeout of the link is not the one asked for"),
    ("tools/opt/main.cho", "timeout = lists.natural(tm_text);\n        if timeout < 1 {", "timeout = lists.natural(tm_text);\n        if timeout < 0 - 5 {", "--timeout-ms 0 is accepted"),
]


def mutants(cancho):
    ok = True
    for path, old, new, what in MUTANTS:
        d = tempfile.mkdtemp()
        try:
            counted = True
            for f in SRC:
                os.makedirs(os.path.dirname(os.path.join(d, f)), exist_ok=True)
                text = open(os.path.join(ROOT, f)).read()
                if f == path:
                    for o, n in (old if isinstance(old, list) else [(old, new)]):
                        if text.count(o) != 1:
                            print(f"FAIL mutant '{what}': the text it changes occurs {text.count(o)} times in {f}")
                            ok = False
                            counted = False
                        text = text.replace(o, n, 1)
                open(os.path.join(d, f), "w").write(text)
            if not counted:
                continue
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
