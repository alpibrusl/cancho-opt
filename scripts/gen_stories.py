#!/usr/bin/env python3
"""Writes the stories of cancho-opt from real runs of `opt` (docs/design.md section 13):

  * docs/stories.html, the whole file;
  * the regions of docs/index.html between `<!--stories-->` and `<!--/stories-->`, `<!--run:watch-->` and `<!--/run-->`,
    and the header and footer between `<!-- gen:nav -->` and `<!-- /gen:nav -->`, `<!-- gen:footer -->` and `<!-- /gen:footer -->`.

Nothing a reader sees as a number is typed by hand: each is read from the JSON `opt` printed, or computed from an independent reference
(examples/train/reference.py), and rounded to six significant digits so a different libm cannot fail the check by its last bit.

    OPT=build/opt python3 scripts/gen_stories.py            # rewrite
    OPT=build/opt python3 scripts/gen_stories.py --check    # fail if a page differs from what a fresh run gives

The stories are invented and seeded. One (the kernels as an objective) needs cancho-gpu's binary, is not run here, and is marked as such."""
import argparse
import html
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "examples", "train")
sys.path.insert(0, EX)
import logistic  # noqa: E402
import reference  # noqa: E402

OPT = os.path.abspath(os.environ.get("OPT", os.path.join(ROOT, "build", "opt")))
PY = sys.executable
GH = "https://github.com/alpibrusl/cancho-opt/blob/main/"
GPU = "https://github.com/alpibrusl/cancho-gpu/blob/main/"


def g(x, digits=6):
    return "%.*g" % (digits, x)


def rounded(v):
    if isinstance(v, float):
        return float("%.6g" % v)
    if isinstance(v, list):
        return [rounded(x) for x in v]
    if isinstance(v, dict):
        return {k: rounded(x) for k, x in v.items()}
    return v


def compact(d):
    return json.dumps(rounded(d), separators=(",", ":"))


def opt(*args):
    """Runs opt; returns (the JSON it printed, its exit status). A fit that does not converge exits 1 and still prints its result."""
    r = subprocess.run([OPT, "minimize", *args], capture_output=True, text=True)
    try:
        return json.loads(r.stdout), r.returncode
    except ValueError:
        raise SystemExit(f"opt {args}: {r.stdout}{r.stderr}")


def gradcheck(program, x0):
    r = subprocess.run([PY, os.path.join(EX, "gradcheck.py"), program, x0], capture_output=True, text=True)
    return float(r.stdout.split()[-1]), r.returncode


def collect():
    d = {}
    prog = lambda name: os.path.join(EX, name)
    newton = reference.newton_logistic()
    r, _ = opt("--method", "bfgs", "--x0", "0,0,0", "--", PY, prog("logistic.py"))
    d["convex"] = {"r": r, "newton": newton, "gap": abs(r["fun"] - newton), "maxjac": max(abs(v) for v in r["jac"])}
    rows = []
    for i, start in enumerate(reference.XOR_STARTS, 1):
        r, _ = opt("--method", "bfgs", "--x0", start, "--", PY, prog("mlp_xor.py"))
        rows.append({"n": i, "fun": r["fun"], "nfev": r["nfev"], "converged": r["converged"], "solved": reference.xor_solved(r)})
    d["xor"] = rows
    r, _ = opt("--method", "spsa", "--x0", "0,0,0", "--maxiter", "300", "--", PY, prog("noisy_logistic.py"))
    d["noisy"] = {"r": r, "start": logistic.loss_grad(0, 0, 0)[0], "end": logistic.loss_grad(*r["x"])[0], "best": newton}
    r, _ = opt("--x0", "0.5,0.1,-0.5,0.1,0.8,0.8,0", "--maxfev", "4000", "--", PY, prog("relu_fit.py"))
    d["relu"] = r
    # a wrong gradient: two versions of the logistic program, and the right one
    src = open(prog("logistic.py")).read()
    ridge = "return loss, [g[0] / n + L2 * w1, g[1] / n + L2 * w2, g[2] / n]"
    assert "g[2] += p - y" in src and ridge in src
    gc = []
    with tempfile.TemporaryDirectory() as t:
        versions = [("the program as it is", prog("logistic.py")),
                    ("the bias gradient doubled", os.path.join(t, "doubled.py")),
                    ("the ridge term missing from the gradient", os.path.join(t, "noridge.py"))]
        open(versions[1][1], "w").write(src.replace("g[2] += p - y", "g[2] += 2 * (p - y)"))
        open(versions[2][1], "w").write(src.replace(ridge, "return loss, [g[0] / n, g[1] / n, g[2] / n]"))
        for label, path in versions:
            dis, code = gradcheck(path, "0.3,-0.2,0.1")
            r, status = opt("--method", "bfgs", "--x0", "0,0,0", "--", PY, path)
            gc.append({"label": label, "dis": dis, "gc_exit": code, "converged": r["converged"], "nfev": r["nfev"], "fun": r["fun"],
                       "status": status, "message": r["message"]})
    d["gradient"] = gc
    # the replay: one fit, and one refusal
    fit, _ = opt("--method", "bfgs", "--x0", "0,0,0", "--", PY, prog("logistic.py"))
    ref = subprocess.run([OPT, "minimize", "--x0", "1,1", "--", "nosuch-program"], capture_output=True, text=True)
    d["watch"] = {"fit": fit, "refusal": json.loads(ref.stderr), "refusal_status": ref.returncode}
    return d


# ---------------------------------------------------------------- the stories (id, title, short result, full body)

def stories(d):
    c, x, n, rl, gr = d["convex"], d["xor"], d["noisy"], d["relu"], d["gradient"]
    n_conv, n_solved = sum(r["converged"] for r in x), sum(r["solved"] for r in x)
    cmd = lambda s: f'<pre class="code">{html.escape(s, quote=False)}</pre>'
    out = []

    out.append(("convex", "Eight points no line separates",
        f"BFGS converged in {c['r']['nfev']} evaluations, to within {c['gap']:.1e} of an independent Newton solve.",
        f"""<p>Logistic regression on eight invented points (two features), with a small ridge so the answer is unique and its gradient written out. <code>opt</code> asks the program for the loss and the gradient at a point and runs BFGS.</p>
{cmd('$ opt minimize --method bfgs --x0 0,0,0 -- python3 examples/train/logistic.py' + chr(10) + compact({k: c['r'][k] for k in ('method', 'x', 'fun', 'nfev', 'converged')}))}
<p>It converged in <b>{c['r']['nfev']}</b> evaluations to a loss of <b>{g(c['r']['fun'])}</b>. A Newton solve of the same loss, written separately (<code>examples/train/reference.py</code>), gives {g(c['newton'])}: they differ by {c['gap']:.1e}. The largest gradient component at the answer is {c['maxjac']:.1e}.</p>
<p class="note"><b>Shows:</b> BFGS on a convex problem reaching the answer an independent method reaches, with the evaluations counted. <b>Does not show:</b> that this is the right model, only the minimum of this loss; or any other data.</p>"""))

    rows = "".join(
        f'<tr><td class="num">{r["n"]}</td><td class="num">{g(r["fun"], 4)}</td><td class="num">{r["nfev"]}</td>'
        f'<td>{"yes" if r["converged"] else "no"}</td><td class="{"yes" if r["solved"] else "no"}">{"yes" if r["solved"] else "no"}</td></tr>' for r in x)
    out.append(("xor", "XOR: converged is not solved",
        f"BFGS reported converged from {n_conv} of {len(x)} starts; the network classifies all four points from {n_solved}.",
        f"""<p>A 2-2-1 sigmoid network (nine parameters) on XOR, cross-entropy loss, backpropagation written by hand. Eight fixed starting points, BFGS from each.</p>
{cmd('$ opt minimize --method bfgs --x0 <start> -- python3 examples/train/mlp_xor.py')}
<div class="tablewrap"><table><thead><tr><th class="num">start</th><th class="num">final loss</th><th class="num">evaluations</th><th>converged</th><th>solved XOR</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><code>converged</code> was true from <b>{n_conv}</b> of the {len(x)} starts and the network classified all four points from <b>{n_solved}</b>. The others stopped at a local minimum, where the gradient is zero and the loss is not.</p>
<p class="note"><b>Shows:</b> that <code>converged</code> means the gradient is small, not that the model is right; a network has local minima that a convex fit does not. <b>Does not show:</b> how often random starts succeed (eight fixed ones are not a sample), or anything about larger networks.</p>"""))

    out.append(("noisy", "A loss you can only sample",
        f"After 300 iterations the true loss went from {g(n['start'], 4)} to {g(n['end'], 4)}; the optimum is {g(n['best'], 4)}.",
        f"""<p>The logistic loss estimated on five of the eight points, drawn from a hash of the point (so a run is reproducible, and the value is noisy). No gradient is printed. SPSA needs two evaluations an iteration whatever the dimension.</p>
{cmd('$ opt minimize --method spsa --x0 0,0,0 --maxiter 300 -- python3 examples/train/noisy_logistic.py' + chr(10) + compact({k: n['r'][k] for k in ('method', 'x', 'fun', 'nfev', 'nit', 'completed')}))}
<p>The <code>fun</code> it prints is one noisy sample. The <b>true</b> loss, computed here from the full data, went from <b>{g(n['start'], 4)}</b> at the start to <b>{g(n['end'], 4)}</b> at the answer; the best possible is {g(n['best'], 4)}. It moved the right way and had not arrived after {n['r']['nit']} iterations ({n['r']['nfev']} evaluations).</p>
<p class="note"><b>Shows:</b> SPSA lowering a noisy loss with no gradient. <b>Does not show:</b> that SPSA converges well on any problem (<a href="{GH}docs/design.md">design.md</a> section 11 says what is and is not claimed).</p>"""))

    out.append(("corners", "A loss with corners",
        f"Nelder-Mead reached a loss of {rl['fun']:.1e} in {rl['nfev']} evaluations.",
        f"""<p>A two-unit ReLU network fitted to <code>|x|</code> on nine points. <code>max(0, z)</code> has no derivative at 0, so the program prints a value only, and Nelder-Mead needs nothing more.</p>
{cmd('$ opt minimize --x0 0.5,0.1,-0.5,0.1,0.8,0.8,0 --maxfev 4000 -- python3 examples/train/relu_fit.py' + chr(10) + compact({k: rl[k] for k in ('method', 'fun', 'nfev', 'converged')}))}
<p>Loss <b>{rl['fun']:.2e}</b> after <b>{rl['nfev']}</b> evaluations, converged: {str(rl['converged']).lower()}.</p>
<p class="note"><b>Shows:</b> a derivative-free method on a loss a gradient method cannot take. <b>Does not show:</b> that the simplex finds the global minimum from other starts (one start was run).</p>"""))

    grows = "".join(
        f'<tr><td>{r["label"]}</td><td class="num">{r["dis"]:.1e}</td><td>{"flags it" if r["gc_exit"] else "passes"}</td>'
        f'<td>{"yes" if r["converged"] else "no (" + html.escape(r["message"].lower().rstrip(".")) + ")"}</td><td class="num">{r["nfev"]}</td><td class="num">{g(r["fun"], 5)}</td></tr>' for r in gr)
    best = g(d["convex"]["newton"], 5)
    out.append(("gradient", "Check the gradient before you trust BFGS",
        f"A gradient missing its ridge term made BFGS stop with precision loss at {g(gr[2]['fun'], 4)}, above the optimum {g(d['convex']['newton'], 4)}; a doubled bias gradient still converged.",
        f"""<p><code>examples/train/gradcheck.py</code> compares a program's gradient with central differences at a point. Three versions of the logistic program, checked at (0.3, -0.2, 0.1) and then given to BFGS from the origin; the best possible loss is {best}.</p>
<div class="tablewrap"><table><thead><tr><th>version</th><th class="num">largest relative disagreement</th><th>gradcheck</th><th>BFGS converged</th><th class="num">evaluations</th><th class="num">loss reached</th></tr></thead><tbody>{grows}</tbody></table></div>
<p>The check flags both wrong versions. BFGS shows only one of them: with the bias gradient doubled, the gradient still vanishes where the true one does (the true bias gradient is zero at the optimum), so BFGS converged to the right answer anyway. With the ridge term missing, the loss and its gradient disagree, the line search fails, and BFGS stops with a loss above the optimum (exit status 1).</p>
<p class="note"><b>Shows:</b> that a wrong gradient can look fine or fail depending on where it is wrong, and that the check finds both. <b>Does not show:</b> that a gradient which passes at one point is right everywhere.</p>"""))

    out.append(("kernels", "A model whose loss is computed by GPU kernels",
        "12 evaluations, within 1.1e-5 of the normal-equations solution, 28 to 36 ms an evaluation, on cancho-gpu's CPU interpreter.",
        f"""<p>A least-squares fit whose loss and gradient are computed by <a href="{GPU}docs/run.md">cancho-gpu</a> kernels (<code>cancho-gpu run</code>), driven by <code>opt minimize --method bfgs</code>: the objective is a program that calls those kernels twice and prints the loss and the gradient.</p>
{cmd('$ opt minimize --method bfgs --x0 0,0,0 -- python3 examples/train/lsq_objective.py <cancho-gpu> <checkout>')}
<p>It converged in 12 evaluations to within 1.1e-5 of the normal-equations solution, at 28 to 36 ms an evaluation over three runs. <b>This story is not run by this page's generator</b>: it needs cancho-gpu's binary. The numbers are those of cancho-gpu's own check (<code>scripts/check_train.py</code>, <a href="{GPU}docs/run.md">run.md</a> section 6).</p>
<p class="note"><b>Shows:</b> a kernel as an optimiser's objective. <b>Does not show:</b> anything on a GPU (the device programs do not take input yet), more than 32 parameters (the limit of <code>opt</code>), or a speed claim.</p>""", True))
    return out


# ---------------------------------------------------------------- the pieces of the pages

def nav(page):
    h = "" if page == "index" else "index.html"
    cur = ' aria-current="page"' if page == "stories" else ""
    return (f'<header class="top"><div class="wrap">\n  <a class="brand" href="index.html">cancho-opt</a>\n  <nav aria-label="Site">\n'
            f'    <a href="{h}#how" class="opt">How it works</a><a href="stories.html"{cur}>Stories</a><a href="{h}#results" class="opt">Evidence</a>'
            f'<a href="{h}#try">Try it</a><a href="{h}#docs" class="opt">Docs</a><a href="https://github.com/alpibrusl/cancho-opt">GitHub</a>\n  </nav>\n</div></header>')


def footer():
    return ('<footer><div class="wrap">\n  <span>cancho-opt &middot; EUPL-1.2 &middot; written in <a href="https://github.com/alpibrusl/cancho">cancho</a></span>\n'
            '  <a href="https://github.com/alpibrusl/cancho-opt">GitHub</a><a href="llms.txt">llms.txt</a>\n'
            '  <span>the sibling tools: <a href="https://github.com/alpibrusl/cancho-stats">stats</a>, <a href="https://github.com/alpibrusl/cancho-table">table</a>, '
            '<a href="https://github.com/alpibrusl/cancho-ml">ml</a>, <a href="https://github.com/alpibrusl/cancho-gpu">gpu</a></span>\n</div></footer>')


def home_stories(items):
    cards = []
    for it in items:
        sid, title, short = it[0], it[1], it[2]
        tag = ' <span class="tag">not run here</span>' if len(it) > 4 else ""
        cards.append(f'    <article class="story"><h3><a href="stories.html#{sid}">{html.escape(title)}</a>{tag}</h3><p>{html.escape(short)}</p></article>')
    return "\n  <div class=\"stories\">\n" + "\n".join(cards) + "\n  </div>\n"


def watch(d):
    w = d["watch"]
    fit = compact({k: w["fit"][k] for k in ("method", "x", "fun", "jac", "nfev", "njev", "nit", "converged", "message")})
    ref = compact(w["refusal"])
    lines = ["$ opt minimize --method bfgs --x0 0,0,0 -- python3 examples/train/logistic.py", fit, "",
             "$ opt minimize --x0 1,1 -- nosuch-program", ref, f"# exit status {w['refusal_status']}"]
    return '\n<pre class="screen" aria-label="A recorded run of opt: a fit, then a refusal with its repair">' + html.escape("\n".join(lines), quote=False) + "</pre>\n"


def stories_page(items):
    body = []
    for it in items:
        sid, title, _short, full = it[0], it[1], it[2], it[3]
        tag = ' <span class="tag">not run by this page\'s generator</span>' if len(it) > 4 else ""
        body.append(f'  <article class="story" id="{sid}"><h3>{html.escape(title)}{tag}</h3>\n{full}\n  </article>')
    return f"""<!doctype html>
<html lang="en">
<!-- Written by scripts/gen_stories.py from real runs of opt; do not edit by hand (docs/design.md section 13). -->
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cancho-opt stories: what opt did on small problems</title>
<meta name="description" content="Small problems fitted with opt, each a real run: a convex fit, XOR from eight starts, a noisy loss, a loss with corners, a wrong gradient.">
<meta name="color-scheme" content="light dark">
<link rel="canonical" href="https://alpibrusl.github.io/cancho-opt/stories.html">
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
{nav("stories")}
<main id="main">
<section id="stories" style="border-top:0">
  <div class="center"><h1>What opt did on small problems.</h1>
  <p class="sub">Each story is a real run of <code>opt</code>, written into this page by <code>scripts/gen_stories.py</code>, with what it shows and what it does not. The data are invented and seeded; none of it measures the world. Numbers are rounded to six significant digits; the runs were made on Linux x86-64.</p></div>
  <div class="stories">
{chr(10).join(body)}
  </div>
</section>
</main>
{footer()}
</body>
</html>
"""


def replace_region(text, open_marker, close_marker, new):
    a = text.index(open_marker) + len(open_marker)
    b = text.index(close_marker, a)
    return text[:a] + new + text[b:]


def render(d):
    items = stories(d)
    idx = open(os.path.join(ROOT, "docs", "index.html"), encoding="utf-8").read()
    idx = replace_region(idx, "<!--stories-->", "<!--/stories-->", home_stories(items))
    idx = replace_region(idx, "<!--run:watch-->", "<!--/run-->", watch(d))
    idx = replace_region(idx, "<!-- gen:nav -->", "<!-- /gen:nav -->", "\n" + nav("index") + "\n")
    idx = replace_region(idx, "<!-- gen:footer -->", "<!-- /gen:footer -->", "\n" + footer() + "\n")
    return {"docs/index.html": idx, "docs/stories.html": stories_page(items)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(OPT):
        raise SystemExit("set OPT to a built `opt` (cancho build --bin opt)")
    pages = render(collect())
    bad = 0
    for rel, text in pages.items():
        path = os.path.join(ROOT, rel)
        have = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if a.check:
            if have != text:
                print(f"gen_stories: {rel} differs from what a fresh run of opt gives", file=sys.stderr)
                bad += 1
        else:
            open(path, "w", encoding="utf-8").write(text)
            print("wrote", rel)
    if a.check and not bad:
        print("gen_stories: ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
