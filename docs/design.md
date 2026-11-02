# cancho-opt — design

`opt` is the part of `scipy.optimize` that [cancho-quantum](https://github.com/alpibrusl/cancho-quantum) needed, as a command line in
the style of `ml` (cancho-ml) and `stats` (cancho-stats): one binary, JSON on standard output, an error is data with a stable rule tag.
It exists because `scripts/hw_fit.py` of cancho-quantum fits four scale factors of a noise model by minimising a misfit that only a
simulator can compute, and that was the last analysis step still in Python (it called `scipy.optimize.minimize`).

Every claim below is **measured** (with the command that measures it), a **choice**, or **open**.

## 1. What is built, and what is not

| Verb | Status |
|---|---|
| `opt minimize --method nelder-mead` | built (this document) |
| `opt minimize --method bfgs` | built (section 9) |
| `opt minimize --method spsa` | built (section 11) |
| `opt minimize --persistent` | built (section 14): the program is started once and answers a line for each line |
| `opt minimize --method lm` (Levenberg-Marquardt, least squares) | **open**: needs an objective that returns residuals, a different protocol (section 4) |
| `opt root`, `opt curve-fit`, gradient methods (BFGS, L-BFGS-B) | **open**, not needed by anything yet; they are written when a program needs them |

Nelder-Mead first because it needs no derivative: the objective of `hw_fit.py` is a simulation run in another process (`qsim`), and no
gradient of it exists.

## 2. The method is scipy's, step for step (choice, then measured)

`scipy.optimize.minimize(method="Nelder-Mead")` has no hidden parts: it is the algorithm of Nelder and Mead (1965) with the standard
coefficients (reflection 1, expansion 2, contraction 1/2, shrink 1/2), the initial simplex of section 3, the test
`max |x_i - x_0| <= xatol and max |f_i - f_0| <= fatol`, and bounds handled by clipping every point to the box before it is evaluated.
`opt` does the same operations in the same order on the same doubles. **Claim (measured, `scripts/check_scipy.py`):** on the test
objectives of section 6, `opt` and scipy visit the same number of points and return the same minimiser *bit for bit*, not merely to a tolerance.
The check fails if any of the digits differ. Where it does not hold (a tie among more than 16 simplex vertices, where numpy's sort is not
stable and ours is; section 3), the document says so.

Choices that differ from scipy, each a refusal and not a different answer:

* An objective value that is NaN or infinite is refused (`opt.objective-not-finite`). scipy goes on with it, and what comes out then
  depends on how NaN compares, which is no basis for an answer.
* More than 32 parameters is refused (`opt.too-many-parameters`): the simplex is stored in one flat vector and a larger problem has not
  been measured.
* Only the minimiser, the value, the counts and a reason are reported. Not the final simplex.

## 3. Details that decide the numbers

* **Initial simplex.** x0 is first clipped to the box. scipy's default: vertex *k* is x0 with component *k* multiplied by 1.05, or set to 0.00025
  if it is 0. `--step S` instead puts vertex *k* at x0 + S e_k (what `hw_fit.py` passes as `initial_simplex`; the check builds scipy's from the clipped
  x0 too: with an `initial_simplex` and an x0 outside the box scipy would clip it later and differently, a quirk `opt` does not copy). With bounds, a
  vertex above its upper bound is reflected into the interior (2 hi - v) and the simplex is then clipped, as scipy does.
* **Sort.** After every iteration the vertices are sorted by value. Ours is a stable insertion sort. numpy's `argsort` is not guaranteed
  stable (on this machine, with numpy 2.5.3, a symmetric objective such as the sum of squares, whose vertices tie, came out with coordinates
  permuted relative to ours, the values equal), so when two vertices have exactly the same value scipy's answer depends on the numpy build and the
  CPU and no gate could hold on it. The check's cases have no such ties, and a mutant that only changes how ties are ordered is therefore not
  in its list of mutants (it would survive, and that is not a defect of the check but the point of this paragraph).
* **Budget.** As scipy 1.18: with neither limit given both are 200 N; with only `--maxfev` given, no iteration limit. `nfev` never exceeds
  `--maxfev`: the evaluation that would is not made and the iteration is abandoned where it stands (scipy does the same; in a shrink the
  vertices already moved keep their old values until the next sort, a quirk that is copied because the answers are compared bit for bit). Exit
  status 0 when converged, 1 when a limit was hit (the result is still printed, `"converged": false`), 2 on a refusal.
* **Sums are sequential**, in the order numpy's `add.reduce` uses along an axis of a small array.

## 4. The objective: a program, or one of three built in

The objective of the problem we have is a *program*: `opt minimize --x0 1,1,1 -- ./myobjective arg…` starts `./myobjective arg…` for each
point, writes the point on its standard input as numbers in the shortest form that reads back (`1.2e-3`) separated by spaces and one
newline, and reads its standard output as one number. Exit status other than 0, an output that is not exactly one number, or a number that is
not finite, is a refusal (`opt.objective-failed`, `opt.objective-output`, `opt.objective-not-finite`), with the point it failed at in the error.
The environment of the program is `PATH` and `HOME=/tmp` only. This is the same convention as `qsim` and the commands `hw` spawns.

Four objectives are built in, for tests and for trying the command: `sphere` (the sum of squares), `rosenbrock` (the N-dimensional one of
scipy's `rosen`), `quadratic` (sum of (x_i - i)^2, minimum at 0, 1, 2, …) and `absolute` (sum of (i + 1) |x_i - i|, a corner at the
minimum, where the simplex shrinks).

## 5. Authority and limits (choice)

Spawning a program needs `exec` and a clock; reading standard input and writing standard output are the only other effects. No file is read or
written, no network, no FFI. A program that is not given (`--objective NAME`) still links the spawn code. At most 32 parameters, 10^7
evaluations.

## 6. Gates (written before the code; each can fail)

1. **`scripts/check_scipy.py`**: Nelder-Mead on the four objectives in 1 to 8 dimensions, 21 hand-chosen and 30 seeded random starts, default
   simplex and `--step`, with and without bounds, with budgets that stop it early: same `x`, same `fun`, same `nfev` and `nit` as scipy, bit for
   bit. A mutant (a changed coefficient, a wrong clip) must make it fail: `scripts/check_scipy.py --mutants`.
2. **`scripts/check_cli.py`**: every refusal has its rule tag and exit status 2 and prints nothing on standard output; a program objective
   gives the same answer as the built-in one it computes; a failing objective is reported with its point.
3. In cancho-quantum: `scripts/check_hw.py` compares `hw fit` with `scripts/hw_fit.py` (the second half of this work).
4. **`scripts/check_bfgs.py`** (section 9): BFGS against scipy with sequential dot products, bit for bit, and its mutants.

## 8. The rule tags

| Tag | When |
|---|---|
| `opt.usage` | no or unknown command, a flag without a value, no `--x0`, no objective or two |
| `opt.unknown-flag` | a flag `minimize` does not take |
| `opt.unknown-method` | `--method` other than `nelder-mead` |
| `opt.bad-number` | `--x0`, `--lower`, `--upper`, `--step`, `--xatol`, `--fatol`, `--maxfev` or `--maxiter` is not a number of the right kind |
| `opt.bounds-length` | `--lower` or `--upper` does not have as many numbers as `--x0` |
| `opt.bounds-order` | a lower bound above its upper bound |
| `opt.too-many-parameters` | more than 32 |
| `opt.objective-failed` | the program could not be started, or exited with a status other than 0 (the error names the point) |
| `opt.objective-output` | the program's output is not exactly one number (the point is named) |
| `opt.objective-not-finite` | the objective is NaN or infinite (the point is named) |

Each is one JSON object on standard error, `{"error": "<tag>", "message": ..., "repair": ...}`, with the point added for the last three, exit
status 2, nothing on standard output. `scripts/check_cli.py` has a case for every tag.

## 9. BFGS (`--method bfgs`)

`scipy.optimize.minimize(method="BFGS")` with a gradient, ported from the scipy 1.18 source: the BFGS update of the inverse Hessian (`_minimize_bfgs`), the MINPACK line search `dcsrch` that scipy tries first
(`line_search_wolfe1`), and the second line search it falls back to (`line_search_wolfe2`, with `zoom`, `cubicmin` and `quadmin`), and `ScalarFunction`'s bookkeeping of which evaluations are counted.

**The objective.** A built-in objective gives its gradient (`sphere`, `rosenbrock`, `quadratic`; `absolute` has none and is refused). A program prints *one line*: the value, then the gradient, `n + 1` numbers separated by
white space. It is started once for each distinct point. `nfev` and `njev` are counted as scipy counts them (a value asked for at a point it already evaluated is not counted again, a gradient the same), not as the number of times the
program ran, so that the counts compare with scipy's. Exit status 0 when `gtol` was reached, 1 when `--maxiter` was, or the line search could not go on (scipy's "precision loss"), 2 on a refusal.

**What is claimed, and what is not.** scipy does its matrix products and dot products in BLAS, whose order of summation and use of fused multiply-add we cannot know and do not copy; `opt` sums sequentially (with `std.linalg`).
So against scipy *as it is* the answers agree only to the last bits. The claim a gate can hold is against **scipy with exactly those products replaced by sequential sums** (`np.dot` and the 2-norm, patched for the duration of the check, nothing else of
scipy touched): there `opt` agrees **bit for bit**, with the same minimiser, value, gradient, iterations, evaluation counts and status (section 9.2). That is a claim about the port of the algorithm; the difference from
real scipy is a property of BLAS, measured in section 9.2 and not hidden.

Choices that differ from scipy: only an analytic gradient (`jac=None`, which makes scipy difference the function, is refused: `opt.no-gradient`); no bounds, no `xrtol`, `c1` and `c2` fixed at scipy's defaults (1e-4, 0.9), `gtol` (default 1e-5, `--gtol`) is on the largest component of the gradient
(scipy's default norm); at most 32 parameters.

### 9.1 Why a gradient method at all

The VQE example of cancho-quantum (`examples/vqe`) found that a single Nelder-Mead run in 12 dimensions stops on a point that is not a minimum and reports `converged`, and needed 77,592 evaluations for 16 angles. For a rotation
gate the gradient of the energy is exact from two more evaluations per angle (the parameter-shift rule), which is what a program can give this method.

### 9.2 Measured

`OPT=build/opt python3 scripts/check_bfgs.py --wide --blas`:

* **65 runs** (15 chosen, 40 seeded random, 10 chosen because scipy calls its second line search there) of `sphere`, `rosenbrock` and `quadratic` in 1 to 8 parameters, `gtol` from 1e-5 to 1e-14, with `--maxiter` limits: bit for bit against the oracle.
* **500 seeded random Rosenbrock problems** (2 to 8 parameters, starts in [-6, 6]): bit for bit, 0 differ. 62 of them make scipy fall back to `line_search_wolfe2`, so the second line search is exercised and agrees; in those it accepted a step
  19 times in 1,500 problems, **every time through `_zoom`**, so the branch of `scalar_search_wolfe2` that accepts a step without a zoom is ported as scipy has it but **not exercised by any case** (open).
* **Against scipy as it is (BLAS):** over 120 seeded random Rosenbrock problems 102 have the same counts, all 120 reach the same minimum, and the iteration counts differ by at most 2. The 15% that differ are marginal step-size decisions that the
  last bit of a dot product flips; this is *not* a defect of the port, which is what the sequential oracle shows.
* **Mutants:** 13 deliberate bugs (the two line-search constants swapped, the sign of the update, the `rho s s^T` term, the initial step guess, the iteration-limit status, an uncounted gradient, the curvature test of `zoom`, the sign in `quadmin`,
  the radicand of `cubicmin`, the average in `dcstep`, the bisection and the initial bound of `dcsrch`, the gradients of `quadratic` and `rosenbrock`) are all killed. Two are **not listed**, with the reason in the script: removing
  the `alpha1 < 0` guard of `wolfe1` (unreachable: it needs the value to have risen on the last step) and relaxing the curvature test in the main loop of `wolfe2` (the unexercised branch above).
* **`scripts/check_cli.py`:** every new rule tag (`opt.no-gradient`, `opt.flag-not-for-method`), a program that prints the value and gradient of the sum of squares gives the built-in answer bit for bit, a program that prints only the value or
  a NaN is refused, and `--maxiter` reaches status 1 with the limit exact.

**Corrected in place:** the first version of the mutant list had a mutant of the first line search that no input can kill (the guard above) and a mutant of the second whose branch no case reaches; the first was shown unreachable and removed, the second is recorded
as unexercised. And the first plan for this section was "equal to scipy to a tolerance"; the sequential oracle showed the tolerance was hiding nothing and the stronger claim holds.

## 10. Time against scipy (measured, one machine)

`OPT=build/opt python3 scripts/bench_scipy.py --markdown`. Machine: Intel Xeon 2.10 GHz, 4 cores (neither side uses more than one), x86-64 Linux; scipy 1.18.1, numpy 2.5.3; `opt` built by a *debug build* of the compiler (whether that
changes the code it emits was not checked here). The built-in Rosenbrock objective, starting at (-1.2, 1, -1.2, 1, ...). `opt` is timed as a whole process, start included; scipy's `minimize` alone, **not** counting Python's
start or the import of scipy (which favours scipy by far more than a process start costs); each time the minimum of 7 runs (5 for Nelder-Mead at 5 and 10 parameters, 3 for the pure-Python objective).

| method | parameters | opt | scipy (numpy `rosen`) | scipy (pure-Python objective) | iterations / evaluations: opt | scipy | same work |
|---|---|---|---|---|---|---|---|
| BFGS | 2 | 1.3 ms | 3.7 ms | 2.5 ms | 34 / 41 | 34 / 41 | yes |
| BFGS | 5 | 1.5 ms | 6.2 ms | 4.5 ms | 51 / 62 | 51 / 62 | yes |
| BFGS | 10 | 2.0 ms | 11.2 ms | 8.2 ms | 75 / 85 | 74 / 136 | no |
| BFGS | 20 | 5.3 ms | 19.6 ms | 23.9 ms | 165 / 188 | 165 / 188 | yes |
| BFGS | 30 | 18.2 ms | 24.8 ms | 24.9 ms | 220 / 247 | 222 / 250 | no |
| Nelder-Mead | 2 | 1.3 ms | 4.9 ms | 2.6 ms | 117 / 219 | 117 / 219 | yes |
| Nelder-Mead | 5 | 1.5 ms | 18.4 ms | 12.3 ms | 511 / 828 | 511 / 828 | yes |
| Nelder-Mead | 10 | 4.3 ms | 164.9 ms | 71.5 ms | 4754 / 6604 | 4949 / 6946 | no |

Process start and a one-parameter run of `opt`: 1.2 ms.

**What it shows.** Where both did the same work (the rows marked yes), `opt` took 2.8 to 12 times less time than scipy with numpy's `rosen` (1.7 to 9 times less than scipy with a pure-Python objective). The rows marked *no* are not a speed comparison:
scipy's `rosen` sums with numpy's pairwise sum from 9 terms and its BLAS products order their sums differently, so the iterations differ (section 9.2); at 10 parameters Nelder-Mead's 165 ms against 4.3 ms is mostly 6,946 evaluations against 6,604 of an
objective that costs scipy about 20 microseconds a call and `opt` well under one. The advantage **shrinks with size** for BFGS (about 3.7 times at 20 parameters, 1.4 at 30): the update's matrix products are plain sequential loops
(`std.linalg`) and scipy's are BLAS, so beyond the 32 parameters `opt` accepts scipy would be expected to win; that was not measured.

**What it does not show.** These are built-in objectives, which run inside `opt`. A real objective is a program that `opt` starts for every evaluation (section 4), at the 1.2 ms of a process start (the VQE BFGS run of cancho-quantum:
56 gradients in 0.1 s), while scipy calls a Python function in tens of microseconds. For a cheap objective scipy in-process is far faster; for one that costs milliseconds (a `qsim` run) the start no longer matters. One machine, one
build, one starting point per size; the timings move with the machine and are not a gate (`scripts/bench_scipy.py` prints them, it does not assert them).

## 11. SPSA (`--method spsa`)

Simultaneous-perturbation stochastic approximation (Spall 1992), for objectives that are *noisy* and have no gradient: the sampled VQE of cancho-quantum, where BFGS stops on "precision loss" and Nelder-Mead on its limit
(`examples/vqe/README.md`). Each iteration draws a random direction Δ with entries ±1, evaluates the objective at x + c_k Δ and x − c_k Δ (**two evaluations whatever the number of parameters**: a finite difference
along Δ) and moves x by `a_k` times the gradient estimate `(f₊ − f₋) / (2 c_k) · Δ`. The gains fall as `a_k = a / (k + A)^0.602` and `c_k = c / k^0.101` (k = 1, 2, …), Spall's recommended exponents; `A` (`--stability`,
default `0.1 · maxiter`) keeps the early steps from being too long.

**The method.** `--method spsa --x0 LIST (--objective … | -- PROGRAM…) [--maxiter N] [--a A] [--c C] [--stability S] [--seed K]`. Defaults: 100 iterations, `a` 0.2, `c` 0.1, seed 1: they suit variables and gradients of order one, and
`a` and `c` are the two numbers a problem has to be tuned for (`c` should be at least the noise's scale). The objective is the value-only one of Nelder-Mead (section 4). One more evaluation at the end gives `fun`; `nfev` is
`2 · nit + 1`. **There is no convergence test**: SPSA ends after its iterations, so the result has `"completed": true`, not `"converged"`, and the exit status is 0. Flags of the other methods are refused
(`opt.flag-not-for-method`), as is `--seed 0` (the generator's one fixed point).

**The random numbers** are `opt`'s own: xorshift64* (the generator of cancho-quantum's `qc.measure`), seeded by `--seed`; entry i of Δ is +1 when the i-th draw of the iteration is at least 0.5 and -1 otherwise. The same objective and
seed give the same iterates on every platform.

**What is claimed, and how it is measured (`scripts/check_spsa.py`).** scipy has no SPSA, so the oracle is `qiskit_algorithms.optimizers.SPSA` (0.4.0), an independent implementation of the same algorithm, run with the same gain
sequences and with its source of perturbations replaced by this generator's stream. The iterates agree to a tolerance (not bit for bit: the gains use `pow`, which `std.math` and the C library may round differently in the last place) on
deterministic objectives; on noisy ones the gradient estimate is checked to be unbiased over many seeds. Its mutants are listed in the script. **Not claimed:** that SPSA converges well on any problem (that is the experiment of cancho-quantum's
README), the choice of the defaults beyond "order one", blocking, second-order SPSA, resampling, averaging of the last iterates, calibration of `a` and `c` (Qiskit has them; here they are flags).

**Time against Qiskit's SPSA (measured, one machine).** `OPT=build/opt python3 scripts/bench_spsa.py --markdown`; the machine and versions of section 10 (qiskit-algorithms 0.4.0). The built-in sphere objective, the same gains (a 0.2, c 0.1, A a tenth of the iterations), `opt` timed as a whole process (start included) and
`SPSA.minimize` alone with a Python objective (the import and Python's start are not counted, which favours Qiskit); minimum of 7 runs. The two draw different perturbations, so their iterates differ, but they do the same work: two evaluations per iteration and one at the end (the counts match).

| parameters | iterations | opt | Qiskit SPSA | Qiskit / opt | evaluations: opt | Qiskit |
|---|---|---|---|---|---|---|
| 2 | 100 | 1.4 ms | 7.8 ms | 5.6x | 201 | 201 |
| 2 | 1000 | 1.5 ms | 55.7 ms | 37.5x | 2001 | 2001 |
| 8 | 100 | 1.5 ms | 5.4 ms | 3.7x | 201 | 201 |
| 8 | 1000 | 1.6 ms | 55.6 ms | 35.5x | 2001 | 2001 |
| 32 | 100 | 1.8 ms | 6.2 ms | 3.4x | 201 | 201 |
| 32 | 1000 | 2.2 ms | 63.2 ms | 28.8x | 2001 | 2001 |

`opt` is 3.4 to 5.6 times faster at 100 iterations and 29 to 38 times at 1,000 (a second run of the same script gave 3.7 to 4.8 and 32 to 49: the machine moves the ratio by a third). The reason is plain: at 1,000 iterations `opt` takes 1.5 to 2.2 ms, of which 1.2 ms is the process start, so the 2,001 evaluations cost it about 0.3 to 1 ms; Qiskit spends 55 to 63 ms,
about 55 to 65 microseconds an iteration in Python. **What it does not show** is the same as in section 10: a built-in objective runs inside `opt`; a program objective costs the 1.2 ms of a process start per evaluation, which is what dominates the
sampled VQE (1,601 evaluations in 2.3 s, `cancho-quantum` `docs/benchmark.md`). For an objective that is cheap and in Python, Qiskit's SPSA in-process is the faster way to call it.

## 12. Examples: training a model with `opt` (written before the examples; gates, then measured numbers)

Fitting a model is minimising a loss over its parameters, and the objective of section 4 is a program that is given the parameters and
prints the loss (and, for `--method bfgs`, the gradient). `examples/train/` holds one program for each way that is used, and
`scripts/check_examples.py` runs every one and fails if its claim does not hold, so the README of the examples cannot drift from the code.
They need `python3` and nothing else (the standard library). **Measured while writing them:** `opt` starts the program it is given by that name, without searching `PATH`, so a bare `python3` is `opt.objective-failed` and the commands name the interpreter by absolute path; the environment it gives the program is `PATH` and `HOME=/tmp` (section 4).

| Use case | Method | Program | Claim the gate checks |
|---|---|---|---|
| a convex fit with an exact gradient | `bfgs` | `logistic.py`, 3 parameters | converged, largest gradient component below `gtol`, and the loss equal to an independent Newton solve to 1e-9 (measured: 15 evaluations, 2.95e-7, 7.6e-13 apart) |
| a small network by backpropagation | `bfgs` | `mlp_xor.py`, 9 parameters | at least one of a fixed list of 8 starts classifies the four points, and **the starts that do not are reported, not hidden**: measured, 1 of 8 solves it and 7 of 8 report `converged` at a local minimum (a network has local minima; a convex fit has none) |
| a gradient that cannot be written down, or a loss that is noisy | `spsa` | `noisy_logistic.py`, 3 parameters | the loss falls below a bound that the start is above; **not** that SPSA reaches the optimum (section 11); measured, the true loss goes from 0.693 to 0.376 in 300 iterations, the optimum being 0.231 |
| a loss with a corner, so no gradient | `nelder-mead` | `relu_fit.py`, 7 parameters | a ReLU network fits `|x|` on a grid to a stated tolerance (measured: 1.4e-13 in 2,039 evaluations) |
| trusting a gradient before using it | none (a check) | `gradcheck.py` | every program that prints a gradient agrees with central differences to 1e-6, and a deliberately wrong gradient fails |

**Choices.** The programs are Python because the objective is a program and the repository's own gates are Python; they are examples of
the protocol, not part of `opt`. 32 parameters is the limit of `opt` (section 5), so these are small networks; the cost of a program
objective is a process start per evaluation (section 10), so they are not an answer for a network with millions of parameters.

**Open.** An objective computed by a cancho-gpu kernel: `cancho-gpu check --run` takes no input (its tensors are the interpreter's own
made-up ones), so a kernel cannot be given parameters from a command line; docs/training.md of cancho-gpu records what it would take.

## 13. The landing page and the stories (written before the generator; gates, then measured numbers)

The page follows cancho-www's `docs/landing-pattern.md` (the ids `why`, `stories`, `how`, `watch`, `status`, `results`, `try`, `agent`, `docs` in that
order, a Stories link in the header, the footer) and takes its structure from cancho-ml's page. Every figure and every printed run on it is **written by
`scripts/gen_stories.py` from a real run of `opt`**, not typed: the region `<!--stories-->` of `docs/index.html`, the region `<!--run:watch-->`, and the whole of
`docs/stories.html` are its output.

**The stories** are invented and seeded (eight points, a 2-2-1 network, a ReLU, a noisy sample), say so, and each says what it shows and what it does not:
a convex fit, XOR from eight starts, a noisy loss with SPSA, a loss with corners with Nelder-Mead, and a wrong gradient. One more is **not** run by the generator
(it needs cancho-gpu's binary) and is marked as such with the figures of cancho-gpu's own check.

| Gate | What it requires | Can it fail? |
|---|---|---|
| `python3 scripts/gen_stories.py --check` | the page, the stories page and the run replay equal what a fresh run of `build/opt` prints, rounded as the generator rounds them | yes: edit a number in either page, or change an example |
| `python3 scripts/check_landing.py` | the ids of the pattern in order, a Stories link in the header, the stories region, one `<h1>`, a title and description inside the SEO limits, every in-page link names an id, every relative link is a file, no external request | yes |
| the claims about scipy and Qiskit | **not** rerun by these gates: they are the runs of sections 2, 9 and 11, reported there, made by hand on the maintainer's machine; this container has no scipy | not applicable |

**Choices.** The stories use Python objectives, as `examples/train/`; the numbers are rounded to six significant digits, so a different libm that changes
the last bits does not fail `--check` (an iteration count that changes would, and would be a finding). The stories are checked on one platform, Linux x86-64.

## 14. A persistent objective (`--persistent`)

> **Order, honestly.** `CLAUDE.md` asks for the design before the code. The protocol below was first written in cancho-gpu's `docs/resident.md` section 4
> (the reason for it); the code was then written in this change and this section, and `scripts/check_persistent.py`, were written to the code as it
> stood, and the gate was run against deliberate bugs afterwards (below). The gates are still ones that fail: each of the eight mutants fails one.

Section 4's objective is a program started for each point. That costs a process start per evaluation (section 10), and for a program that holds
state worth keeping (a compiled kernel and data on a GPU, a simulator that has loaded its model) it repeats work whose result never changes.
`opt minimize --persistent [--timeout-ms N] … -- PROGRAM ARG…` starts the program **once**, and for each point writes one line and reads one line.

* **The protocol is section 4's, with a line instead of a process.** The request is the point, numbers in the shortest form that reads back,
  separated by single spaces, one newline. The answer is one line: for Nelder-Mead and SPSA one number; for BFGS the value and then the gradient,
  n + 1 numbers (the same parsing and the same finiteness check as the per-process program). The program's standard input closing is the end.
* **Choices.** Its standard error is discarded (the per-process mode reads and drops it too, so nothing is lost, and a program that writes a
  lot of it cannot block). The environment is `PATH` and `HOME=/tmp` as before. An answer is at most 65,536 bytes. **One answer per request:**
  anything after the first newline is `opt.objective-output`, since a program that answers twice would be one answer behind from then on.
  `--timeout-ms N` (default 3,600,000, the per-process mode's limit) bounds the wait for each answer. At the end `opt` closes the program's
  input, waits **five seconds** for it to exit, and kills it if it has not; its exit status is not looked at, because every answer asked for has been read.
* **No new rule tag.** The program ending, closing its output, being too slow, or not being startable is `opt.objective-failed` (with the point);
  an answer that is not what the method needs, or has a second line, is `opt.objective-output`; an infinite one `opt.objective-not-finite`.
  `--persistent` with a built-in objective, or `--timeout-ms` without `--persistent`, is `opt.usage`; a `--timeout-ms` that is not a positive
  integer is `opt.bad-number`. (`NaN` is not a number to the reader, so it is `opt.objective-output`, as it is without `--persistent`.)
* **Authority** is unchanged: `exec` and a clock, and the pipes of the child it starts. `opt` reads no file.
* **Implementation.** `lib/persist/persist.cho` holds the child, its two pipes and an event poller in one value (`Link`), threaded through the
  three methods as an `Option` that is empty for every other objective; `nm`, `bfgs` and `spsa` are otherwise unchanged. It builds with the
  compiler written in cancho alone (`CANCHO_COMPILER=auto`, no fallback to the Rust one) once two constructs it has no code for are avoided: a
  `match` on an enum whose arm assigns `Some(l)` to an outer variable, and an `Option` assigned over an `Option`.

**Measured** (`OPT=build/opt python3 scripts/check_persistent.py`, cancho `a49ea779` built with `CANCHO_COMPILER=auto`, Apple M4 Max, Python 3.14.5
as the program): 19 checks pass, and the 8 mutants (`--mutants`) are all killed. (1) Nelder-Mead (two cases, one with bounds), BFGS and SPSA print
**byte for byte** what they print with a process for each point. (2) The program is started once for 159 evaluations, against 159 times.
(3) A program sees its input close; one that ignores that is killed after 5.0 s and is gone. (4) Ten refusals, each with its tag, and no program
left running after them. (5) **159 evaluations of Rosenbrock take 0.02 to 0.03 s through a persistent program and 2.7 to 2.9 s with a process each**
(about 100 times; the per-process time is Python's start-up, 17 ms an evaluation, so with a program whose start is cheaper the ratio is smaller). The existing
checks are unchanged: `check_cli.py` 56 ok, `check_examples.py` 10 ok, `check_scipy.py` and `check_bfgs.py` 65 runs each, bit for bit.

**Mutants** (each fails the check): a second line not noticed; the timeout not applied (twice: in the link and in the flag); the program's
input closed only after the wait; a program that ignores the end not killed; a closed output not an error; the last character of the answer
dropped; `--timeout-ms 0` accepted. **A mutant that survived the first set** was the newline read as part of the answer: it is equivalent (the
number reader ignores trailing white space), so it was replaced by the last character dropped. A mutant that did not build (the input
never closed: the pipe is then not consumed, a linear-value error) was replaced by closing it late.

**Open.** A program that writes its answer in several writes with pauses is waited for up to the timeout, not checked for progress. A request
longer than the pipe holds (about 8 KiB on macOS, with 32 parameters a request is under 1 KiB) is a refusal and not a split write. The
`NaN`/`inf` text a Python `print(float('nan'))` writes is `opt.objective-output`, not `opt.objective-not-finite`. Whether `opt` should hold
several persistent programs (for a team of evaluations in parallel) is not asked by anything.

## 7. Open

* Levenberg-Marquardt and a residual protocol (section 1).
* The final simplex, and restarts, which scipy does not do either.
* Whether `opt` should link the pinned cancho-tools contract package as `ml` and `stats` do. It does not yet: this first slice has no
  dependency, so it builds with the compiler it names and nothing else; the cost is that its refusals are written by hand in the
  contract's shape and not by its module.
