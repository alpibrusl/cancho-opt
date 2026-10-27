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

## 7. Open

* Levenberg-Marquardt and a residual protocol (section 1).
* The final simplex, and restarts, which scipy does not do either.
* Whether `opt` should link the pinned cancho-tools contract package as `ml` and `stats` do. It does not yet: this first slice has no
  dependency, so it builds with the compiler it names and nothing else; the cost is that its refusals are written by hand in the
  contract's shape and not by its module.
