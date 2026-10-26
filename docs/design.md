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

## 7. Open

* Levenberg-Marquardt and a residual protocol (section 1).
* The final simplex, and restarts, which scipy does not do either.
* Whether `opt` should link the pinned cancho-tools contract package as `ml` and `stats` do. It does not yet: this first slice has no
  dependency, so it builds with the compiler it names and nothing else; the cost is that its refusals are written by hand in the
  contract's shape and not by its module.
