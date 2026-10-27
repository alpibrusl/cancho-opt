# cancho-opt

Numerical optimisation as an agent-first command line, in [cancho](https://github.com/alpibrusl/cancho): the part of `scipy.optimize`
that [cancho-quantum](https://github.com/alpibrusl/cancho-quantum) needed, built like its siblings
[cancho-ml](https://github.com/alpibrusl/cancho-ml) (`ml`) and [cancho-stats](https://github.com/alpibrusl/cancho-stats) (`stats`): one
binary, JSON on standard output, an error is data with a stable rule tag.

```
opt minimize --x0 -1.2,1 --objective rosenbrock
{"method":"nelder-mead","x":[1.0000220217835696,1.0000422197517715],"fun":8.177661197416674e-10,"nfev":159,"nit":85,"converged":true,"message":"Optimization terminated successfully."}

opt minimize --method bfgs --x0 -1.2,1 --objective rosenbrock
{"method":"bfgs","x":[0.9999999710013876,0.9999999461190724],"fun":2.535309047218261e-15,"jac":[-1.7045156985790e-6,8.232592607626e-7],"nfev":39,"njev":39,"nit":32,"converged":true,"message":"Optimization terminated successfully."}

opt minimize --x0 1,1,1 --lower 0.05,0.05,0.05 --upper 20,20,20 --step 0.5 --maxfev 400 -- ./my-objective ARG...
```

The objective of a real problem is a program: for each point, `opt` writes the numbers on its standard input and reads one number from its
standard output. That is how cancho-quantum's `hw fit` minimises the misfit of a noise model that only the simulator `qsim` can compute.

**Status.** `minimize --method nelder-mead`, which is `scipy.optimize.minimize(method="Nelder-Mead")` step for step: the same iterates, bit for bit
(`scripts/check_scipy.py`, 65 runs, 9 deliberate mutants all killed); and `--method bfgs` (with a gradient: a built-in objective's, or a program that prints the value and then the gradient), scipy's BFGS with its two line
searches (MINPACK's `dcsrch`, then `zoom`), bit for bit against scipy with its BLAS dot products replaced by sequential sums (`scripts/check_bfgs.py`: 565 runs, 13 mutants killed; against real BLAS the last bits differ and about 1 run
in 7 takes one or two more iterations: `docs/design.md` section 9). Levenberg-Marquardt and a finite-difference gradient are not built; `docs/design.md` says what is claimed, how it is measured and what is open.

```
cancho build --bin opt            # or: cancho build lib/nm/objective.cho lib/nm/lists.cho lib/nm/nm.cho lib/bfgs/dcsrch.cho lib/bfgs/bfgs.cho tools/opt/main.cho --std -o build/opt
OPT=build/opt python3 scripts/check_scipy.py [--mutants]    # needs scipy; --mutants needs CANCHO=the compiler
OPT=build/opt python3 scripts/check_bfgs.py [--mutants] [--wide] [--blas]
OPT=build/opt python3 scripts/check_cli.py
```

Authority: the binary starts the objective program (`exec`, and a clock for its timeout) and reads and writes standard input and
output; it reads no file and has no network. Licence EUPL-1.2, copyright Alpibru S.L.
