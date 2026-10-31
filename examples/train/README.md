# Training a model with `opt`

Fitting a model is minimising a loss over its parameters; the objective of `opt` is a program that is given the parameters and prints the loss.
One example for each way that is used. `scripts/check_examples.py` runs every one and fails if what is claimed here stops being true
(`docs/design.md` section 12). Python 3, standard library only.

Two things to know before copying a command:

* **Give the interpreter by absolute path.** `opt` does not search `PATH` for the program (a bare `python3` or `sh` is `opt.objective-failed`), so
  the commands below use `$PY`:
  ```
  PY=$(command -v python3)
  ```
* **At most 32 parameters**, and a process start per evaluation (`docs/design.md` sections 5 and 10). These are small models.

## 1. A convex fit with an exact gradient: `bfgs`

`logistic.py`: logistic regression on eight points no line separates. The program prints the loss, then its gradient, on one line.

```
$ build/opt minimize --method bfgs --x0 0,0,0 -- $PY examples/train/logistic.py
{"method":"bfgs","x":[2.286438728302232,2.41774422323849,2.0496226761310191e-16],"fun":0.23084656539527135,"jac":[2.954758915665878e-7,5.441381900617159e-9,-1.3877787807814457e-17],"nfev":15,"njev":15,"nit":14,"converged":true,"message":"Optimization terminated successfully."}
```

The loss equals an independent Newton solve of the same problem to 1e-9 (0.23084656539450893): a convex problem has one answer, so this is the use
case where `converged` can be believed.

## 2. A small network by backpropagation: `bfgs`

`mlp_xor.py`: a 2-2-1 sigmoid network on XOR, nine parameters, cross-entropy loss. Its backward pass is the one cancho-gpu's
`tests/chx/bwd_sigmoid.chx` computes (`dz = dy * (s - s*s)`).

```
$ build/opt minimize --method bfgs --x0 -1,1,1,-1,0,0,1,1,0 -- $PY examples/train/mlp_xor.py
... "fun":7.738798264405215e-06, ... "converged":true ...
$ build/opt minimize --method bfgs --x0 1,1,-1,-1,0.5,-0.5,1,-1,0 -- $PY examples/train/mlp_xor.py
... "fun":0.47738566121466885, ... "converged":true ...
```

**`converged: true` is not "solved".** From the 8 starts listed in `scripts/check_examples.py`, 1 reaches loss 8e-6 and classifies all four points;
the other 7 stop at a local minimum (losses 0.477 and 0.3466) and BFGS reports success at each, because the gradient there is zero. A network has
local minima that a convex fit does not; judge the loss, not the flag, and try several starts.

## 3. A noisy loss, or no gradient: `spsa`

`noisy_logistic.py`: the same loss estimated on a sample of five of the eight points, drawn from a hash of the point so a run is reproducible. One
number is printed. SPSA needs two evaluations an iteration whatever the dimension (`docs/design.md` section 11).

```
$ build/opt minimize --method spsa --x0 0,0,0 --maxiter 300 -- $PY examples/train/noisy_logistic.py
{"method":"spsa","x":[0.8644191200576172,0.17608273243147496,-0.14011537282043465],"fun":0.591314864244987,"nfev":601,"nit":300,"completed":true,"message":"Ran all iterations; SPSA has no convergence test."}
```

The `fun` printed is one noisy sample. The *true* loss at the answer is 0.3760, down from 0.6931 at the start, with the optimum at 0.2308: SPSA
moved the right way and had not arrived after 300 iterations. That is all the gate claims.

## 4. A loss with corners, so no gradient: `nelder-mead`

`relu_fit.py`: a two-unit ReLU network fitted to `|x|` on nine points, seven parameters, a value only. `max(0, z)` has no derivative at 0 and
cancho-gpu has no ReLU kernel yet (`docs/training.md` there), so this is the case a gradient method cannot take.

```
$ build/opt minimize --x0 0.5,0.1,-0.5,0.1,0.8,0.8,0 --maxfev 4000 -- $PY examples/train/relu_fit.py
{"method":"nelder-mead","x":[0.9443774722838956,6.021776799225106e-7,-2.599609937045664,0.000001675634156459979,1.0588989355395166,0.3846735149232503,-9.041462744933883e-7],"fun":1.4265138609738428e-13,"nfev":2039,"nit":1355,"converged":true,"message":"Optimization terminated successfully."}
```

## 5. Trusting a gradient before using it

A wrong gradient makes BFGS stop early or wander, and says nothing. `gradcheck.py` compares a program's gradient with central differences:

```
$ python3 examples/train/gradcheck.py examples/train/logistic.py 0.3,-0.2,0.1
largest relative disagreement 8.833e-11
$ python3 examples/train/gradcheck.py examples/train/mlp_xor.py 0.5,-0.4,0.3,0.8,0.1,-0.1,0.7,-0.6,0.2
largest relative disagreement 7.089e-11
```

It exits 1 above 1e-6; the gate runs it on a copy of `logistic.py` whose bias gradient is doubled and requires that to fail.

## Not here: an objective computed by a cancho-gpu kernel

`cancho-gpu check --run` takes no input (its tensors are the interpreter's own made-up ones), so a `.chx` kernel cannot be given parameters from a
command line, and no program built on it can be an objective for `opt` yet. cancho-gpu's `docs/training.md` records what that would take.
