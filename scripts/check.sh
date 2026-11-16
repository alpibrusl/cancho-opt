#!/bin/sh
# The gate that CI runs: CANCHO=<the compiler of cancho.toml> sh scripts/check.sh   (needs python3 with numpy and scipy 1.18.1; scipy's numbers are what the gates compare with)
# Run and found green on 2026-11-16: the build; check_cli (no input reaches a panic); check_scipy (Nelder-Mead against scipy, bit for bit) with its mutants (each must be killed);
# check_bfgs (BFGS against scipy with sequential dot products, bit for bit) with its mutants and the wide run; and the gates that came with main: check_adam, check_persistent, check_examples, check_landing.
# What this does NOT show: check_adam prints "skip torch.optim.Adam: torch is not installed" and passes without its torch oracle; check_spsa is not run (needs qiskit-algorithms, not installed or verified);
# the benchmarks are not run; nor is any platform other than Linux x86-64.
set -eu
here=$(cd "$(dirname "$0")/.." && pwd)
: "${CANCHO:=cancho}"
export CANCHO
cd "$here"
"$CANCHO" build --bin opt >/dev/null
export OPT="$here/build/opt"
python3 scripts/check_cli.py
python3 scripts/check_scipy.py --mutants
python3 scripts/check_bfgs.py --mutants
python3 scripts/check_bfgs.py --wide
python3 scripts/check_adam.py
python3 scripts/check_persistent.py
python3 scripts/check_examples.py
python3 scripts/check_landing.py
echo "gate: ok"
