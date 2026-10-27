# CLAUDE.md -- cancho-opt

Design before code: each verb gets a section in `docs/design.md` before its implementation, and every claim in it is either measured (with the command that
measures it) or marked as a choice or open.

- **Agent-first, like cancho-ml and cancho-stats.** JSON on standard output; an error is data with a stable `opt.<name>` rule tag, exit status 2 and a repair.
  No payload is ever logged.
- **scipy is the oracle**, and the claim is bit for bit: Nelder-Mead against scipy as it is (`scripts/check_scipy.py`), BFGS against scipy with its BLAS dot products replaced by sequential sums (`scripts/check_bfgs.py`;
  against real BLAS only the last bits can agree, `docs/design.md` section 9). Both checks must pass, and every mutant they list must be killed (`--mutants`), or be explained where it is left out.
- **No source file over 2,000 lines. Every refusal has a rule tag. No input reaches a panic** (`scripts/check_cli.py`).
- **Gates are written before the code they judge and must be able to fail.**
- **Measured claims only.** A claim found false is corrected in place, in the document that made it.
- Licence EUPL-1.2, copyright Alpibru S.L. Commit trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; PRs stay draft until the maintainer says merge; no
  force-push to main.
