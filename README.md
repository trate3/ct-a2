# Certified Taps: code and data

This repository holds the code and the derived data for the paper's experiments: the cost of running a tap during generation (EQ1), white-box probes against black-box monitors on ImpossibleBench (EQ2 and EQ3), and the certification and attestation protocol.

## Setup

Python 3.11 or newer.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt
```

## Checks and tables

```bash
bash check.sh             # unit tests, the protocol demo and both table scripts; ends with "check.sh: ALL CHECKS PASSED"
bash reproduce_eq1.sh     # EQ1 table from data/eq1
bash reproduce_eq2.sh     # EQ2 and EQ3 tables from data/eq2
bash reproduce_eq1.sh gpu-plan
bash reproduce_eq2.sh gpu-plan
```

The table scripts print each cell next to the number in the paper. The two `gpu-plan` commands print the full GPU runs (models, revisions and commands) without running anything. GPU software versions are in `requirements-gpu.txt`.

The protocol demo runs on its own with `python -m evals_runner protocol-integrity-exp e2e --n 8` (add `--tamper-environment` to see the certifier abort).

## Prompts

The custom prompts and the monitor registry are in `prompts/` (see `prompts/README.md`).

## License

MIT, see `LICENSE`.
