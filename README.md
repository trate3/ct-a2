# Certified Taps: code and data

This is the code and the small derived data behind the paper's EQ1 (the cost of running a tap, a probe on the model's internal activations), EQ2 and EQ3 (white-box probes against black-box monitors on ImpossibleBench). It also has the certification and attestation code. The travel scenario and the output-replacement illustration are not included.

## Setup

You need Python 3.11 or newer. No GPU is needed.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-cpu.txt
bash check.sh
```

The install is a large download because it includes torch. `check.sh` takes a few minutes and ends with `check.sh: ALL CHECKS PASSED`.

`bash reproduce_eq1.sh` and `bash reproduce_eq2.sh` rebuild the paper's tables from the shipped data and print each cell next to the number in the paper. `ALIGNMENT.md` says which file reproduces which claim, and `data/README.md` describes the data files.

## GPU and confidential-computing runs

Re-running EQ1 needs one GPU with 96 GB. Re-running EQ2 and EQ3 needs several large GPUs. These two commands print the full steps and run nothing:

```bash
bash reproduce_eq1.sh gpu-plan
bash reproduce_eq2.sh gpu-plan
```

Certification with real attestation needs a confidential-computing machine. See `tee/README.md`.

## Models

Weights are not included. The Hugging Face ids and pinned revisions are in `data/README.md` (the Gemma 4 E2B revision is in `prompts/monitors/registry.json`), and the software versions are in `requirements-gpu.txt`. The custom prompts are in `prompts/`.

## Limits

- Timings depend on the hardware. The EQ1 table was measured on a non-TEE RTX PRO 6000 workstation, and the monitor timings also on a second machine.
- GPU re-runs match only approximately. New agent runs and monitor replies give AUROCs near ours, not equal to them.
- Attestation time and total certification time with real attestation are pending, so those cells of the EQ1 table are empty. The certification code runs end to end with a local key.
- The NVIDIA GPU attestation extension in `tee/extension/` is planned. It covers the GPU only and is not a full workload attestation.
- Raw agent trajectories, activations and full monitor replies are not included. `data/README.md` says how to regenerate them.
- Account names, host names and paths were removed, so hashes recorded inside some data files differ from a recomputation. `CHANGES.md` lists the edits.

## License

MIT, see `LICENSE`.
