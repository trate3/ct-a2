"""Run an evaluation or local service with ``python -m evals_runner``."""

from __future__ import annotations

import argparse
import importlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m evals_runner")
    commands = parser.add_subparsers(dest="command", required=True)
    chosen = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else None
    for name, module_name, help_text in (
        ("protocol-integrity-exp", "protocol_integrity_exp",
         "compare honest execution with environment-message substitution"),
        ("rh-rollouts", "rh_rollouts",
         "Reward hacking: complete agent rollouts on Impossible-LiveCodeBench (resumable)"),
        ("rh-monitor", "rh_monitor",
         "Reward hacking: matched black-box and CoT monitor scores for finished rollouts"),
        ("rh-eq2-report", "rh_eq2_report",
         "Reward hacking EQ2: black-box vs white-box taps, paired contrasts, early detection"),
        ("rh-eq2-cv-report", "rh_eq2_cv_report",
         "Reward hacking EQ2: split monitors vs train-cv probes, non-inferiority, cost"),
        ("rh-t1-eq2-report", "rh_t1_eq2_report",
         "T1 EQ2: status-filtered terminal and early monitor metrics with a test freeze guard"),
        ("rh-t1-invalid-sensitivity", "rh_t1_invalid_sensitivity",
         "T1 EQ2 supplemental: invalid-as-maximum AUROC sensitivity after the initial aggregate"),
        ("rh-t1-finalstate-compare", "rh_t1_finalstate_compare",
         "T1 EQ2 exploratory: paired final-state vs history-only monitor aggregate"),
        ("rh-probe", "rh_probe",
         "Reward hacking: fit and freeze the probe taps on training data; score rollouts (CPU)"),
    ):
        # Import only the selected command's module (each module may need optional data or
        # submodules); `--help` imports all and notes any that cannot load.
        if chosen is not None and name != chosen:
            continue
        command = commands.add_parser(name, help=help_text)
        try:
            module = importlib.import_module(f".{module_name}", __package__)
        except ImportError as exc:
            if chosen is not None:
                raise
            command.description = f"(unavailable here: {exc})"
            continue
        module.configure_parser(command)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
