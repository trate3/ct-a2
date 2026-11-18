"""Shared set-up of the decode scripts (profile pass, length check, correctness trial): model, histories, tap, subsets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evals.reward_hacking import llm
from evals.tap_live import live_client, live_tap

TARGET = "qwen3.8-27b"


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--histories", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--prefill-chunk", type=int, default=0)
    ap.add_argument("--reasoning-effort", default=None)


def fixed_subset(avail: list[dict], k: int) -> list[dict]:
    """k histories at evenly spaced positions of the available list (declared, independent of any measurement)."""
    if len(avail) <= k:
        return list(avail)
    return [avail[round(i * (len(avail) - 1) / (k - 1))] for i in range(k)] if k > 1 else [avail[0]]


def setup(args, inject: dict | None = None, *, time_capture: bool = False):
    inject = inject or {}
    device = torch.device(args.device)
    doc = json.loads(Path(args.histories).read_text())
    avail = [h for h in doc["histories"] if h["status"] == "ok"]
    fmt = inject.get("fmt") or llm.get_format(TARGET, **({"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else {}))
    if "model" in inject:
        model, tok = inject["model"], inject["tokenizer"]
    else:
        model, tok = live_client.load_model(args.model, args.tokenizer, device=args.device, dtype=args.dtype)
    tap = live_tap.LiveProbeTap(args.artifact, device=device, scope="call", time_capture=time_capture, name="A1")
    return device, doc, avail, fmt, model, tok, tap
