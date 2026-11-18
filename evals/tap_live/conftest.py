"""Shared CPU fixtures: tiny Qwen3.5-style model, tiny tokenizer, format with the tiny stop ids."""
from __future__ import annotations

import pytest
import torch

from evals.reward_hacking import tiny_qwen as tq
from evals.tap_live import tiny_model as tm


@pytest.fixture(scope="session")
def tok():
    return tq.tiny_tokenizer()


@pytest.fixture(scope="session")
def model(tok):
    torch.manual_seed(0)
    return tm.tiny_model(tm.tiny_vocab(tok), layers=4, seed=0)


@pytest.fixture(scope="session")
def fmt(tok):
    return tm.tiny_format(tok)


def n_tap_hooks(model, layer: int) -> int:
    from evals.reward_hacking import activations as act
    return sum("LiveProbeTap" in getattr(h, "__qualname__", "") for h in act.find_decoder_layers(model)[layer - 1]._forward_hooks.values())


@pytest.fixture(autouse=True)
def _no_leaked_tap_hooks(model):
    """A test that attaches a tap must not leave its hook on the shared model for the next test."""
    yield
    from evals.reward_hacking import activations as act
    for blk in act.find_decoder_layers(model):
        for k in [k for k, h in blk._forward_hooks.items() if "LiveProbeTap" in getattr(h, "__qualname__", "")]:
            del blk._forward_hooks[k]
