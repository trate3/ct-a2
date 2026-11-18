"""Test support: a tiny random-init `Qwen3_5ForConditionalGeneration` built from the real Qwen3.8-27B config
(`evals/reward_hacking/data/qwen3.8-27b.config.json`: dense hybrid model, linear-attention and full-attention layers,
vision tower shrunk), the tiny BPE tokenizer of `evals.reward_hacking.tiny_qwen` (it has the Qwen special tokens), a
`qwen3.8-27b` format whose stop ids are the tiny tokenizer's, and random tap artifacts. CPU only; nothing is downloaded."""
from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from evals.reward_hacking import llm
from evals.reward_hacking import tap as tapmod
from evals.reward_hacking import tiny_qwen as tq

CONFIG_PATH = pathlib.Path(tq.CONFIG_PATH).parent / "qwen3.8-27b.config.json"
HIDDEN = 64


def tiny_config(vocab: int, layers: int = 4):
    from transformers import AutoConfig
    d = json.loads(CONFIG_PATH.read_text())
    d.pop("architectures")
    cfg = AutoConfig.for_model(**d)
    tc, vc = cfg.text_config, cfg.vision_config
    tc.hidden_size, tc.intermediate_size, tc.num_hidden_layers = HIDDEN, 96, layers
    tc.layer_types = (["linear_attention"] * 3 + ["full_attention"]) * ((layers + 3) // 4)
    tc.layer_types = tc.layer_types[:layers]
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
    tc.linear_num_key_heads, tc.linear_num_value_heads = 2, 4
    tc.linear_key_head_dim = tc.linear_value_head_dim = 16
    tc.rope_parameters = {**tc.rope_parameters, "mrope_section": [1, 1, 0]}
    tc.vocab_size, tc.eos_token_id, tc.bos_token_id, tc.dtype = vocab, 0, 0, "float32"
    vc.depth, vc.hidden_size, vc.num_heads, vc.intermediate_size = 2, 32, 2, 48
    vc.out_hidden_size, vc.num_position_embeddings = HIDDEN, 16
    cfg.dtype = "float32"
    return cfg


def tiny_model(vocab: int, layers: int = 4, seed: int = 0):
    from transformers import AutoModelForImageTextToText
    torch.manual_seed(seed)
    return AutoModelForImageTextToText.from_config(tiny_config(vocab, layers), dtype=torch.float32).eval()


def tiny_vocab(tok) -> int:
    return max(len(tok), 400)


def tiny_format(tok):
    """The real Qwen3.8 format (chat template, parser) with the tiny tokenizer's stop ids."""
    fmt = llm.get_format("qwen3.8-27b", reasoning_effort="medium")
    fmt.stop_token_ids = (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"))
    return fmt


def random_artifact(layer: int, pooling: str = "token_mean", *, alpha: float = 1.0, d: int = HIDDEN, seed: int = 0,
                    threshold: float = 0.0, ema_init: str = "first", normalise: bool = True):
    rng = np.random.default_rng(seed)
    return tapmod.TapArtifact(
        layer=layer, w=rng.normal(size=d) / np.sqrt(d), b=float(rng.normal()),
        mu=rng.normal(size=d) * 0.1 if normalise else None, sigma=(rng.uniform(0.5, 2.0, size=d) if normalise else None),
        alpha=alpha, threshold=threshold, ema_init=ema_init, pooling=pooling, model_id="tiny", method="random")
