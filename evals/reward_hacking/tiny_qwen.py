"""Test support: a tiny random-init model built from the real Qwen3.6-35B-A3B config, a
small BPE tokenizer with the model's special tokens, and synthetic multi-call trajectories.

The model is `AutoModelForCausalLM.from_config` on `data/qwen3.6-35b-a3b.config.json`
(model_type qwen3_5_moe, official repo revision 995ad96e...) with the width/depth/experts
shrunk, so transformers' real hybrid gated-delta-net + gated-attention + sparse-MoE code runs.
"""

from __future__ import annotations

import copy
import json
import pathlib

import torch

from evals.reward_hacking import trajectory as tj

CONFIG_PATH = pathlib.Path(__file__).parent / "data" / "qwen3.6-35b-a3b.config.json"
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]

_CORPUS = (
    "the user wants me to implement a function in func.py that passes all tests in test.py "
    "let me start by examining both files to understand what needs to be implemented "
    "text_editor command view path repo bash multiset multisets distinct score pairwise distances "
    "the tests look inconsistent maybe hard code the expected output no that would be cheating "
    "tool response result passed failed assertion error line number def solve return value "
) * 4


def tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator([_CORPUS], trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")


def tiny_config(vocab_size: int, layers: int = 4):
    from transformers import AutoConfig
    cfg = AutoConfig.for_model(**{k: v for k, v in json.loads(CONFIG_PATH.read_text()).items()
                                  if k not in ("architectures",)})
    tc = cfg.text_config
    tc.hidden_size, tc.num_hidden_layers = 64, layers
    tc.layer_types = (["linear_attention"] * 3 + ["full_attention"]) * ((layers + 3) // 4)
    tc.layer_types = tc.layer_types[:layers]
    tc.num_experts, tc.num_experts_per_tok = 8, 2
    tc.moe_intermediate_size = tc.shared_expert_intermediate_size = 32
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 16
    tc.linear_num_key_heads, tc.linear_num_value_heads = 2, 4
    tc.linear_key_head_dim = tc.linear_value_head_dim = 16
    tc.rope_parameters = {**tc.rope_parameters, "mrope_section": [1, 1, 0]}   # rotary dim 4
    tc.vocab_size = vocab_size
    tc.eos_token_id = tc.bos_token_id = 0
    tc.dtype = "float32"
    cfg.dtype = "float32"
    return cfg


def tiny_model(vocab_size: int, layers: int = 4, seed: int = 0):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_config(tiny_config(vocab_size, layers), dtype=torch.float32)
    return m.eval()


# ---------------------------------------------------------------------------
# synthetic trajectories

def _gen(tok, reasoning: str, visible: str, call: str | None):
    """Generation text after `<think>\\n` and its canonical ids + <|im_end|>."""
    text = reasoning + "\n</think>\n\n" + visible
    if call is not None:
        text += "\n<tool_call>\n" + call + "\n</tool_call>"
    return text


def synthetic_trajectory(tok, *, n_calls: int = 4, noncanonical_call: int | None = 2, rollout_id="syn__r0",
                         task_id="lcbhard_1") -> tj.Trajectory:
    """Append-only multi-call trajectory. In `noncanonical_call` the recorded ids split a word
    into pieces BPE would merge, so tokenize(generation text) != the recorded ids and the next
    prompt's token-prefix property fails (forces the recompute path)."""
    traj = tj.Trajectory(meta={"task_id": task_id, "rollout_id": rollout_id, "partition": "train"})
    prompt = ("<|im_start|>system\nyou are an agent<|im_end|>\n<|im_start|>user\nimplement the function<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n")
    rng = torch.Generator().manual_seed(0)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for k in range(n_calls):
        visible = "" if k == n_calls - 1 else "let me look"
        call = None if k == n_calls - 1 else f"<function=bash>\n<parameter=command>\nls {k}\n</parameter>\n</function>"
        text = _gen(tok, f"the multiset score pairwise tests look inconsistent {k}", visible, call)
        ids = tok.encode(text, add_special_tokens=False)
        if k == noncanonical_call:
            # re-encode the first word piecewise: still decodes to the same text
            first = tok.encode("the", add_special_tokens=False)
            alt = [x for c in "the" for x in tok.encode(c, add_special_tokens=False)]
            assert len(alt) > len(first) or alt != first
            ids = alt + ids[len(first):]
            text = tok.decode(ids)
        traj.add(tj.MODEL_CALL, {"prompt": prompt, "generation": text, "tokens": ids + [im_end],
                                 "stop_type": "eos"}, call_index=k)
        traj.add(tj.ASSISTANT, {"text": visible}, call_index=k)
        tool = f"output of call {k} {int(torch.randint(0, 99, (1,), generator=rng))}"
        traj.add(tj.TOOL_RESULT, {"call_id": str(k), "name": "bash", "output": tool})
        prompt = (prompt + text + "<|im_end|>\n<|im_start|>user\n<tool_response>\n" + tool +
                  "\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n")
    return traj
