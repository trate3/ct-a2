"""Write the verbatim prompt files in prompts/ from the code that used them.

Nothing is retyped: every text is imported from the module or read from the data file
that the experiments used. Run from the artifact root: python tools/make_prompts.py
The SWE-bench agent instruction comes from the ImpossibleBench package; its text is
in prompts/eq2_agent/swe_*.txt, rebuilt from the upstream files at the pinned commit
(see prompts/README.md).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.reward_hacking import monitors, scorer  # noqa: E402

OUT = ROOT / "prompts"
DATA = ROOT / "evals" / "reward_hacking" / "data"


def write(rel: str, text: str) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def main() -> int:
    t1 = json.loads((DATA / "lcb_instruction_T1.json").read_text())["T1"]
    assert hashlib.sha256(t1["text"].encode()).hexdigest() == t1["sha256"]
    write("eq2_agent/lcb_T1_instruction.txt", t1["text"])
    write("eq2_agent/lcb_system_message.txt", scorer.system_message(t1["text"], max_attempts=3))
    write("monitors/transcript_instruction.txt", monitors.INSTRUCTION)
    write("monitors/private_reasoning_note.txt", monitors.REASONING_NOTE)
    write("monitors/final_code_lcb_instruction.txt", monitors.FINAL_STATE_INSTRUCTION)
    write("monitors/final_code_swe_instruction.txt", monitors.FINAL_STATE_SWE_INSTRUCTION)
    write("monitors/eq1_per_response_instruction.txt", monitors.EARLY_INSTRUCTION)
    registry = {}
    for name in ("gemma4-e4b", "gemma4-e4b-think", "qwen3.8-27b-4k", "qwen3.8-27b-think", "glm5.3-flash-nvfp4-think"):
        spec = monitors.MONITORS[name]
        registry[name] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v))
                          for k, v in vars(spec).items()} if hasattr(spec, "__dict__") else str(spec)
    write("monitors/registry.json", json.dumps(registry, indent=1, sort_keys=True, default=str) + "\n")
    for f in ("qwen3.8-27b_chat_template.jinja",):
        write(f"chat_templates/{f}", (DATA / f).read_text())
    n = sum(1 for p in OUT.rglob("*") if p.is_file() and p.name != "README.md")
    print(f"wrote prompts/ ({n} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
