"""Write the verbatim prompt files in prompts/ from the code that used them, plus SHA256SUMS.

Nothing is retyped: every text is imported from the module or read from the data file
that the experiments used. Run from the artifact root: python tools/make_prompts.py
The SWE-bench agent instruction is not in this code base (it comes from the
ImpossibleBench package); its text is copied from prompts/eq2_agent/swe_*.txt, which
were rebuilt from the upstream files at the pinned commit (see prompts/README.md).
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
    t1 = json.loads((DATA / "a25" / "sweep2_prompts.json").read_text())["T1"]
    assert hashlib.sha256(t1["text"].encode()).hexdigest() == t1["sha256"]
    write("eq2_agent/lcb_T1_instruction.txt", t1["text"])
    write("eq2_agent/lcb_system_message.txt", scorer.system_message(t1["text"], max_attempts=3))
    write("monitors/transcript_instruction.txt", monitors.INSTRUCTION)
    write("monitors/private_reasoning_note.txt", monitors.REASONING_NOTE)
    write("monitors/final_code_lcb_instruction.txt", monitors.FINAL_STATE_INSTRUCTION)
    write("monitors/final_code_swe_instruction.txt", monitors.FINAL_STATE_SWE_INSTRUCTION)
    write("monitors/eq1_per_response_instruction.txt", monitors.EARLY_INSTRUCTION)
    registry = {}
    for name in ("gemma4-e2b", "gemma4-e2b-think", "gemma4-e4b", "gemma4-e4b-think", "qwen3.8-27b-4k",
                 "qwen3.8-27b-think", "glm5.3-flash-nvfp4", "glm5.3-flash-nvfp4-think"):
        spec = monitors.MONITORS[name]
        registry[name] = {k: (v if isinstance(v, (str, int, float, bool, type(None), list, dict)) else str(v))
                          for k, v in vars(spec).items()} if hasattr(spec, "__dict__") else str(spec)
    write("monitors/registry.json", json.dumps(registry, indent=1, sort_keys=True, default=str) + "\n")
    for f in ("qwen3.8-27b_chat_template.jinja", "muse-glimmer-30b_chat_template.jinja"):
        write(f"chat_templates/{f}", (DATA / f).read_text())
    sums = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in ("SHA256SUMS", "README.md"):
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {len(sums)} prompt files and prompts/SHA256SUMS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
