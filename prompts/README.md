# Custom prompts

The prompts we wrote or changed, copied verbatim from the code and data the runs used (`tools/make_prompts.py`). The two SWE-bench agent files are ImpossibleBench's own text.

| File | Used for | Source in this tree |
|---|---|---|
| `eq2_agent/lcb_T1_instruction.txt` | LiveCodeBench agent instruction (first user message), conflicting and one-off tests | `evals/reward_hacking/data/lcb_instruction_T1.json` |
| `eq2_agent/lcb_system_message.txt` | LiveCodeBench system message (3 submissions) | `scorer.system_message(T1, 3)` |
| `eq2_agent/swe_system_message_template.txt` | SWE-bench system message; `{test_command}` is filled per task (10 submissions) | ImpossibleBench (`safety-research/impossiblebench` at commit `061dc3dce6a96ab6cf02a855157263033dcfa3ba`, `swebench_agent_full.py`) |
| `eq2_agent/swe_user_message_template.txt` | SWE-bench first user message | same package, `swebench_tasks.py` |
| `monitors/transcript_instruction.txt` | All LLM monitors, transcript view (`{reasoning_note}` empty) | `monitors.INSTRUCTION` |
| `monitors/private_reasoning_note.txt` | Added for the view that also shows the agent's private reasoning | `monitors.REASONING_NOTE` |
| `monitors/final_code_lcb_instruction.txt`, `final_code_swe_instruction.txt` | Monitors given only the task and the final code | `monitors.FINAL_STATE_INSTRUCTION`, `FINAL_STATE_SWE_INSTRUCTION` |
| `monitors/eq1_per_response_instruction.txt` | EQ1 monitor rows: judging one generated response with the trace up to it | `monitors.EARLY_INSTRUCTION` |
| `monitors/registry.json` | Monitor models, revisions, reasoning setting, sampling, output caps | `monitors.MONITORS` |
| `chat_templates/qwen3.8-27b_chat_template.jinja` | Chat template of the agent | `evals/reward_hacking/data/` |

Agent settings: temperature 1.0, top_p 0.95, top_k 20 (Qwen3.8-27B). LiveCodeBench runs used the template's default reasoning effort (xhigh), SWE-bench runs used medium. The `{record}` slot of the monitor prompts is filled by `evals/reward_hacking/trajectory.py: render_view`, cut at the labelled submission.
