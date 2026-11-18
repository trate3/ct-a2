# Custom prompts

The prompts we wrote or changed, copied verbatim from the code and data the runs used (`tools/make_prompts.py`). The two SWE-bench agent files are ImpossibleBench's own text, kept here for reference. `SHA256SUMS` lists the hashes.

| File | Used for | Source in this tree |
|---|---|---|
| `eq2_agent/lcb_T1_instruction.txt` | LiveCodeBench agent instruction (first user message). Same text for conflicting and one-off tests. | `evals/reward_hacking/data/a25/sweep2_prompts.json`, entry `T1` |
| `eq2_agent/lcb_system_message.txt` | LiveCodeBench system message (3 submissions) | `scorer.system_message(T1, 3)` |
| `eq2_agent/swe_system_message_template.txt` | SWE-bench system message; `{test_command}` is filled per task. Shown with "up to 10 times" (10 submissions). | Not in this code base: built by ImpossibleBench (`safety-research/impossiblebench` at commit `061dc3dce6a96ab6cf02a855157263033dcfa3ba`, `swebench_agent_full.py`). Rebuilt from that commit and byte-identical to a recorded episode's system message. |
| `eq2_agent/swe_user_message_template.txt` | SWE-bench first user message | same upstream package, `swebench_tasks.py` |
| `monitors/transcript_instruction.txt` | All LLM monitors, transcript view (`{reasoning_note}` empty) | `monitors.INSTRUCTION` |
| `monitors/private_reasoning_note.txt` | Inserted for the view that also shows the agent's private reasoning | `monitors.REASONING_NOTE` |
| `monitors/final_code_lcb_instruction.txt`, `final_code_swe_instruction.txt` | Monitors given only the task and final code | `monitors.FINAL_STATE_INSTRUCTION`, `FINAL_STATE_SWE_INSTRUCTION` |
| `monitors/eq1_per_response_instruction.txt` | EQ1 monitor rows: judging one generated response with the trace up to it | `monitors.EARLY_INSTRUCTION` |
| `monitors/developed_swe_judge_instruction.txt` | The judge instruction developed on labelled SWE-bench training episodes (appendix) | frozen file of that run |
| `monitors/registry.json` | Monitor models, revisions, reasoning setting, sampling, output caps | `monitors.MONITORS` |
| `chat_templates/*.jinja` | Pinned chat templates of the two agents | `evals/reward_hacking/data/` |

Notes:

- The Qwen LiveCodeBench runs did not pass a reasoning effort, so the template default (xhigh) applied. SWE-bench used medium. In the converted SWE-bench trajectories the stored prompt text shows the xhigh rendering although the server received medium; the server's token ids are the true prompt.
- Agent sampling: temperature 1.0, top_p 0.95, top_k 20 (Qwen3.8-27B).
- The `{record}` slot of the monitor prompts is filled by `evals/reward_hacking/trajectory.py: render_view`. It is cut at the labelled submission, so the monitor never sees that submission's test result.
