| Monitor | Monitor hardware (arrangement) | Config | With tap (s per call) | Tap = monitor time (s per call) |
|---|---|---|---:|---:|
| gemma4-e4b | RTX PRO 6000 (agent and monitor on RTX PRO 6000, run one after the other) | reasoning off, cap 1,024 | 237.60 [54.86, 494.45] | 1.50 [1.21, 1.73] |
| gemma4-e4b-think | RTX PRO 6000 (agent and monitor on RTX PRO 6000, run one after the other) | reasoning on, cap 65,536 (context-limited) | 243.63 [60.73, 500.50] | 7.54 [6.67, 8.35] |
| qwen3.8-27b-4k | RTX PRO 6000 (agent and monitor on RTX PRO 6000, run one after the other) | reasoning off, cap 4,096 | 241.89 [59.57, 498.11] | 5.79 [4.94, 6.49] |
| qwen3.8-27b-think | RTX PRO 6000 (agent and monitor on RTX PRO 6000, run one after the other) | reasoning on, cap 8,192 | 253.47 [71.24, 509.88] | 17.38 [16.30, 18.46] |
| glm5.3-flash-nvfp4 | not measured on this hardware | reasoning off, cap 4,096 | | |
| glm5.3-flash-nvfp4-think | not measured on this hardware | reasoning on, cap 16,384 | | |

One saved A0 output per distinct prompt (24 representatives of the 30 positions, each position weighted 1/30); the monitor reads the whole recorded trace through that response (transcript view; private reasoning only in a view that explicitly provides it) and its time, prefill and generation, is measured one request at a time under its registered settings. With tap is the weighted No-tap generation time of the saved calls plus the monitor's time on them, added as two separate measurements; Tap is the monitor time. Model and tokenizer loading and any transfer of the saved output are not included. Where the agent and the monitor ran on different devices the two were run one after the other and the sum is not a same-device or end-to-end wall-clock measurement.

Not measured: glm5.3-flash-nvfp4: not measured on this hardware: the GLM-5.3-Flash NVFP4 checkpoint is about 204 GB (figure from deploy/user/eq2_throughput.sh) and does not fit one 96 GB RTX PRO 6000; its cost on the saved calls is measured on the second machine and composed with the RTX agent time as two devices run one after the other

Not measured: glm5.3-flash-nvfp4-think: not measured on this hardware: the GLM-5.3-Flash NVFP4 checkpoint is about 204 GB (figure from deploy/user/eq2_throughput.sh) and does not fit one 96 GB RTX PRO 6000; its cost on the saved calls is measured on the second machine and composed with the RTX agent time as two devices run one after the other
