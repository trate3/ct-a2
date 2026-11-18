# Main complete-call table inputs (descriptive)

30 of 30 positions have a valid paired realization (complete); 24 distinct prompts, 10 tasks; sources: {'stage1_r0': 0, 'stage2': 30}. prompt-representative design: each position reuses the realization of its prompt group's representative (first in manifest order); positions keep their preselected weights; uncertainty resamples tasks, so reuse within a task is counted together and the 30 positions are not 30 independent prompts. Each position uses its group's representative pair: stage-1 repeat 0 if both calls ended naturally, otherwise the stage-2 natural-completion pair with the same representative seed; whole call time counted; no faster or shorter realization is ever chosen. 6 positions show `representative_seed_reuse`: they reuse their group's representative seed and pair; that is never a mismatch between A0 and A1 or between the timed generation and the output a monitor judges. 0 prompt groups span several tasks.

Counts (valid positions): natural stops A0 30, A1 30; cap hits A0 0, A1 0; context-ceiling hits A0 0, A1 0; identical A0/A1 token streams 30. Weighted mean generated tokens 5626 (range 222 to 23266).

## Primary (weighted over positions; task-clustered bootstrap)

- No tap, mean seconds per call: 236.09  [53.17, 493.20]
- With probe, mean seconds per call: 236.09  [53.17, 493.18]
- Paired A1 - A0, milliseconds per call: -1.41  [-64.77, +58.27]

The paired-difference interval includes zero (unresolved). Intervals: 2000 draws, seed 0, unit task (a task's positions kept together).

## Supporting stage-1 pairs (not the primary)

SUPPORTING, not the primary: the curtailed stage-1 repeat-0 pairs (cap 4,096 tokens) of the representatives; elapsed times are not expected to be equal to the main stage's. 16 representatives have a stage-1 repeat-0 call; 15 pairs ended naturally. Primary positions taken from stage 1: 0.

## Sensitivity (not the primary): sensitivity_stage1_repeat1

SENSITIVITY, not the primary: stage-1 repeat index 1 of each position's representative pair, bounded at the stage-1 cap (None new tokens); capped calls are incomplete realizations.

- No tap 50.89 s; with probe 50.88 s; paired difference -9.46 ms [-16.70, -2.96]; natural stops 17, cap hits 1, mean tokens 1215.

## Sensitivity (not the primary): sensitivity_own_seed_repeat0

SENSITIVITY, not the primary: every position's OWN stage-1 repeat-0 pair (its own seed), bounded at the stage-1 cap (None new tokens); capped calls are incomplete realizations.

- No tap 53.94 s; with probe 53.93 s; paired difference -13.59 ms [-19.19, -8.09]; natural stops 16, cap hits 1, mean tokens 1297.

## Positions

| position | representative | seed reuse | source | A0 s | A1 s | tokens (A0) | finish A0 / A1 | weight |
|---|---|---|---|---|---|---|---|---|
| lcbhard_101@0.25 | lcbhard_101@0.25 | no | stage2 | 27.07 | 27.07 | 681 | natural / natural | 0.0333 |
| lcbhard_101@0.5 | lcbhard_101@0.5 | no | stage2 | 36.10 | 36.09 | 879 | natural / natural | 0.0333 |
| lcbhard_101@0.75 | lcbhard_101@0.75 | no | stage2 | 298.87 | 298.91 | 6896 | natural / natural | 0.0333 |
| lcbhard_18@0.25 | lcbhard_18@0.25 | no | stage2 | 38.92 | 38.92 | 971 | natural / natural | 0.0333 |
| lcbhard_18@0.5 | lcbhard_18@0.5 | no | stage2 | 42.74 | 42.74 | 1049 | natural / natural | 0.0333 |
| lcbhard_18@0.75 | lcbhard_18@0.75 | no | stage2 | 69.34 | 69.34 | 1635 | natural / natural | 0.0333 |
| lcbhard_22@0.25 | lcbhard_22@0.25 | no | stage2 | 155.66 | 155.63 | 3790 | natural / natural | 0.0333 |
| lcbhard_22@0.5 | lcbhard_22@0.5 | no | stage2 | 21.25 | 21.25 | 491 | natural / natural | 0.0333 |
| lcbhard_22@0.75 | lcbhard_22@0.5 | yes | stage2 | 21.25 | 21.25 | 491 | natural / natural | 0.0333 |
| lcbhard_45@0.25 | lcbhard_45@0.25 | no | stage2 | 17.20 | 17.20 | 435 | natural / natural | 0.0333 |
| lcbhard_45@0.5 | lcbhard_45@0.5 | no | stage2 | 110.50 | 110.53 | 2726 | natural / natural | 0.0333 |
| lcbhard_45@0.75 | lcbhard_45@0.75 | no | stage2 | 47.44 | 47.43 | 1133 | natural / natural | 0.0333 |
| lcbhard_55@0.25 | lcbhard_55@0.25 | no | stage2 | 9.91 | 9.92 | 252 | natural / natural | 0.0333 |
| lcbhard_55@0.5 | lcbhard_55@0.5 | no | stage2 | 20.79 | 20.79 | 523 | natural / natural | 0.0333 |
| lcbhard_55@0.75 | lcbhard_55@0.75 | no | stage2 | 8.97 | 8.97 | 222 | natural / natural | 0.0333 |
| lcbhard_64@0.25 | lcbhard_64@0.25 | no | stage2 | 36.76 | 36.75 | 909 | natural / natural | 0.0333 |
| lcbhard_64@0.5 | lcbhard_64@0.5 | no | stage2 | 75.97 | 75.97 | 1756 | natural / natural | 0.0333 |
| lcbhard_64@0.75 | lcbhard_64@0.5 | yes | stage2 | 75.97 | 75.97 | 1756 | natural / natural | 0.0333 |
| lcbhard_71@0.25 | lcbhard_71@0.25 | no | stage2 | 1035.30 | 1035.50 | 23266 | natural / natural | 0.0333 |
| lcbhard_71@0.5 | lcbhard_71@0.25 | yes | stage2 | 1035.30 | 1035.50 | 23266 | natural / natural | 0.0333 |
| lcbhard_71@0.75 | lcbhard_71@0.25 | yes | stage2 | 1035.30 | 1035.50 | 23266 | natural / natural | 0.0333 |
| lcbhard_80@0.25 | lcbhard_80@0.25 | no | stage2 | 10.66 | 10.66 | 269 | natural / natural | 0.0333 |
| lcbhard_80@0.5 | lcbhard_80@0.5 | no | stage2 | 21.67 | 21.66 | 544 | natural / natural | 0.0333 |
| lcbhard_80@0.75 | lcbhard_80@0.75 | no | stage2 | 68.83 | 68.82 | 1673 | natural / natural | 0.0333 |
| lcbhard_9@0.25 | lcbhard_9@0.25 | no | stage2 | 38.83 | 38.83 | 963 | natural / natural | 0.0333 |
| lcbhard_9@0.5 | lcbhard_9@0.5 | no | stage2 | 47.22 | 47.22 | 1152 | natural / natural | 0.0333 |
| lcbhard_9@0.75 | lcbhard_9@0.75 | no | stage2 | 133.45 | 133.46 | 3136 | natural / natural | 0.0333 |
| lcbhard_99@0.25 | lcbhard_99@0.25 | no | stage2 | 847.19 | 846.97 | 21547 | natural / natural | 0.0333 |
| lcbhard_99@0.5 | lcbhard_99@0.25 | yes | stage2 | 847.19 | 846.97 | 21547 | natural / natural | 0.0333 |
| lcbhard_99@0.75 | lcbhard_99@0.25 | yes | stage2 | 847.19 | 846.97 | 21547 | natural / natural | 0.0333 |
