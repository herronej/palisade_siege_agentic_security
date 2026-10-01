# Prompt-only baseline (E1a): provenance

R1 asked whether PALISADE needed any code: "give the policy, paper, and
codebase to a capable agent such as Claude Code or Codex and ask it to express
the in-code policy as detailed instructions." This directory is that artifact,
frozen before any run used it.

## Files

| File | What it is |
|---|---|
| `policy.md` | The policy body the agent wrote. Frozen (read-only); sha256 `493b3e5569618cbeb76093ac428be9ac3c87b711498313b2e95b94ecaa39ca63`, 8,135 words. |
| `generation_prompt.md` | The exact instructions the agent was given. |
| `inputs.sha256` | sha256 of all 69 files staged as the agent's only inputs. |
| `files_read.txt` | The 67 of those the agent reports reading. |

## How it was produced

- **Agent:** a fresh Claude Code subagent (Claude Opus 5) with no prior context, 2026-09-16; 99 tool calls, about 35 minutes.
- **Inputs, staged into one directory and hashed before the run:**
  - PALISADE's source (`src/palisade/`, without `tests/`) at repository commit `3a33b36`, plus the uncommitted posture-flag changes (defaults unchanged);
  - the contract library (`palisade_contracts/`, without `tests/`);
  - the paper: the -21 draft as PDF and as extracted text.
- **Isolation:** the agent was told to read nothing else, in particular nothing from the benchmark, its instances, the evaluation reports or the analysis tools. `files_read.txt` lists only staged files. The policy names no benchmark class or instance.
- **Unread inputs:** a `.DS_Store` file, and `paper.pdf`, which the agent could not render; it worked from the extracted text, so it did not see Fig. 1.

## Choices the agent reported

- **Configuration encoded:** the strongest one: every gate, the model-backed tier and static analysis on, with the code's defaults. That means the weapons-phrase screen on, lifecycle-hook denials at SEV1, and sticky high-stakes denials on.
- **Answers:** citation and value violations are written as DENY. The deployment default only annotates the answer; a binary monitor needs the stricter form.
- **User exemption:** the taint rule uses the source-exempt form the paper says the evaluation scores. The deployed tool gate is stricter, and the policy says so in a note.
- **Reworded content:** it counts as untrusted when an untrusted source *chose* an operative element of the action. This follows the recorded-label intent, and is broader than the code's content match (whole values, and distinctive fragments of at least 12 characters).
- **Memory:** reads and writes are handled with the retrieval rules.

## What it could not express, or only approximately

- **Hash and embedding checks** (corpus pin, tool-descriptor pins, embedding z-scores): described, but a model cannot recompute them.
- **Label bookkeeping:** exact-string keys and fragment matching are stated as rules; a model applies them less precisely.
- **Rewrite-and-allow paths:** argument minimization and sanitizing a return become "deny and re-issue" or "strip, then allow".
- **Trust arithmetic:** stated as numbers; the monitor keeps the tally itself.
- **Code not wired into the sidecar:** the agent observed that the scheduler-field rule and the dependency walk's resolver are not connected in `sidecar.py`, and included both as policy.

## How the runs use it

- **Monitor, SIEGE:** `MONITOR_HEADER + policy.md + siege_deployment_config()`, all in `../prompt_monitor.py`.
  - The third part is needed because the policy's allocation list and ceilings come from the operator file shipped with the code (`stf218`, `gen150-vista`; 2 h; `batch`).
  - SIEGE builds G5 with a different benchmark policy (`msr_thermo`; 48 h; any partition).
  - The gates get that configuration as constructor arguments. The monitor gets the same values, rendered from the same objects.
  - `policy.md` is not edited. The hash of the whole prompt is recorded in every run.
- **Planner, AgentDojo:** AgentDojo's default system message, then `PLANNER_HEADER`, then `policy.md` (`tools/agentdojo_e1.py --prompt-policy`). No deployment configuration is added, since the policy's HPC settings have no counterpart there.
