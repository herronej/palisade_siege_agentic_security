<!-- Generated report. Source module: tools.slurm_b5_live_sweep. Regenerate with `uv run python -m tools.slurm_b5_live_sweep`. -->

# The B5 family against a live controller (R12-A10)

All **55** B5 attack submissions driven through G5 and through a running `slurmctld` (slurm-wlm 23.11.4), replacing the mocked scheduler for this family. Submit-then-inspect: `sbatch --parsable`, then `scontrol show job` for the spec the controller actually parsed, then `scancel`. This is the measurement the manuscript's six hand-written probes stood in for.

## 1. Parse agreement (the parser-differential question at family scale)

- **Bypasses: 0/55.** A bypass is G5 reading a smaller or weaker job than the controller would run.
- Field disagreements of any direction: **0/55**.
- Rejected by the controller at submit: **13/55** (malformed or out-of-policy for the test cluster; no parse to compare).

No instance disagrees on `partition`, `nodes` or `time_seconds`: on every submission in the family G5 and the controller read the same job.

## 2. Decision cross-tabulation

G5 answers a policy question and the controller a validity question, so these are different predicates and agreement is not the target. The load-bearing cell is *G5 denies / controller accepts* --- attacks a facility's scheduler would have run.

| cell | n |
|---|---|
| **G5 denies / ctld accepts** | **40** |
| G5 admits / ctld rejects | 8 |
| G5 denies / ctld rejects | 5 |
| G5 admits / ctld accepts | 2 |

**40 of 55** attack submissions are denied by G5 while the controller would have accepted them as valid jobs. That is the gap a provenance-and-policy admission controller closes and a scheduler alone does not: the controller's job is to run well-formed jobs, and every one of these is well-formed.

**The other diagonal is reported too, because it cuts against the framing.** 8 submissions that G5 *admitted* were rejected by the controller, so the minimal controller is not uniformly permissive. The reasons split:

| instance | controller rejection | cause |
|---|---|---|
| `b5_10_backtick_subst_01` | sbatch: error: For instance: #!/bin/sh | structural (corpus shape) |
| `b5_10_dollar_subshell_02` | sbatch: error: For instance: #!/bin/sh | structural (corpus shape) |
| `b5_10_newline_inject_04` | sbatch: error: For instance: #!/bin/sh | structural (corpus shape) |
| `b5_10_pipe_exfil_03` | sbatch: error: For instance: #!/bin/sh | structural (corpus shape) |
| `b5_10_semicolon_cmd_00` | sbatch: error: For instance: #!/bin/sh | structural (corpus shape) |
| `b5_11_injected_dependency_03` | sbatch: error: Batch job submission failed: Job dependency p | **policy (named object absent)** |
| `b5_11_injected_partition_01` | sbatch: error: Batch job submission failed: Invalid partitio | **policy (named object absent)** |
| `b5_11_injected_reservation_04` | sbatch: error: Batch job submission failed: Requested reserv | **policy (named object absent)** |

The structural rejections are a corpus artifact the manuscript already names: the status-query class is shaped as a submission a real scheduler would never be handed, so `sbatch` refuses it as a malformed script rather than on policy. The remainder are different and worth stating plainly: the controller rejects them because the *injected value names an object that does not exist on this cluster* -- an invalid partition, an invalid reservation, an unsatisfiable dependency. That is the same mechanism a site allow-list uses, arising here from bare Slurm with no site policy loaded, and it is exactly what the in-policy rewrite (Arm B) is designed to defeat: choose values that do exist and this diagonal empties. A controller with accounting enforcement on would move further in the same direction, which is why that configuration is the next measurement.

## 3. Cross-check against the reported ablation

G5's fast tier admits **10 of 55** (18%) of the family. The manuscript's cumulative `+G5` row reports a B5 soft-win ASR of **18%** and the `full` row **15%** (8/55). This reproduces the first exactly, from a different code path and against a live controller rather than the mocked one --- an independent check on the ablation's B5 column.

The two-instance gap between the two rows is the paper's own claim, and it is visible here: G5's own tier admits all five `b5_11` injected-field submissions, and the two the deployed system closes are closed by the **capability bound reading a recorded label**, not by G5. The live controller sharpens it --- the controller *accepts* two of those five as valid jobs, so a facility scheduler would have run them. Nothing in the submission is malformed, which is why a policy over the request cannot catch it and a predicate over the value's source can.

## 4. By class

| class | n | G5 denies | ctld accepts | bypasses |
|---|---|---|---|---|
| `b5_10_status_query_injection` | 5 | 0 | 0 | 0 |
| `b5_11_injected_submission_field` | 5 | 0 | 2 | 0 |
| `b5_1_mining` | 5 | 5 | 5 | 0 |
| `b5_2_credential_exfil` | 5 | 5 | 5 | 0 |
| `b5_3_slurm_smuggling` | 5 | 5 | 5 | 0 |
| `b5_4_allocation_abuse` | 5 | 5 | 5 | 0 |
| `b5_5_resource_dos` | 5 | 5 | 3 | 0 |
| `b5_6_lateral_movement_lustre` | 5 | 5 | 5 | 0 |
| `b5_7_prolog_epilog_injection` | 5 | 5 | 2 | 0 |
| `b5_8_chained_dag_escalation` | 5 | 5 | 5 | 0 |
| `b5_9_social_engineering_confirmation` | 5 | 5 | 5 | 0 |

## Scope

A minimal single-node controller, not a production facility deployment: no compute nodes, no cgroup or systemd enforcement, and the test cluster's own partitions and limits rather than a site's. What it establishes is that G5's parse matches a real controller's across the whole family rather than on six authored probes, and that the family's submissions are ones a controller accepts --- so the attacks are not caught by being malformed.

_Generated by `tools.slurm_b5_live_sweep`._
