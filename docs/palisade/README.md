# Result-to-module map

Every number the paper reports is produced by a named module in this
repository. This file is the index: find the claim, get the module that
generates it and the report it writes.

Run any module from the repository root:

```bash
uv run python -m tools.<module>
```

Each writes its Markdown report into this directory. Reports committed here are
the state behind the reported numbers at the release tag — re-running a module
overwrites its report, which is how you check a reproduction against ours.

**Column meanings.** *Offline* means the result reproduces deterministically at
seed 42 with no network and no endpoint. *Served* means it needs an
OpenAI-compatible endpoint (see `REPRODUCING.md`); those reproduce to the
reported **range**, not to point equality. *Slurm* means it needs the
containerized controller under `docker/slurm/`.

---

## Abstract and headline claims

| Claim | Module | Report | Mode |
|---|---|---|---|
| 205 attack instances, 42 classes, 181 benign controls | `tools.corpus_datasheet` | `corpus_datasheet.md` | offline |
| Hard wins 13/32 at `full`, 4/32 at `+all` | `tools.soft_hard_decomposition` | `soft_hard_decomposition.md` | offline / served |
| PAIR and TAP return 28.0–31.0 of 32 | `tools.optimizer_hardwin` | `optimizer_hardwin.md` | served |
| Recorded-label residual 4 of 32, invariant | `tools.optimizer_hardwin` | `optimizer_hardwin.md` | served |
| 3.3% of benign controls blocked | `tools.benign_fpr` | `benign_fpr.md` | offline |
| 0.28 ms per benign turn | `tools.overhead` | `overhead.md` | offline |

## Section III — Architecture

| Claim | Module | Report | Mode |
|---|---|---|---|
| §III-B fast-tier per-gate cost; 0.28 ms per turn | `tools.overhead` | `overhead.md` | offline |
| §III-C G5's eleven deterministic policy checks | `siege.eval.g5_runner` | — | offline |
| §III-C G5 parse soundness under mutation | `tools.fuzz_slurm_parser` | — | offline |
| §III-E propagation closure coverage (L̂ vs L\*) | `tools.taint_coverage` | `taint_coverage.md` | offline |
| §III-F contract library, effective coverage | `siege.report_suites` | — | offline |

## Section IV — SIEGE

| Claim | Module | Report | Mode |
|---|---|---|---|
| Table III family composition; per-class counts and provenance | `tools.corpus_datasheet` | `corpus_datasheet.md` | offline |
| §IV-B benign control composition (24 workload + 157 near-manifold) | `tools.benign_workload_fpr` | `benign_workload_fpr.md` | offline |
| §IV-B G2's 100-scenario off-corpus harness | `siege.eval.g2_runner` | — | offline |
| §IV-C hard-win adjudication and its two filters | `siege.scorer` | — | offline |
| Appendix C-B 40-instance stratified re-adjudication, 40/40 agreement | `tools.scorer_validation` | `scorer_validation.md` | offline |

## Section V — Evaluation

### V-A Setup and enforcement-point validity

| Claim | Module | Report | Mode |
|---|---|---|---|
| All 55 B5 submissions driven through a live `slurmctld` | `tools.slurm_b5_live_sweep` | `slurm_b5_live_sweep.md` | slurm |
| No parse disagreement on partition, nodes, wall-clock | `tools.slurm_live_differential` | `slurm_live_differential.md` | slurm |
| Parser differential vs. the controller's own grammar | `tools.slurm_parser_differential` | `slurm_parser_differential.md` | slurm |
| Comparison against a facility `job_submit.lua` policy | `tools.slurm_policy_peer` | `slurm_policy_peer.md` | offline |

### V-B Corpus results (the nine-configuration ladder)

| Claim | Module | Report | Mode |
|---|---|---|---|
| Ablation across all nine configurations | `siege.full_ablation` | `full_ablation.md` | offline / served |
| Attack success 100% → 4%; hard wins 32 → 13 → 4 | `tools.soft_hard_decomposition` | `soft_hard_decomposition.md` | offline / served |
| Figure 2, per-boundary decomposition | `tools.eval_figures` | `figures/` | offline |
| Ablation figures as rendered plots | `tools.ablation_plots` | `figures/` | offline |
| `+semgrep` closes B4.9 over the model-backed tier | `siege.full_ablation` | `full_ablation.md` | offline |

### V-C Where the guarantee sits

| Claim | Module | Report | Mode |
|---|---|---|---|
| Soft-win vs hard-win separation along the ladder | `tools.soft_hard_decomposition` | `soft_hard_decomposition.md` | offline |
| Cluster-robust intervals [0.0, 8.2] and [2.0, 17.5] | `tools.cluster_stats` | `cluster_stats.md` | offline |
| Leave-one-class-out: `full` hard-win within 6.2–9.0% | `tools.leave_one_out_ablation` | `leave_one_out_ablation.md` | offline |
| Per-gate leave-one-out attribution | `tools.leave_one_out_ablation` | `leave_one_out_ablation.md` | offline |

### V-D The fail-open on absence

| Claim | Module | Report | Mode |
|---|---|---|---|
| XC.1's `create_file` sink carries no capability field | `tools.taint_coverage` | `taint_coverage.md` | offline |
| 55 of 181 benign controls reach a sink; 54 carry an unresolvable argument | `tools.taint_interlock` | `taint_interlock.md` | offline |
| The remediation's cost is unpriced on this corpus | `tools.taint_interlock` | `taint_interlock.md` | offline |
| What that remediation actually costs on legitimate retrieve-then-act work (132-session provenance control) | `tools.failclosed_provenance` | `failclosed_provenance.md` | offline |
| AgentDojo bounds that rule's cost at 17% | `tools.agentdojo_e1` | `agentdojo_e1.md` | served |

### V-E Adaptive attacks (Table I)

| Claim | Module | Report | Mode |
|---|---|---|---|
| Table I, PAIR and TAP over the 32 sink-capable instances | `tools.optimizer_hardwin` | `optimizer_hardwin.md` | served |
| Results by access tier | `tools.adaptive_tier_table` | `adaptive_tier_table.md` | served |
| ASR-vs-query-budget curves with seed-level intervals | `tools.adaptive_budget_curve` | `adaptive_budget_curve.md` | served |
| Budget-curve figure | `tools.budget_curve_figure` | `figures/` | offline |
| Adaptive ablation across configurations | `tools.adaptive_ablation` | `adaptive_ablation.md` | served |
| Propagation-search adversary (not scored in Table I) | `tools.taint_search` | `taint_search.md` | served |
| Ceiling on what propagation-by-reference (opaque handles) could close | `tools.handle_coverage` | `handle_coverage.md` | offline |
| Adaptive scheduler-injection adversary vs. live G5 | `tools.g5_search` | `g5_search.md` | served |
| Adaptive taint-clearing via slow-tier sanitization | `tools.sanitize_search` | `sanitize_search.md` | served |

### V-F Operating cost

| Claim | Module | Report | Mode |
|---|---|---|---|
| 8 of 181 blocked (4.4%); 7 lockouts (3.9%, CI [1.6, 7.8]) | `tools.benign_fpr` | `benign_fpr.md` | offline |
| Disabling the G1 weaponization signature: 13/32 either way | `tools.benign_fpr` | `benign_fpr.md` | offline |
| Benign-FPR operating curve over signature families | `tools.fpr_curve` | `fpr_curve.md` | offline |
| Utility / DoS dashboard, task completion under the guard | `tools.utility_dos` | `utility_dos.md` | offline |
| End-to-end utility with the guard on | `tools.end_to_end_utility` | `end_to_end_utility.md` | served |
| Table VI per-gate latency and sustained throughput | `tools.overhead` | `overhead.md` | offline |
| Throughput and tail latency under concurrency (1–512) | `tools.gate_scaling` | `gate_scaling.md` | offline |

## Appendices

| Claim | Module | Report | Mode |
|---|---|---|---|
| Appendix D, Table V — AgentDojo v1.2.1, four suites | `tools.agentdojo_e1` | `agentdojo_e1.md` | served |
| Appendix E-B — trust-scorer probe-then-strike sweep | `tools.trust_oscillation` | `trust_oscillation.md` | offline |
| Appendix E-B — trust scorer as a curve | `tools.trust_curve` | `trust_curve.md` | offline |
| Appendix E-C — per-gate cost breakdown | `tools.overhead` | `overhead.md` | offline |
| Appendix B — contract library and its effective coverage | `tools.tolerance_sensitivity` | `tolerance_sensitivity.md` | offline |

## Comparisons and controls not in the headline

These are reported around the main claims — baselines the paper positions
against, and controls that bound what a claim does *not* establish.

| Subject | Module | Report | Mode |
|---|---|---|---|
| Detection-only baseline (real Prompt-Guard-86M vs. the regex reimplementation) | `tools.detector_baselines` | `detector_baselines.md` | offline |
| Expanded detector panel (Prompt Guard 2, LLM judge) | `tools.detector_panel_r4` | `detector_panel.md` | served |
| Spotlighting and the human-approval bound | `tools.baseline_panel_r5` | `baseline_panel.md` | offline |
| Dataflow-IFC peer (CaMeL / FIDES) shared-sink coverage | `tools.dataflow_peer` | `dataflow_peer.md` | offline |
| SIEGE instances outside every IFC baseline by construction | `tools.siege_ifc_coverage` | `siege_ifc_coverage.md` | offline |
| LLM judge wired inside the stack | `tools.judge_slow_tier` | `judge_slow_tier.md` | served |
| Judge union with the label bound | `tools.judge_union` | `judge_union.md` | served |
| PAIR/TAP against the judge | `tools.judge_adaptive` | — | served |
| Blind LLM-generated held-out attack split | `tools.blind_redteam` | `blind_redteam.md` | served |
| Held-out dual-use recall (the honest control; **not** `b1_10`) | `tools.dual_use_heldout` | `dual_use_heldout.md` | served |
| Dual-use across CYBER / CHEM / BIO / NUCLEAR tiers | `tools.dual_use_tiers` | `dual_use_tiers.md` | served |
| Dual-use sink axis firing independently of taint | `tools.dual_use_sink` | `dual_use_sink.md` | offline |
| Benign tool→sink friction (taint-bound cost) | `tools.tool_sink_fpr` | `tool_sink_fpr.md` | offline |
| Benign provenance-carrying control | `tools.benign_provenance_control` | — | offline |
| Citation-grounding control: what G6 cannot catch | `tools.grounding_control` | `grounding_control.md` | offline |
| Grounded-but-wrong blind spot | `tools.grounded_but_wrong` | `grounded_but_wrong.md` | offline |
| Non-privileged-sink over-count control | `tools.nonpriv_overcount` | `nonpriv_overcount.md` | offline |
| Paired significance for the Pareto claim | `tools.paired_significance` | `paired_significance.md` | offline |
| Competitive robustness sweep | `tools.competitive_robustness` | `competitive_robustness.md` | served |
| Baseline hard-win under a detection-only defense | `tools.baseline_hardwin` | `baseline_hardwin.md` | offline |
| G1 dual-use classifier precision/recall | `tools.dual_use_precision` | — | served |
| G4/G5 code-intent precision on HPC scripts | `tools.hpc_code_intent_precision` | — | served |
| Dual-use-adjacent benign slice (feeds the FPR breakdown) | `tools.dual_use_benign` | `dual_use_benign.md` | offline |
| In-context probe for the `b5_3` `python_fetch_exec` residual | `tools.hpc_fetch_exec_probe` | — | served |

Two modules generate no result of their own. `tools.palisade_screen` holds the
screening primitives the AgentDojo integration calls (Appendix D) and is
exercised through `tools.agentdojo_e1`; `tools.eval_figures` and
`tools.ablation_plots` render figures from reports other modules write.

---

## Verifying this map

`tools/tests/` holds a test per analysis module, pinning the shape of its
report and — where the number is a fixed corpus property — the number itself.
`uv run pytest tools/` checks the map's offline half without re-running the
served rows.
