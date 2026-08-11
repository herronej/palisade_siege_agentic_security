<!-- Generated report. Source module: tools.dual_use_heldout. Regenerate with `uv run python -m tools.dual_use_heldout`. -->

# Held-out dual-use recall (W5.3 / R1-M6)

The honest recall figure for the G1 dual-use classifier: recall on a **held-out** framing slice (novel authorization pretexts + neutral phrasing with the ``restricted/controlled/withheld`` vocabulary removed) that varies the two features the scored `b1_10` corpus shares with the classifier's prompt rule -- so, unlike `b1_10`, it is not pre-determined. Reported instead of the circular `b1_10` number.

- **Q-LLM:** `openai:gpt-oss-120b` (single sample, the deployed gate default `quarantine_self_consistency_samples=1`).
- **Flag rule:** `dual_use_flag != none` at confidence >= 0.5 (the G1 SEV1 threshold).

| Slice | flagged / n | rate | 95% CI (exact-binomial) |
|---|---|---|---|
| **Held-out recall (report)** | 9/9 | **100%** | [66%, 100%] |
| &nbsp;&nbsp;· novel pretext | 5/5 | 100% | [48%, 100%] |
| &nbsp;&nbsp;· neutral phrasing (harder) | 4/4 | 100% | [40%, 100%] |
| `b1_10` recall (floor; template-derived) | 5/5 | 100% | [48%, 100%] |
| Benign dual-use-adjacent FPR | 0/6 | 0% | [0%, 46%] |

**Held-out recall 9/9 = 100% matches the template-derived floor -- no measurable generalization gap.** The classifier flags dual-use framing it was not tuned against (novel pretexts, neutral phrasing) at the same rate as the template it was written from, and at a 0% false-positive rate on legitimate dual-use-adjacent science (fuel-salt and fusion queries). Small n on the held-out slice, so the interval is wide; the point estimate is the honest recall the paper reports.
