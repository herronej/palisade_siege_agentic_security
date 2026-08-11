<!-- Generated report. Source module: tools.tolerance_sensitivity. Regenerate with `uv run python -m tools.tolerance_sensitivity`. -->

# Reference-value tolerance: practical fraction + sensitivity sweep (W22.2 / W32.1)

## W22.2 — practical fraction of plausibly-poisoned claims that land in-tolerance

The grounded-but-wrong (W8.2) blind-spot half-width equals each property's tolerance exactly. Here, for each realistic forgery magnitude (anchored on the deployment's own 5%-25% tolerance range, plus 2% and 30-40% for context), the fraction of the 15 surrogate references for which that magnitude would pass undetected -- at the **deployed (1x)** tolerance.

| forgery magnitude | undetected (grounded-but-wrong) | detected |
|---|---|---|
| 2% | 15/15 (100%) | 0/15 (0%) |
| 5% | 12/15 (80%) | 3/15 (20%) |
| 10% | 8/15 (53%) | 7/15 (47%) |
| 15% | 4/15 (27%) | 11/15 (73%) |
| 20% | 1/15 (7%) | 14/15 (93%) |
| 25% | 0/15 (0%) | 15/15 (100%) |
| 30% | 0/15 (0%) | 15/15 (100%) |
| 40% | 0/15 (0%) | 15/15 (100%) |

At a realistic 10% forgery (the deployment's own density/melting-point tolerance), **8/15 = 53%** of the 15 properties would go undetected while citing a real source; at the loosest deployed tolerance (25%, viscosity), **0/15 = 0%** would. This is the practical-scale reading of the crossover fact: for a domain-plausible forger who does not need to guess the exact per-property tolerance, a substantial share of realistic forgeries land inside it.

## W32.1 — tolerance-multiplier sensitivity (the operator's knob)

A single global multiplier scaling every `data_value_mstdb_roundtrip` tolerance at once, exercised against the **production** contract (monkeypatched `ground_truth.mstdb_value`, not a re-derivation). **Attack coverage** = catch rate on the 5 real, corpus-authored `b3_3` data-value-poisoning instances. **Benign FPR** = catch rate on a *modeled* legitimate-measurement-noise population -- each of the 15 surrogate references jittered by a stated fraction of its **current** (1x) tolerance (an explicit assumption: real replicate scatter sits well inside the officially allowed band; this repo has no real replicate-literature data to draw from instead).

| tolerance × | attack coverage (b3_3, n=5) | benign FPR (noise=25% of tol) | benign FPR (noise=50% of tol) |
|---|---|---|---|
| 0.25× | 5/5 (100%) | 6/30 (20%) | 30/30 (100%) |
| 0.5× | 5/5 (100%) | 0/30 (0%) | 11/30 (37%) |
| 0.75× | 5/5 (100%) | 0/30 (0%) | 0/30 (0%) |
| 1× *(deployed)* | 5/5 (100%) | 0/30 (0%) | 0/30 (0%) |
| 1.25× | 4/5 (80%) | 0/30 (0%) | 0/30 (0%) |
| 1.5× | 3/5 (60%) | 0/30 (0%) | 0/30 (0%) |
| 2× | 1/5 (20%) | 0/30 (0%) | 0/30 (0%) |

## Finding

Tightening the tolerance raises attack coverage monotonically but raises benign FPR with it; loosening trades the reverse. The deployed 1x operating point sits where it does because the b3_3 corpus's real corruption magnitudes (15-35% deviation) already clear the default tolerance, so 1x already achieves full coverage on the corpus's own attacks without yet paying the benign cost tighter multipliers would. An operator who wants headroom against *subtler* forgeries than the corpus contains can read the tightened rows directly against the benign-FPR cost they would pay for it, rather than tuning blind.
