<!-- Generated report. Source module: tools.leave_one_out_ablation. Regenerate with `uv run python -m tools.leave_one_out_ablation`. -->

# Per-gate leave-one-out ablation (E3)

Each gate's contribution **removed from the otherwise-complete stack** -- the dual of the cumulative sweep in `full_ablation.md`. Rolled up per boundary family over 205 attack + 181 benign instances (seed 42). Posture is the deterministic fast-tier `full PALISADE` (Semgrep / Q-LLM off), so every `full -Gx` column is comparable to the `full PALISADE` column of `full_ablation.md`.

> The reference row `full PALISADE` is also the paper's "Fast-tier only (all gates)" row. `-G6` is the egress grounding sink (citation-provenance binding): removing it lets forged citations reach the answer, a grounding hard win. `-G2` removes the enforcement-only wrapper, which is bundled inside `full` with the trust scorer -- read it as a floor, not a boundary defense.

## Hard-win rate (the §8.2 capability bound -- headline)

A `taint:T` value reaching a high-privilege sink. Lower is better; a `full -Gx` cell **above** the `full PALISADE` reference is the hard win that gate was holding shut.

| Configuration | B1 | B3 | B4 | B5 | XC | Overall hard-win | Benign FPR |
|---|---|---|---|---|---|---|---|
| full PALISADE | 0% | 5% | 2% | 5% | 45% | 8% | 8/181 = 4.4% |
| full -G1 | 0% | 5% | 2% | 5% | 45% | 8% | 6/181 = 3.3% |
| full -G2 | 0% | 5% | 2% | 5% | 45% | 8% | 8/181 = 4.4% |
| full -G3 | 0% | 5% | 2% | 5% | 45% | 8% | 7/181 = 3.9% |
| full -G4 | 11% | 5% | 2% | 5% | 45% | 10% | 8/181 = 4.4% |
| full -G5 | 0% | 5% | 2% | 5% | 45% | 8% | 3/181 = 1.7% |
| full -G6 | 0% | 18% | 2% | 5% | 45% | 10% | 8/181 = 4.4% |

## Soft-win ASR (inspection evasion)

Attack action admitted by a live gate (capability tag may remain intact). This is where most single-gate contributions show up at the fast tier.

| Configuration | B1 | B3 | B4 | B5 | XC | Overall ASR | Benign FPR |
|---|---|---|---|---|---|---|---|
| full PALISADE | 67% | 34% | 58% | 15% | 64% | 44% | 8/181 = 4.4% |
| full -G1 | 89% | 34% | 58% | 15% | 64% | 49% | 6/181 = 3.3% |
| full -G2 | 67% | 34% | 58% | 15% | 64% | 44% | 8/181 = 4.4% |
| full -G3 | 67% | 97% | 58% | 15% | 64% | 56% | 7/181 = 3.9% |
| full -G4 | 78% | 34% | 91% | 15% | 82% | 56% | 8/181 = 4.4% |
| full -G5 | 67% | 34% | 58% | 96% | 64% | 66% | 3/181 = 1.7% |
| full -G6 | 67% | 34% | 58% | 15% | 64% | 44% | 8/181 = 4.4% |

_Reference (`full PALISADE`): overall soft-win ASR 44%, overall hard-win 5-12% (95% CI, N=205). Cells pool the family's per-class binomials; most families are n=20-50, so read the columns, not single cells._
