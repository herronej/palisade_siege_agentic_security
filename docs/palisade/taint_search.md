<!-- Generated report. Source module: tools.taint_search. Regenerate with `uv run python -m tools.taint_search`. -->

# Propagation-search adversary: ASR-vs-budget, Q→first-hard-win (W2.5)

An ε-greedy bandit searches 10 semantics-preserving laundering transforms for one that carries an untrusted payload into a capability sink tag-dropped past the taint guard. 6 seeds × 40 queries per (mode, tier); a hard win = the transformed value evades the guard (would reach the sink with its label stripped).

| guard | tier | hard-ASR | Q→first-hard-win | distinct evading transforms |
|---|---|---|---|---|
| `exact` | black_box | 97% | 1 | 9 |
| `exact` | grey_box | 96% | 1 | 9 |
| `exact` | white_box | 100% | 1 | 1 |
| `containment` | black_box | 72% | 1 | 5 |
| `containment` | grey_box | 89% | 1 | 5 |
| `containment` | white_box | 100% | 1 | 1 |

## Acceptance (W2.5): does the search beat the static xc_1 count of 4?

Against the **hardened** (containment) guard, the search finds **5** distinct semantics-preserving transforms that still evade the taint bound: `caesar_5`, `concat_2`, `concat_4`, `interleave_zwsp`, `reverse`. This exceeds the 4 static hard-wins -- so containment narrows but does not close the propagation residual: transforms that break textual survival (reversal, Caesar-n≠13, fine-grained split, zero-width interleave) still carry the payload to the sink unrecognized. Closing them needs dataflow lineage, not content containment.

## ASR-at-budget curve (containment guard, black-box)

| queries | soft-ASR | hard-ASR |
|---|---|---|
| 24 | 71% | 71% |
| 48 | 65% | 65% |
| 72 | 71% | 71% |
| 96 | 67% | 67% |
| 120 | 69% | 69% |
| 144 | 69% | 69% |
| 168 | 70% | 70% |
| 192 | 70% | 70% |
| 216 | 70% | 70% |
| 240 | 72% | 72% |
