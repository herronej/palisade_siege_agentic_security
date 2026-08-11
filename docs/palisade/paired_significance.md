<!-- Generated report. Source module: tools.paired_significance. Regenerate with `uv run python -m tools.paired_significance`. -->

# Paired significance tests: PALISADE vs content detectors (R4-3.4)

> **SUPERSEDED for the manuscript (2026-07-27).** These are *instance-level*
> McNemar tests, which treat the 205 attack instances as independent. They are
> not: 38 of the 42 classes are five near-identical variants of one template, so
> these p-values are optimistic. The manuscript now reports **class-level
> permutation** p-values from `cluster_stats_r4.md`, under which two of the three
> hard-win comparisons become directional rather than significant. This file is
> retained as the record of the instance-level analysis and as the cross-check
> that `tools.cluster_stats` reproduces (all six p-values match exactly). Do not
> quote it in the paper.

PALISADE scored at `full`, taint bound **production**. Every system runs over the same 205 attack instances and the same 181 benign tasks, so outcomes are paired and McNemar's exact test applies. `b` counts instances where the comparator fails and PALISADE does not; `c` the reverse. Only the discordant pairs carry information.

| Axis | Comparator | n | PALISADE | Comparator | both | b | c | exact p | favours |
|---|---|---|---|---|---|---|---|---|---|
| hard win | denylist-detector | 205 | 16 | 29 | 16 | 13 | 0 | **0.0002441** | PALISADE |
| benign block | denylist-detector | 181 | 8 | 0 | 0 | 0 | 8 | **0.007812** | denylist-detector |
| hard win | prompt-guard (any non-BENIGN) | 205 | 16 | 5 | 3 | 2 | 13 | **0.007385** | prompt-guard (any non-BENIGN) |
| benign block | prompt-guard (any non-BENIGN) | 181 | 8 | 121 | 2 | 119 | 6 | **2.321e-28** | PALISADE |
| hard win | prompt-guard (jailbreak-only) | 205 | 16 | 33 | 12 | 21 | 4 | **0.0009105** | PALISADE |
| benign block | prompt-guard (jailbreak-only) | 181 | 8 | 14 | 0 | 14 | 8 | 0.2863 | PALISADE |

_Lower is better on both axes (fewer hard wins, fewer false blocks). A bolded p is significant at 0.05. Where p is not significant the honest wording is 'improves on' rather than 'dominates'._
