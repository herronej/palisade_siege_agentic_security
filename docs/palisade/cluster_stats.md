<!-- Generated report. Source module: tools.cluster_stats. Regenerate with `uv run python -m tools.cluster_stats`. -->

# Cluster-robust intervals over attack classes (R4 / R2-12)

The 205 attack instances are clustered in **42 authored classes**, 38 of them five near-identical variants of one template along one variation axis. Exact-binomial intervals treat them as independent draws, which understates uncertainty. Below, the class is the resampling unit: 10,000 nonparametric cluster-bootstrap replicates, percentile interval. The naive column is the instance-level exact-binomial interval the paper currently reports.

**The benign control is not clustered the same way and we do not pretend otherwise.** Its 157 diverse tasks carry 157 *distinct* variation axes -- they are different tasks, not variants of one -- so the instance-level interval is appropriate there. The 5-way row below resamples the four authored categories plus the original workload as a conservative sensitivity check only; at five clusters the percentile interval is unstable and should not be read as the headline uncertainty.

| statistic | point | cluster 95% CI | naive 95% CI |
|---|---|---|---|
| undefended hard win (205) | 18.0% | [7.2, 30.2] | [13.0, 24.0] |
| deployed hard win (205) | 6.3% | [1.0, 12.8] | [4.5, 12.4] |
| residual | undefended HW (37) | 35.1% | [10.0, 64.3] | [27.1, 60.5] |
| benign false-positive (181) | 4.4% | [0.6, 8.5] | [1.9, 8.5] |

The cluster intervals are the honest ones. Where they are much wider than the naive interval, the instance-level figure was borrowing confidence from within-class repetition.

## Leave-one-class-out sensitivity

Deployed hard-win rate with each class removed in turn (base 6.3%). Answers whether one authored family carries the headline.

**Range 4.5%--6.5%.** Dropping `b1_10_dual_use_framing` raises it most (6.5%); dropping `xc_1_cross_boundary_chain` lowers it most (4.5%).

| class dropped | deployed HW | delta |
|---|---|---|
| `xc_1_cross_boundary_chain` | 4.5% | -1.9 |
| `xc_5_history_forgery` | 4.5% | -1.8 |
| `b3_9_tool_return_injection` | 5.4% | -0.9 |
| `xc_4_taint_laundering` | 5.5% | -0.8 |
| `b4_9_corpus_injected_code` | 6.0% | -0.3 |
| `b1_10_dual_use_framing` | 6.5% | +0.2 |
| `b1_1_direct_injection` | 6.5% | +0.2 |
| `b1_2_credentialing` | 6.5% | +0.2 |

_Eight classes with the largest effect; the rest move it less._

## Class-level paired permutation vs. McNemar

Per-class difference in hard wins or false blocks (PALISADE minus comparator), sign-flipped at the class level over 10,000 permutations. McNemar's instance-level p is shown beside it: it assumes exchangeable instances, which clustering violates, so it is the optimistic one.

| axis | comparator | mean class diff | k | perm p | min p | McNemar p |
|---|---|---|---|---|---|---|
| hard win | denylist-detector | -0.38 | 6 | 0.0313 | 0.031 | 3.05e-05 |
| benign block | denylist-detector | +1.60 | 3 | 0.2434 | 0.250 | 7.81e-03 |
| hard win | prompt-guard (any non-BENIGN) | +0.19 | 6 | 0.3989 | 0.031 | 9.63e-02 |
| benign block | prompt-guard (any non-BENIGN) | -22.60 | 5 | 0.1266 | 0.062 | 2.32e-28 |
| hard win | prompt-guard (jailbreak-only) | -0.48 | 8 | 0.0737 | 0.008 | 1.80e-04 |
| benign block | prompt-guard (jailbreak-only) | -1.20 | 3 | 1.0000 | 0.250 | 2.86e-01 |

`k` is the number of classes with a nonzero difference -- the units the test can actually flip -- and `min p` the smallest two-sided p attainable at that `k`. Where `min p` is near or above 0.05 the design cannot show significance whatever the effect size, so a large p there is an absence of power, not evidence of no difference. The benign rows are in that regime by construction: five clusters admit no p below 0.0625.

**The hard-win rows are the substantive result, and they do not all survive.** McNemar reads the corpus as 205 independent instances; once the class is the unit, only a handful of classes carry each comparison, and the evidence is correspondingly weaker. Any comparison whose permutation p exceeds 0.05 should be reported as directional rather than significant.

## The two comparators the panel above does not cover (R13-14)

The panel tests fixed content classifiers. It misses the peer comparison the architectural-price argument rests on, and the judge -- the one comparator that beats us on the static hard-win axis. **Denominators differ by row and are stated, not pooled:** the value-lineage peer's published design models four sink types, so it is scored on the shared-sink hard wins and the classes inside them, not on the 205.

| axis | comparator | denominator | mean class diff | k | perm p | min p | McNemar p |
|---|---|---|---|---|---|---|---|
| hard win | value-lineage peer (vs deployed) | 27 shared sinks | +1.50 | 4 | 0.1232 | 0.125 | 3.91e-03 |
| hard win | value-lineage peer (vs label predicate) | 27 shared sinks | +2.67 | 5 | 0.0596 | 0.062 | 3.05e-05 |

A **positive** mean class difference is against us: it counts hard wins (or benign false blocks) our mechanism leaves and the comparator closes.

**Peer comparison.** On the 27 shared-sink hard wins across 6 classes, the lineage peer closes **23**, our deployment **14**, our label predicate alone **7**. With only 6 classes the sign-flip design bottoms out at the `min p` shown, so read a large p as absent power rather than as evidence of no difference.

_Generated by `tools.cluster_stats`._
