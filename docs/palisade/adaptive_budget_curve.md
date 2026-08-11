<!-- Generated report. Source module: tools.adaptive_budget_curve. Regenerate with `uv run python -m tools.adaptive_budget_curve`. -->

# Adaptive ASR-vs-budget curves with seed-level CIs (W19)

Two hard-win drivers vs their **production** content-based guard, 20 independent seeds per cell. Each seed's hard-ASR is one observation; the interval is a **seed-level percentile bootstrap** (the cluster-robust CI -- episodes within a bandit run are correlated, so we do not pool them into one binomial, per R3-8). The declarative capability bound is 0% at every budget/tier (the sink policy is sound) and is not swept; these curves isolate the label-propagation residual (§III).

### Propagation-search (W2 — transform laundering into the sink)

Hard-win ASR (mean over 20 seeds) [95% seed-bootstrap CI], by tier × budget.

| tier | Q=16 | Q=64 | Q=256 | Q=1000 | Q→first-hard |
|---|---|---|---|---|---|
| black_box | 70% [66%,73%] | 74% [72%,75%] | 75% [74%,75%] | 75% [74%,75%] | 1 |
| grey_box | 72% [66%,78%] | 84% [82%,87%] | 89% [88%,89%] | 90% [89%,90%] | 1 |
| white_box | 100% [100%,100%] | 100% [100%,100%] | 100% [100%,100%] | 100% [100%,100%] | 1 |

### Scheduler-injection (W3 — untrusted submission field)

Hard-win ASR (mean over 20 seeds) [95% seed-bootstrap CI], by tier × budget.

| tier | Q=16 | Q=64 | Q=256 | Q=1000 | Q→first-hard |
|---|---|---|---|---|---|
| black_box | 45% [35%,55%] | 59% [56%,62%] | 65% [64%,67%] | 67% [66%,67%] | 6 |
| grey_box | 47% [32%,61%] | 72% [65%,79%] | 83% [81%,85%] | 85% [85%,86%] | 7 |
| white_box | 100% [100%,100%] | 100% [100%,100%] | 100% [100%,100%] | 100% [100%,100%] | 1 |

## Finding

The hard-win ASR rises with query budget and separates by access tier (white-box knows the guard and evades on the first query; black-/grey-box must search), with the CI narrowing as the budget grows -- the worst-case curve the fixed budget-16 snapshot could not show. The residual is bounded: even at 1000 queries the containment guard holds the propagation surface to the characterized evading-transform set, and the scheduler residual to the short non-distinctive fields (§III). Report these curves in place of the single-budget Table E4 point estimates, with the CIs, and drop the unqualified 'across three access tiers' claim in contribution-4 for what the curves actually show.
