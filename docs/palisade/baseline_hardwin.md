<!-- Generated report. Source module: tools.baseline_hardwin. Regenerate with `uv run python -m tools.baseline_hardwin`. -->

# Detection-only baseline hard-win rate (tab:e1 YY)

A detection-only defense (`denylist-detector`, its own public injection/jailbreak/mining/IOC patterns -- a LlamaFirewall / PromptGuard-style surface detector) over the full 200-instance SIEGE corpus, scored on the same hard-win metric as PALISADE. It blocks an attack iff it flags the content; where it misses, the value reaches the sink as undefended (no capability backstop), so `baseline_hard_win = undefended_hard_win AND not flagged`.

## Headline (security subset, matches the tab:e2 basis)

| Defense | Hard-win rate |
|---|---|
| Undefended (no defense) | **21%** |
| Detection-only baseline (`denylist-detector`) | **16%** |
| PALISADE (full) | **3%** |

_Security subset: 150 instances (misuse + science excluded). The detector's surface-catch rate over all attacks is 20%, but its catch rate over the undefended **hard-win** instances is only 22% -- the hard-wins are tag-dropped values whose text is benign, so a content detector cannot see them. That is the structural gap: the baseline leaves nearly the whole undefended hard-win rate standing; the capability bound is what closes it._

## All-200 basis

| Defense | Hard-win rate |
|---|---|
| Undefended | 18% |
| Detection-only baseline | 14% |
| PALISADE (full) | 2% |

## Per family

| Family | n | Undef HWR | Baseline HWR | PALISADE HWR | Detector catch on hard-wins |
|---|---|---|---|---|---|
| b1 | 45 | 11% | 0% | 0% | 100% |
| b3 | 38 | 21% | 21% | 0% | 0% |
| b4 | 45 | 11% | 9% | 0% | 20% |
| b5 | 55 | 9% | 9% | 0% | 0% |
| xc | 22 | 64% | 55% | 18% | 14% |
