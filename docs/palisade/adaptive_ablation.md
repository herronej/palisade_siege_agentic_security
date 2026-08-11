<!-- Generated report. Source module: tools.adaptive_ablation. Regenerate with `uv run python -m tools.adaptive_ablation`. -->

# Adaptive-adversary ablation (WI19/WI21)

Reactive adaptive attackers (PAIR/TAP B1 + embedding A1 + cross-gate chains) through the same nine-config gate stack + Q-LLM slow tier as `full_ablation.md`, scored by the same soft/hard discriminator.

- **attacker proposer:** deterministic stand-in (offline)
- **defender Q-LLM:** openai:gpt-oss-120b
- **budget:** 2 queries/probe, **seeds:** 1

| config | n | adaptive soft-win ASR | adaptive hard-win ASR | benign BPR |
|---|---|---|---|---|
| baseline | 6 | 100% (61%-100%) | 33% (10%-70%) | 100% (86%-100%) |
| +G4 | 6 | 100% (61%-100%) | 33% (10%-70%) | 100% (86%-100%) |
| +G4+G3 | 6 | 100% (61%-100%) | 33% (10%-70%) | 100% (86%-100%) |
| +G4+G3+G1 | 6 | 83% (44%-97%) | 33% (10%-70%) | 100% (86%-100%) |
| +G4+G3+G1+G5 | 6 | 83% (44%-97%) | 17% (3%-56%) | 100% (86%-100%) |
| full PALISADE | 6 | 83% (44%-97%) | 17% (3%-56%) | 100% (86%-100%) |
| full +Semgrep | 6 | 83% (44%-97%) | 17% (3%-56%) | 100% (86%-100%) |
| full +Q-LLM | 6 | 83% (44%-97%) | 17% (3%-56%) | 100% (86%-100%) |
| full +both | 6 | 83% (44%-97%) | 17% (3%-56%) | 100% (86%-100%) |

_Compare cell-for-cell with the static ablation (`full_ablation.md`): same configs, same Q-LLM, same metric. The reactive optimizer raises the **soft**-win rate over the static floor while the **hard**-win rate tracks the static one -- the empirical soft-evadable / hard-invariant signature. Hard wins arise only in the cross-gate chains; single-gate B1/A1 remain hard-win-free at every config._
