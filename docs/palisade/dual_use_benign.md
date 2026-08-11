<!-- Generated report. Source module: tools.dual_use_benign. Regenerate with `uv run python -m tools.dual_use_benign`. -->

# Dual-use-adjacent benign utility (WI18/WI21)

Benign utility at the G1 slow tier under the deployed **unfiltered** `gpt-oss-120b` stand-in vs a **filtered** safety-aligned stand-in, sliced by whether the (benign) task touches a dual-use-adjacent topic (uranium fluorides, tritium breeding, fusion blankets).

| benign subset | n | BU (unfiltered) | BU (filtered) | filtered false-positives |
|---|---|---|---|---|
| plain | 8 | 100% | 100% | 0% |
| dual-use-adjacent | 6 | 100% | 0% | **100%** |

The filtered model false-positives on **100%** of dual-use-adjacent benign science, blocking legitimate work, while the unfiltered model admits it. Both keep plain benign utility intact. This is the operational reason PALISADE runs an unfiltered slow tier: the security guarantee is carried by the capability bound at the sink, not by a model's refusal, so a refusing model buys no security and costs benign utility.
