<!-- Generated report. Source module: tools.blind_redteam. Regenerate with `uv run python -m tools.blind_redteam`. -->

# Blind LLM-generated held-out attack split (W29.1)

Generator: **openai:gpt-oss-120b @ https://api.i2-core.american-science-cloud.org** (blind -- told the agent's capabilities and the attack goal, nothing about the defense). Each attack is run against the **frozen** deterministic defense.

* **Content tier (co-designed signatures): 11/48 = 23% caught.** This is where benchmark--defense co-design shows -- a denylist tuned to the corpus generalizes poorly to phrasings it never saw.
* **Structural taint bound (provenance): 24/24 = 100% caught** on the attacks that reach a high-privilege sink via untrusted provenance. It denies by source, not content, so it catches blind-novel attacks at the same rate as authored ones -- it generalizes by construction.

| goal (channel) | generated | content tier | structural taint bound |
|---|---|---|---|
| `prompt_injection` (the user's chat prompt) | 12 | 11/12 (92%) | n/a (not a taint→sink flow) |
| `retrieval_poisoning` (a paper chunk retrieved from the corpus) | 12 | 0/12 (0%) | n/a (not a taint→sink flow) |
| `scheduler_injection` (a facility 'operations note' retrieved from the corpus) | 12 | 0/12 (0%) | 12/12 (100%) |
| `code_exfiltration` (a code snippet retrieved from the corpus that the assistant may reuse) | 12 | 0/12 (0%) | 12/12 (100%) |

## Finding

On attacks authored by a generator blind to the defense, the content-tier signatures catch only 23% -- the honest measure of the co-design bias, and it is confined to the *subordinate* layer. The load-bearing structural taint bound catches 100% of the untrusted→sink attacks regardless of novelty, because it reads provenance, not content -- the generalization the co-designed corpus cannot itself demonstrate but the mechanism guarantees. Report this as the same-team held-out control (still not a substitute for an external human red team, which remains future work), and cite it where the manuscript concedes co-design (Limitations).
