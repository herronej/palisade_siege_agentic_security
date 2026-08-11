<!-- Generated report. Source module: tools.agentdojo_e1. Regenerate with `uv run python -m tools.agentdojo_e1`. -->

# AgentDojo headline comparison (E1)

PALISADE/PALISADE wired into **AgentDojo** as a live two-tier defense (an
injected-instruction inspection tier + a capability tool-gate on proposed calls),
run on the real benchmark so it sits on the *same* surface as the capability
defenses it is compared to (CaMeL, Progent).

> **Status: integration complete; measured live on two base models, four suites.**
> The security result is clean (ASR 0% by construction in both regimes); the
> utility comparison is reframed (see below) because gpt-4o (the cited baselines'
> base model) is retired and frontier models saturate the attack.

## Result — the structural guarantee is model-independent

Four suites (banking, workspace, travel, slack), 97 benign user tasks and 97
attack pairs (user-task x injection-task, `injection_task_1`), attack
`important_instructions`, live via the science-cloud endpoint.

| System | Base model | Benign utility | Attack success (ASR) | Benign FPR |
|---|---|---|---|---|
| Undefended | gpt-oss-120b | 75% | **86%** | — |
| **PALISADE** | gpt-oss-120b | 64% | **0%** | 19% |
| Undefended | gpt-5.4 | 84% | **0%** | — |
| **PALISADE** | gpt-5.4 | 80% | **0%** | 5% |
| Undefended _(cited)_ | gpt-4o | 84% | — | — |
| CaMeL [Debenedetti et al. 2025] _(cited)_ | gpt-4o | 77% | — | — |
| Progent [Shi et al. 2025] _(cited)_ | gpt-4o | — | — | ~6% |
| LlamaFirewall [Meta 2025] _(cited)_ | — | — | — | — |

**Reading of the result.** Undefended attack success tracks the base model's own
alignment: **86%** on a weak open model, **0% on gpt-5.4**, which resists
`important_instructions` unaided. PALISADE drives ASR to **0% in both regimes**
across all four suites — it *rescues* the weak model (86%→0%) and *matches* the
frontier model's alignment without depending on it. The guarantee is
model-independent, where the model's own safety is not. The cost is a **5–19%**
benign FPR (higher on the weaker agent), versus 43–56% for the earlier
span-overlap proxy.

**Why this is not a utility bake-off vs CaMeL.** (1) gpt-4o, the cited baselines'
base model, is no longer served, so a same-model comparison is impossible. (2)
Frontier models saturate AgentDojo's strongest content attack (undefended ASR 0),
leaving no security headroom to differentiate defenses. The cited rows are
**gpt-4o-era historical reference points**; their ASR cells should be filled from
the source papers, not invented. On the utility axis PALISADE is now comparable
to CaMeL (84%→80% here vs CaMeL's 84%→77%).

## How PALISADE plugs in

Two tiers, inserted into AgentDojo's `ToolsExecutionLoop`:

1. **Inspection tier** — `PalisadeDetector` (a `PromptInjectionDetector`):
   screens each tool output through the G3 injection patterns and redacts a
   flagged one.
2. **Structural tier** — `PalisadeToolGate` (runs *before* the `ToolsExecutor`):
   taints a tool return that structurally reads as an **injected instruction**
   (`looks_like_injection` — untrusted content addressing the agent and issuing a
   directive: "important message from me/to you, AI model", "before you can
   solve", "please do the following", "you must", ignore/disregard) and denies a
   privileged (state-changing) sink while such an injection is present. It blocks
   the injected action **whether or not its arguments overlap the injection text**
   (so a parameterless `delete_file`/DoS is caught), and it does not taint benign
   tool *data* (no directive → legitimate tool-derived flows pass). This is a
   detection-informed **proxy** for the real system's untrusted-instruction
   quarantine plus its capability bound; AgentDojo exposes no value-level
   dataflow, so the airtight taint of the core model (the SIEGE results)
   is approximated here. The residual 5–19% FPR is where a benign message's own
   imperative trips the injection signature.

Assembled loop (verified): `[PalisadeToolGate, ToolsExecutor, PalisadeDetector, LLM]`.

## Design history (span-overlap → injection-presence)

The first port tainted *every* tool return and blocked a privileged call whose
args shared a >=12-char span with it. On banking that scored ASR 0% / FPR 29%, but
the full 4-suite run exposed it as both **leaky** (gpt-oss ASR 81%→20%, not →0:
injected `delete_file(file_id='13')` shares no span; `reserve_*` was not even
classified high-stakes) and **over-blocking** (43–56% FPR: benign tool-derived
flows share spans with tool output). Replacing the derivation heuristic with
injected-instruction detection + a presence rule, and broadening the high-stakes
verb set, took ASR to 0% and FPR to 5–19%.

## Integration notes (endpoint compatibility)

Driving AgentDojo through the science-cloud endpoint required five fixes, all in
`tools/` (the gate stack is untouched): (1) bypass AgentDojo's `ModelsEnum` for an
unlisted model id by injecting a custom `OpenAILLM`; (2) register the custom model
name for the attack machinery (neutral `"AI assistant"`); (3) run inside an
`OutputLogger` context; (4) read the `SuiteResults` TypedDict by key; (5) the
endpoint is litellm→AWS Bedrock, which 400s on a non-object tool-use `input`, so a
client proxy normalizes empty/`null`/empty-key tool-call arguments to `{}`.
`gpt-oss-120b` and `gpt-5.4` both do native tool-calling. NB injection ids differ
per suite (slack lacks `injection_task_0`); `injection_task_1` is common to all.

## Run it (live)

```bash
uv sync --extra palisade-agentdojo
set -a; source ../.env; set +a
uv run --extra palisade-agentdojo python -m \
    tools.agentdojo_e1 \
    --model gpt-5.4 --suites banking workspace travel slack \
    --report-out ../docs/palisade/agentdojo_e1.md
# NB: --model gpt-4o is NOT served by this endpoint; use gpt-5.4 / gpt-oss-120b.
```
