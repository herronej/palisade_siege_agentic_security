<!-- Generated report. Source module: tools.siege_ifc_coverage. Regenerate with `uv run python -m tools.siege_ifc_coverage`. -->

# SIEGE vs runnable IFC/guardrail baselines: sink-coverage money-plot (W4.3)

A defense cannot close a hard win at a sink its design does not define. This crosses every undefended SIEGE hard win, grouped by the sink it reaches, against the published enforcement surface of the IFC/guardrail baselines. Hard-win counts are scored from the corpus undefended (seed 42); sink modeling is a checkable property of each baseline's published mechanism.

**Headline: 10 of 37 undefended hard wins reach a sink no runnable IFC baseline models** -- the HPC scheduler submission field and the scientific-citation egress. They fall outside the IFC comparison surface *by construction* (they pass without being evaded); PALISADE's G5 and G6 predicates are what close them.

## Sink types and who models them

| Sink type | SIEGE classes | hard wins | in AgentDojo? | modeled by an IFC baseline? |
|---|---|---|---|---|
| HPC scheduler submission field | `b5_11` | 5 | **no** | **no (by construction)** |
| scientific-citation / reference-value egress | `b3_4` | 5 | **no** | **no (by construction)** |
| cross-boundary tool-call dataflow to a write/exfil sink | `xc_1`, `xc_4`, `b3_9` | 12 | yes | yes |
| untrusted code into a sandbox exec sink | `b4_9` | 5 | yes | yes |
| forged instruction in an uploaded artifact -> tool sink | `b1_9` | 5 | yes | yes |
| forged prior-approval / trust context -> write sink | `xc_5` | 5 | **no** | yes |

## Coverage matrix (does the baseline's design model the sink?)

| Baseline | family | hpc_scheduler | science_egress | cross_boundary_write | code_exec | attached_content | forged_context |
|---|---|---|---|---|---|---|---|
| **PALISADE** | capability bound + contracts | closes | closes | closes | closes | closes | closes |
| CaMeL | capability/IFC (dataflow) | — | — | models | models | models | models |
| FIDES | capability/IFC (deterministic labels) | — | — | models | models | models | models |
| Progent | least-privilege (tool-call policy DSL) | — | — | — | — | — | — |
| LlamaFirewall / Prompt-Guard | detection-only | text-only | text-only | text-only | text-only | text-only | text-only |
| Denylist (this work's reimpl.) | detection-only | text-only | text-only | text-only | text-only | text-only | text-only |

_`models` = the published design defines a check at this sink type; `—` = out of scope for that design; `text-only` = a content detector with no sink model (may still flag on surface text). PALISADE `closes` every row because G1–G6 define a predicate at each sink; its residual (the xc_1 propagation gap, W2) is within-scope, not a coverage gap._

## Sinks outside every IFC baseline (by construction)

- **HPC scheduler submission field** (`b5_11`, 5 hard wins): a tainted value sets a submit_hpc_job field (partition/qos/reservation/dependency/output_path); the submit tool is legitimately needed, so a tool-level allow/deny cannot close it -- only a field-level provenance read at the scheduler sink does (PALISADE G5).
- **scientific-citation / reference-value egress** (`b3_4`, 5 hard wins): a forged citation reaches the answer (the G6 grounding sink); no baseline environment defines a citation-provenance or reference-value contract sink.

## Per-baseline enforcement surface

| Baseline | family | cite | runnable here | closes (of the SIEGE sinks) |
|---|---|---|---|---|
| CaMeL | capability/IFC (dataflow) | Debenedetti et al. 2025 | external repo + served base model (not wired; see agentdojo_e1 for the runnable slot) | 27 hard wins |
| FIDES | capability/IFC (deterministic labels) | Debenedetti et al. 2025 | external repo + served base model (not wired) | 27 hard wins |
| Progent | least-privilege (tool-call policy DSL) | Shi et al. 2025 | external repo + served base model (not wired) | 0 hard wins |
| LlamaFirewall / Prompt-Guard | detection-only | Meta 2025 | RUN (real Prompt-Guard-86M; see detector_baselines_w4) | text-only (see W4.1) |
| Denylist (this work's reimpl.) | detection-only | this work | RUN (offline; see detector_baselines_w4) | text-only (see W4.1) |

- **CaMeL** (Debenedetti et al. 2025): tracks value provenance through the agent's tool calls and refuses a sink fed by untrusted data (a forged approval is untrusted data too, so xc_5 is in scope). But its sinks are its environment's: no HPC-scheduler submission field and no citation/reference-value sink exist to enforce at.

- **FIDES** (Debenedetti et al. 2025): deterministic information-flow labels on tool data; same environment scoping as CaMeL -- no scheduler or citation sink.

- **Progent** (Shi et al. 2025): allow/deny at TOOL (and, by hand-authored argument predicate, argument) granularity. But every SIEGE hard win rides a legitimately-needed tool with a distribution-valid poisoned value (an injected qos, a laundered create_file argument, a forged citation), and Progent reads no provenance, so it cannot separate the injected value from the intended one; it lowers soft-win ASR by denying unneeded tools, but closes none of these hard wins by construction.

- **LlamaFirewall / Prompt-Guard** (Meta 2025): scores text; models no sink at all. Any hard win whose text reads benign passes, and at a sensitivity that catches the laundered ones its benign FPR is 67% (W4.1).

- **Denylist (this work's reimpl.)** (this work): 9-pattern surface denylist; models no sink. The 14% baseline the manuscript reports is this reimplementation.
