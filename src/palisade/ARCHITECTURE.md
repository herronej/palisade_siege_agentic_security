# PALISADE architecture

PALISADE is the security-mediation **sidecar** that wraps `ProjectAgent`'s
tool-call surface with a chain of fast-then-slow **gates**. It is a strictly
modular addition: with the master flag off (the default), `build_capabilities()`
returns an empty list and the agent is byte-identical to baseline VISTA.

The gates are exposed to the agent as PydanticAI `AbstractCapability` subclasses
(`G1PromptCapability`…`G6EgressCapability`) that register lifecycle **hooks**.
Each gate decides via a deterministic **fast tier**, an optional Q-LLM **slow
tier**, and a model-free domain **contract** check, then routes any incident
through a shared adaptive layer (trust scoring + incident playbook) and emits
provenance.

> Gate numbering note: the former **G7 sandbox gate is folded into G4** (one
> sandbox/code boundary). `g7_enabled` is deprecated and aliases `g4_enabled`.

---

## 1. System context & integration

How the sidecar plugs into the agent. The Q-LLM agents and MCP toolsets are
threaded in at build time; the master flag gates the entire thing.

```mermaid
flowchart TB
    User(["User turn"]) --> PA["ProjectAgent.run_stream<br/>(backend)"]
    PA --> SC["PalisadeSidecar<br/>one per session"]

    SC -->|"master flag OFF (default)"| NOOP["empty capability list<br/>→ byte-identical baseline VISTA"]
    SC -->|"master flag ON"| BUILD["build_capabilities()"]

    QLLM["Q-LLM agents (unfiltered, in-deployment)<br/>quarantine · intent · code-intent"] -.->|"wired into slow tiers"| BUILD
    CONLIB["Contract library<br/>salts · spectroscopy · hpc + operator *.py"] -.-> BUILD

    BUILD --> CAPS["capabilities = [G1, G2, G3, G4, G5, G6, Approval]"]
    CAPS --> AGENT["PydanticAI Agent(capabilities=…)"]
    MCP["MCP toolsets<br/>vista_mcp_server: HPC · RAG · display_file<br/>dev_mcp_server: run_bash · create_file · view"] --> AGENT
    AGENT --> LOOP(["agent run · tool-call loop<br/>(gates mediate via hooks)"])
```

---

## 2. Runtime hook lifecycle → gate actions

The spine is PydanticAI's per-turn hook lifecycle. Each hook fires the gate(s)
that registered on it. A fast-tier deny raises `SkipToolExecution` /
`PalisadeDeny`; the deny message becomes the tool result (or refuses the turn).

```mermaid
flowchart LR
    subgraph LIFE["PydanticAI hook lifecycle (per agent turn)"]
        direction TB
        H1["before_run"]
        H2["before_model_request"]
        LLM(["LLM ↻ tool-call loop"])
        H3["prepare_tools"]
        H4["before_tool_validate"]
        H5["before_tool_execute"]
        TOOL(["MCP tool exec"])
        H6["after_tool_execute"]
        H7["after_output_process"]
        H1 --> H2 --> LLM --> H3 --> H4 --> H5 --> TOOL --> H6 --> H7
        LLM -.->|"next step"| H3
    end

    H1 --> A1["G1 refuse turn if session terminated<br/>G6 reset per-turn citation ledger"]
    H2 --> A2["G1 inject trust-tier banner (system prompt)"]
    H3 --> A3["G2 allow-list (fnmatch) + ETDI descriptor-hash<br/>→ drop offending tools before model sees them"]
    H5 --> A5["G2 capability-taint walk + high-stakes guard + Minimize<br/>G4 run_bash / create_file scan<br/>G5 submit_hpc_job policy + human approval"]
    H6 --> A6["G2 Sanitize tool returns (+ tag)<br/>G3 rag_search sanitize / tag / anomaly<br/>G6 parse DOI/arXiv citations into ledger"]
    H7 --> A7["G6 citation grounding + scientific-claim integrity"]

    A1 & A2 & A3 & A5 & A6 & A7 --> ADAPT["Shared session state (§4)"]
```

---

## 3. Per-gate decision model (fast → slow → contracts)

Every gate runs the same three-stage cascade. The fast tier is always on; the
slow tier runs only when a `quarantine_agent` is wired (`quarantine_enabled`);
the contract check needs no model and runs whenever the gate is enabled.

```mermaid
flowchart TB
    IN["prompt / tool call / retrieval / output"] --> FAST

    FAST["FAST tier · check_fast()<br/>deterministic, always-on<br/>regex · policy · hashing · path rules"]
    FAST -->|deny| DENY
    FAST -->|allow| SLOW

    SLOW["SLOW tier · check_slow()<br/>Q-LLM quarantine model<br/>only if quarantine_agent wired"]
    SLOW --> CON

    CON["CONTRACTS · enforce()<br/>domain claims, model-free<br/>whenever gate enabled"]
    CON -->|violation| DENY
    CON -->|clean| ALLOW["GateDecision(allow=True)<br/>(may carry rewritten args / sanitized result / taint tag)"]

    DENY["GateDecision(allow=False, incident_level=SEVn)"] --> INC["IncidentManager.record()"]
    ALLOW --> CLEAN["TrustScorer.record_clean_call(gate)"]
```

### Gate reference

| Gate | Boundary | Fast tier (deterministic) | Slow tier / contracts | Key hooks |
|---|---|---|---|---|
| **G1 Prompt** | B1 prompt injection | jailbreak signatures, PII / CUI markers, upload MIME + size | Q-LLM intent self-consistency; **SEV1 → terminate session** | `before_run`, `before_model_request` |
| **G2 Tool** | B2 tool surface | allow-list (fnmatch), ETDI descriptor hash (rug-pull), JSON-Schema, capability-taint + high-stakes | Minimize (strip args) + Sanitize (returns) | `prepare_tools`, `before_tool_validate`, `before_tool_execute`, `after_tool_execute` |
| **G3 RAG / Memory** | B3 retrieval | query-injection regex, per-chunk tag, embedding-cluster anomaly (z-score), hybrid BM25+vector | Q-LLM chunk intent + domain contracts over claims | `before_tool_execute` (inject hybrid), `after_tool_execute` (`rag_search`) |
| **G4 Sandbox / Code** | B4 code (← G7) | Tier-0 path confinement + execution IOCs; Tier-1 Semgrep (optional) | Q-LLM code-intent | `before_tool_execute` (`run_bash`, `create_file`) |
| **G5 HPC Job** | B5 HPC | allocation policy + resource ceilings, mining-binary denylist, SLURM parse | chained-job DAG walk + HPC contracts; human approval | `before_tool_execute` (`submit_hpc_job`) |
| **G6 Egress** | output integrity | citation grounding (DOI / arXiv), scientific-value claims via contracts | annotate (default) or block (`ModelRetry`) | `before_run`, `after_tool_execute` (`rag_search`), `after_output_process` |

---

## 4. Adaptive layer — trust, incidents, sticky high-stakes

Incidents feed a **per-capability** Bayesian trust scorer (independent `Beta`
posteriors, so a clean stream of G2 calls can't wash out a G3 breach). The
session tier is the **worst** capability tier, clamped by the incident floor.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> NORMAL: initial_trust = 1.0
    NORMAL --> ELEVATED: trust below 0.7  /  SEV2 (+force re-auth)
    ELEVATED --> RESTRICTED: trust below 0.4
    RESTRICTED --> TERMINATED: trust below 0.1
    NORMAL --> TERMINATED: SEV1 (terminate)
    ELEVATED --> NORMAL: clean calls rebuild posterior\n(blocked by sticky floor)
    RESTRICTED --> ELEVATED: clean calls / re-auth

    note right of TERMINATED
        SEV1 is final. G1 refuses every
        later turn. Re-auth does NOT revive.
    end note
    note left of NORMAL
        Re-auth clears sticky high-stakes
        floors + the SEV2 escalation floor.
    end note
```

**Incident playbook (`IncidentManager` → `TrustScorer.notify_incident`):**

| Severity | Log level | Action |
|---|---|---|
| **SEV3** | INFO | scoring only, no tier change |
| **SEV2** | WARNING | elevate session tier + force re-authentication |
| **SEV1** | ERROR | terminate session (every further turn refused) |

**Sticky high-stakes** (`sticky_high_stakes=true`): a high-stakes denial
(G5-above-ceiling, G6-destructive, G3-CUI) ratchets a per-capability floor that
clean calls can't lift — only explicit re-auth clears it. This breaks the
probe-then-strike oscillation primitive.

Shared session state lives on the sidecar, not the LLM context:
`CapabilityRegistry` (taint tags: sensitivity / dual-use / provenance chain),
`TrustScorer`, `IncidentManager`.

---

## 5. Provenance & control plane

```mermaid
flowchart LR
    GATES["Gate decisions + incidents"] --> PE["ProvenanceEmitter"]
    PE -->|"flowcept_enabled"| FLOW{{"Flowcept broker<br/>(per-session workflow)"}}
    PE -->|"else"| JSONL[("JSONL log / module logger")]
    FLOW -.->|"broker unreachable → degrade"| JSONL

    TS["TrustScorer"] --- API["control plane (mounted only when enabled)"]
    API --> STATE["GET /palisade/state<br/>per-capability scores · sticky · contract coverage"]
    API --> REAUTH["POST /palisade/reauth<br/>clear sticky lock-in"]
```

Supporting utilities (live runtime, outside the gate cascade):
`tool_registry.py` (ETDI schema pinning for G2), `ingestion.py` (upload
deny-list, pre-sandbox), `provenance_manifest.py` (tamper-resistant upload
audit), `quarantine.py` (Q-LLM agent builder).

---

## 6. Offline evaluation & red-teaming

These subsystems are **not in the live request path**. They drive authored
attack sequences through the *real, unmodified* gate stack to measure it.

```mermaid
flowchart TB
    subgraph OFFLINE["Offline only — never in the live agent request path"]
        subgraph SAB["siege · deterministic benchmark harness"]
            CORP[("corpus/<br/>b1 prompt · b3 retrieval · b4 code<br/>b5 HPC · xc cross-boundary · benign")]
            CORP --> IL["instance_loader"] --> SR["session_runner"]
            SR --> TRC["trace_recorder → Trace"]
            AM["ablation_matrix<br/>baseline → +G4 → +G3 → +G1 → +G5 → full"] -.-> SR
            TRC --> SCORE["scorer · BU / UA / ASR metrics"]
        end
        subgraph RT["redteam · adaptive adversary"]
            ENV["RedTeamEnv (gym-style)"] --> ATT["Attacker / Policy"]
            ATT --> OPT["llm_optimizer · embedding_optimizer · manifold"]
            REW["reward (WinKind) · metrics (ASR@budget)"] -.-> ATT
        end
        RT -->|"proposes artifacts"| SR
    end

    SR ==>|"runs each action through"| REAL["REAL gate stack (unmodified)"]
    REAL ==> TRC
```

| Attack class | Boundary |
|---|---|
| `b1_*` | B1 prompt injection (direct, crescendo, obfuscation, suffix, dual-use framing, …) |
| `b3_*` | B3 retrieval (corpus / embedding / data-value poisoning, citation forgery, tool-return injection, …) |
| `b4_*` | B4 code (malicious code, network exfil, typo-squat, path traversal, …) |
| `b5_*` | B5 HPC (mining, credential exfil, SLURM smuggling, DAG escalation, …) |
| `xc_*` | cross-boundary chains (taint laundering, runtime supply chain, history forgery, …) |
| `benign_workload` | non-attack control |

---

## Configuration quick-reference

All under `Settings.palisade`, overridable via
`PALISADE_<FIELD>` (double underscore = nested). Everything
defaults **off / minimal**.

| Flag | Default | Effect |
|---|---|---|
| `ENABLED` | `false` | master toggle; off → every hook is a no-op |
| `G1_ENABLED` … `G6_ENABLED` | `false` | per-gate toggles |
| `QUARANTINE_ENABLED` | `false` | master slow-tier toggle (Q-LLM) |
| `QUARANTINE_MODEL` | `openai:gpt-oss-120b` | Q-LLM model spec; an unfiltered, in-deployment endpoint (CUI never leaves) |
| `STICKY_HIGH_STAKES` | `true` | high-stakes denials don't auto-recover within a session |
| `SEMGREP_ENABLED` | `false` | G4 Tier-1 Semgrep scan |
| `CONTRACTS_DIR` | `../palisade_contracts` | operator contract directory |
| `FLOWCEPT_ENABLED` / `FLOWCEPT_ENDPOINT` | `false` / `null` | ship provenance to a Flowcept broker |
