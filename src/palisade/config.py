"""
PALISADE sidecar configuration.

This module defines `PalisadeSettings`, a Pydantic model holding every
toggle and tunable for the PALISADE security sidecar. It is mounted on
the top-level `Settings` as a nested field, so the entire configuration
is reachable as `settings.palisade.*` and overridable from the
environment with `PALISADE_<FIELD>=...` (double
underscore acts as the nested delimiter -- see `Settings.model_config`
in `vista_backend.config`).

By default every flag here is False, including the master `enabled`
flag. The sidecar's runtime hooks are written so that with `enabled=False`
they are byte-identical no-ops; this is the contract that PALISADE is
genuinely modular (see the flag-off regression test).

The fields are grouped as follows:

- `enabled` and `g{N}_enabled` are the per-gate toggles used everywhere.
- `quarantine_*` configures the Q-LLM.
- `initial_trust`, `tier_transition_thresholds`, `sticky_high_stakes`
  configure the trust scorer.
- `flowcept_*` configures the provenance bus.
- `semgrep_*` configures G4.
- `contracts_dir` configures the contract library.

Later phases may add fields here; the convention is that every field
defaults to a value that keeps the system off / minimal so that adding
a new field never changes behavior of existing deployments.
"""

import logging

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class PalisadeSettings(BaseSettings):
    """
    All PALISADE configuration.

    Read from the environment as `PALISADE_<FIELD>=...`, or mounted as a
    nested field on a host application's own settings object -- the sidecar
    only ever reads the instance it is handed at construction.
    """

    model_config = SettingsConfigDict(
        env_prefix="PALISADE_",
        env_file=".env",
        extra="ignore",
    )

    # -----------------------------------------------------------------
    # Master and per-gate toggles
    # -----------------------------------------------------------------

    enabled: bool = False
    """
    Master toggle. When False, `PalisadeSidecar` constructs but does
    not invoke any gate logic, and its `process_tool_call` is a strict
    pass-through. The Q-LLM agent is not instantiated. The flag-off path
    is verified byte-identical to baseline VISTA by the regression test.
    """

    g1_enabled: bool = False
    """Prompt Gate. Enables prompt-side checks in `run_stream` and the
    tier-banner system prompt."""

    g2_enabled: bool = False
    """Tool Gate. Enables tool-call gating inside `process_tool_call`.
    """

    g2_fail_closed_unlabeled: bool = False
    """
    When True, G2's high-stakes guard denies a sink call whose argument
    resolves to *no* capability label, in any session holding a live
    untrusted value from a source outside the trusted set (user, system,
    operator). When False (the default), such an argument is admitted.

    The taint rule decides on a label that is present. An argument carrying
    no label at all is a different case, and admitting it is a fail-open on
    the missing label rather than a policy decision about it -- the failure
    mode behind the `xc_1` cross-boundary chains, whose `create_file` sink
    argument shares no text with the poisoned read and so resolves to
    nothing. `CapabilityRegistry.propagate` already refuses the analogous
    join rather than defaulting an unregistered source to untainted; this
    flag extends the same posture to the sink.

    Default off because the rule is an over-approximation, not a free win.
    Labels are resolved by content, so a freshly authored argument -- which
    is most of what a planning model emits -- resolves to nothing whether
    the session is benign or not. Once any untrusted value is live, the
    denial therefore lands on benign high-stakes calls too, and how often
    depends entirely on the workload's ingest pattern. Measure the benign
    cost on your own controls before enabling; do not assume the corpus
    figure transfers. Denials are SEV3 (recorded, no session floor), so the
    cost shows up as refusals rather than as lockouts.
    """

    g2_minimize_via_retry: bool = False
    """
    When True, G2's Minimize layer raises `ModelRetry` after stripping
    sensitive content from outgoing tool arguments, teaching the model
    what was removed so it can re-issue a clean call. When False (the
    default), Minimize silently substitutes the rewritten arguments and
    the call proceeds. Default off to preserve current behavior and
    avoid an unbounded retry-loop budget impact; flip on after 
    evaluation measures that cost. (R3)."""

    g3_enabled: bool = False
    """RAG / Memory Gate. Enables sanitization of `rag_search` returns.
    """

    g3_query_injection_enabled: bool = True
    """
    Enable the G3 fast-tier query-injection regex (DAN-family,
    instruction-override, role-impersonation patterns). Cheap and
    deterministic; safe-on by default. Set to False only when the
    deployment has measured a high false-positive rate on its own
    benign-query workload and is OK losing the defense. The regex
    patterns are curated in `gates/g3_rag.py:DEFAULT_QUERY_INJECTION_PATTERNS`.
    """

    g3_anomaly_z_threshold: float = 3.0
    """
    Per-batch z-score threshold for the G3 embedding-cluster anomaly
    detector. A retrieved chunk is flagged when its mean
    pairwise distance to other chunks in the same retrieval batch has
    `|z| > g3_anomaly_z_threshold`. The default 3.0 corresponds to
    roughly one outlier per 370 clean batches under normality and is a
    deliberately conservative starting point -- per the work item, this
    detector catches naive embedding attacks but is bypassable by
    attackers crafting natural-norm embeddings, so the false-positive
    cost matters more than the false-negative cost. Each deployment
    should calibrate against its own RAG corpus and lower the threshold
    only after measuring the benign-workload FPR.
    """

    g3_anomaly_min_batch_size: int = 3
    """
    Minimum retrieval-batch size for the G3 embedding-cluster anomaly
    detector to run at all. Below 3 chunks there is no meaningful
    pairwise-distance distribution (a 2-chunk batch yields a single
    distance, and z-scores require std > 0). The detector returns a
    benign "batch too small" result for batches below this threshold;
    this is not a false-negative for PALISADE's overall posture
    because the deterministic G3 fast-tier checks still run on those
    batches.
    """

    g3_hybrid_retrieval: bool = False
    """
    Enable the G3 hybrid (BM25 + vector) retrieval defense
    (Semantic Chameleon arXiv 2603.18034). When True, the PALISADE
    backend transparently injects `hybrid=True` into every
    `rag_search` tool call so the MCP server returns merged
    BM25+vector results. Defeats gradient-guided embedding-poisoning
    attacks (PoisonedRAG, AgentPoison) by demoting chunks that score
    only on the vector modality. Default False to keep the legacy
    vector-only behavior byte-identical for deployments that
    haven't built a BM25 corpus yet.

    A deployment that flips this on without first running
    `build_rag.py` against the relevant KB will see the MCP server
    log a warning and degrade to vector-only retrieval per-KB. The
    setting is therefore safe to enable preemptively; it's a no-op
    on KBs that lack a `bm25_corpus.json`.
    """

    g3_hybrid_alpha: float = 0.5
    """
    Weight on the vector modality when `g3_hybrid_retrieval=True`.
    1.0 = vector-only (the legacy behavior), 0.0 = BM25-only (not
    recommended in production), 0.5 = equal weight (the Semantic
    Chameleon paper's reported configuration). Palisade's backend
    forwards this verbatim to the MCP server's `rag_search`
    `alpha` parameter; the merge math lives there.

    Outside `[0.0, 1.0]` is rejected by the MCP server's argument
    validator -- the pydantic validation on the call site catches
    misconfigurations at boot rather than at runtime.
    """

    g4_enabled: bool = False
    """Sandbox / Code Gate (B4). Enables the merged sandbox-code gate on
    ``run_bash`` / ``create_file``: Tier-0 deterministic path-confinement +
    execution-IOC checks (always-on, no dependency), the optional Semgrep
    tier (see ``semgrep_enabled``), and the Q-LLM code-intent slow tier.
    Subsumes the former G7 sandbox gate."""

    g5_enabled: bool = False
    """HPC Job Gate. Enables gating of `submit_hpc_job` and related
    HPC tool calls."""

    g6_enabled: bool = False
    """Egress Gate (output-side). Enables citation/provenance binding and
    scientific-value contract checks on the agent's final answer."""

    g6_block_mode: bool = False
    """When True, a G6 egress violation (ungrounded/misattributed citation or
    out-of-bounds scientific value) raises ``ModelRetry`` to force a regenerate
    instead of annotating the answer with a warning. Default False (annotate):
    a single bad citation shouldn't nuke an otherwise-correct response. Flip on
    for fabricated-citation / value-sensitive deployments after measuring the
    retry-loop cost."""

    g7_enabled: bool = False
    """DEPRECATED. The former G7 sandbox gate is merged into G4 (the
    sandbox/code boundary is one gate). Kept for backward compatibility:
    if set without ``g4_enabled``, the ``_fold_deprecated_g7`` validator
    enables G4 (with a warning) so the deterministic sandbox checks are
    never silently dropped on upgrade. Prefer ``g4_enabled``."""

    # -----------------------------------------------------------------
    # Quarantine (Q-LLM) configuration
    # -----------------------------------------------------------------

    quarantine_enabled: bool = False
    """
    Master slow-tier toggle. When False, every gate runs fast-only
    regardless of which gates are enabled. This is the cheapest way to
    smoke-test PALISADE in production: enable the deterministic
    checks before paying the Q-LLM serving cost.
    """

    quarantine_model: str = "openai:gpt-oss-120b"
    """
    PydanticAI model spec for the Q-LLM (see the `pydantic-ai` skill
    for the `provider:model-id` format). Should be an *unfiltered*,
    in-deployment endpoint: unfiltered so it does not safety-refuse
    adversarial content (a filtered model default-denies the slow tier
    and false-positives on dual-use-adjacent benign science), and
    in-deployment so CUI content never leaves. The default targets the
    in-deployment ``gpt-oss-120b`` endpoint (set ``OPENAI_API_KEY`` +
    base URL); a locally served model (Ollama, vLLM) is an alternative
    for the same posture.
    """

    quarantine_temperature: float = 0.0
    """
    Sampling temperature for the Q-LLM's *primary* (single-sample) verdict. The
    slow tier is a block-deciding classifier, so its default verdict should be
    stable and reproducible -- pinned to 0.0 rather than an accidental provider
    default. Self-consistency (``quarantine_self_consistency_samples >= 2``) uses
    a separate nonzero sampling temperature so its votes stay diverse; see
    ``quarantine.SELF_CONSISTENCY_SAMPLING_TEMPERATURE``. Note the offline
    ablation harness builds its Q-LLM agents without this override (provider
    default) to preserve its multi-draw pooling variance, so this field does not
    move the reported ablation numbers.
    """

    judge_enabled: bool = True
    """
    Run the detection-only LLM judge as a stage of the slow tier
    (``Gate._apply_judge_check``). Part of the slow tier rather than an
    independent layer: it is consulted only where a quarantine agent is already
    wired, so a fast-tier-only deployment pays nothing for it.

    It is strictly additive. A MALICIOUS verdict denies at SEV2; a BENIGN verdict
    does nothing at all -- it may not clear taint, relax a label, or license a
    sink. A transport failure is not a verdict and leaves the decision untouched,
    which is deliberate: an additive detector that failed closed would turn the
    endpoint's availability into a denial-of-service vector, and the capability
    bound -- which calls nothing -- is what still holds while the judge is down.

    Set False to ablate it. Note that unlike the rest of the slow tier the judge
    is a *content* decision, so it carries the evasion properties of any
    detector: an adaptive optimizer finds an objective-preserving phrasing it
    admits for roughly half of objectives within eight queries. It buys
    inspection depth, not capability closure; the label bound is what is
    unmoved by that search.
    """

    judge_model: str = "gpt-oss-120b"
    """Model id for the slow-tier judge. Uses the same OpenAI-compatible
    endpoint as the Q-LLM (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``)."""

    quarantine_self_consistency_samples: int = 1
    """
    Number of Q-LLM samples for the slow-tier self-consistency check. Applies to
    every model-backed slow tier -- G1 intent, G2/G3 sanitize, and G4/G5
    code-intent (all five are wired from this one field by the sidecar). With
    ``>= 2`` samples the runners default-deny on disagreement over the discrete
    verdict, drawing at a nonzero temperature so the samples differ. Default 1 to
    keep latency low until the deployment has measured the Q-LLM's per-call
    latency.
    """

    # -----------------------------------------------------------------
    # Trust scorer and tier policy
    # -----------------------------------------------------------------

    initial_trust: float = 1.0
    """Starting trust score for a new session. Range [0.0, 1.0]."""

    tier_transition_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "ELEVATED": 0.7,
            "RESTRICTED": 0.4,
            "TERMINATED": 0.1,
        }
    )
    """
    Per-tier trust thresholds. A session whose trust score falls below
    the value for a tier transitions into that tier. Defaults are
    conservative; per-deployment calibration against a held-out benign
    workload is recommended before tightening.
    """

    sticky_high_stakes: bool = True
    """
    When True (default), high-stakes capabilities -- G6 destructive
    commands, G5 above-ceiling resource requests, G3 CUI-tier corpora
    -- do not auto-recover within a session once denied. Only an
    explicit re-authentication event clears the stickiness. This
    breaks the probe-then-strike oscillation-attack primitive by
    design. Set False only when running the oscillation-attack
    benchmark.
    """

    # -----------------------------------------------------------------
    # Provenance / Flowcept integration -- logging,
    # Flowcept shipping
    # -----------------------------------------------------------------

    flowcept_enabled: bool = False
    """
    When True, ship provenance events to Flowcept. When False,
    `ProvenanceEmitter` still serializes events but writes them to a
    log file rather than to the Flowcept broker. False is the safe
    default for development.
    """

    flowcept_endpoint: str | None = None
    """
    Flowcept broker (MQ) endpoint, e.g. ``redis://broker.local:6379``.
    Required when `flowcept_enabled=True`; `ProvenanceEmitter` raises at
    construction time if Flowcept is enabled without an endpoint (fail
    fast).

    The endpoint is applied to Flowcept's own configuration via the
    documented environment overrides it reads at startup (`MQ_URI` /
    `KVDB_URI`, plus `MQ_ENABLED`), so a deployment that does not ship a
    `~/.flowcept/settings.yaml` still routes provenance to this broker.
    Any Flowcept env the operator has already exported wins (the sink
    only fills in unset values).
    """

    flowcept_start_persistence: bool = True
    """
    Whether the embedded Flowcept controller also runs the DB-persistence
    consumer (`start_persistence`).

    True (default): this process publishes to the MQ *and* persists to the
    configured Flowcept DB (e.g. MongoDB) -- the self-contained
    Redis+Mongo deployment works out of the box. Set
    `MONGO_URI`/`MONGO_HOST` in the environment (or a Flowcept settings
    file) to point at the DB.

    False: producer-only. PALISADE publishes provenance to the MQ and a
    *separate* Flowcept persistence process (the `DocumentInserter`
    consumer) is responsible for writing it to the DB. Use this when the
    deployment already runs a shared Flowcept consumer so multiple
    producers don't each open a DB writer.
    """

    provenance_log_path: str | None = None
    """
    Filesystem path for the JSONL provenance log used by
    `ProvenanceEmitter` when `flowcept_enabled=False`. When None
    (default), provenance events are emitted via the module logger
    instead of a dedicated file -- ops can still capture them by
    routing the `palisade.provenance` logger to the
    desired sink. When set, the emitter appends one JSON line per
    event to this path and flushes after each write so audit data
    survives a SIGKILL. will layer AU-9 tamper-evidence
    on top of this same path; Flowcept broker emission
    is enabled via `flowcept_enabled`.
    """

    # -----------------------------------------------------------------
    # G4 code scanning
    # -----------------------------------------------------------------

    semgrep_enabled: bool = False
    """
    Enable the Semgrep tier (Tier 1) of G4's code scan. Requires the
    optional `semgrep` dependency (install with
    `pip install vista-backend[palisade-g4]`). When False, G4 still runs
    its always-on Tier-0 deterministic sandbox checks (path confinement +
    execution IOCs) and the Q-LLM code-intent slow tier; only the Semgrep
    layer is skipped. When True but the CLI is missing, G4 degrades to
    Tier-0 rather than fail-closed (and logs a warning at startup).
    """

    semgrep_config: str | None = None
    """
    Extra Semgrep ruleset spec layered on top of the always-loaded
    PALISADE-bundled rules (`palisade/contracts/semgrep/`). Default is
    None -- bundled rules only, so the scan runs fully offline (the bundled
    set is the curated PALISADE defense; no semgrep.dev fetch). Set this to
    a registry pack (e.g. `"p/security-audit"`) or a local path to add
    breadth when network egress is available -- a registry spec makes the
    scan network-dependent and is unsuitable for air-gapped / CUI deployments.
    """

    # -----------------------------------------------------------------
    # G5 HPC job gating
    # -----------------------------------------------------------------

    g5_allocation_policy_path: str | None = None
    """
    Filesystem path to the operator-supplied G5 allocation policy
    (`g5_allocation_policy.json`: authorized allocations + caps, mining
    denylist, host allow-list). When None (the default), the sidecar
    looks for `<contracts_dir>/g5_allocation_policy.json`. A missing,
    unreadable, malformed, or wrong-version file falls back to the
    bundled defaults with a WARNING -- the same posture as
    `g3_kb_policy.json` and `palisade_tool_manifest.json`.
    """

    g5_resource_ceilings: dict[str, dict] = Field(default_factory=dict)
    """
    Per-allocation resource caps (node / time / GPU / partition),
    expressed without authoring a full policy JSON file. Keyed by
    allocation (account) name; each value carries `max_nodes`,
    `max_time_seconds`, `max_gpus`, and `permitted_partitions`. These
    apply only when the operator policy file does not itself supply an
    `allocations` map -- the file wins when both are present.

    Defaults to empty, which means *no* settings-driven ceiling
    enforcement: consistent with the module-wide convention that every
    field defaults to keep the system off / minimal, and with the
    bundled `AllocationPolicy` carrying no allocations (so submissions
    flow until an operator opts in to allocation enforcement).
    """

    g5_binary_denylist: frozenset[str] = frozenset()
    """
    Additional mining-binary IOCs to deny at G5, unioned at startup with
    the gate's bundled denylist (`DEFAULT_MINING_BINARY_DENYLIST`, >= 10
    public-feed entries) and the policy file's `binary_denylist`. The
    bundled list is always in effect; this setting (and the policy file)
    can only ever tighten coverage. Defaults to empty.
    """

    g5_chained_job_dag_enabled: bool = True
    """
    Toggle for the G5 slow-tier chained-job dependency DAG walker. When
    True (the default), the slow tier walks `--dependency=afterok/...`
    references and re-applies the fast-tier checks to each dependent job
    (catching the cross-boundary chain attack). The walk is still gated
    by the master slow-tier toggle (`quarantine_enabled`) and `g5_enabled`,
    so this only matters when those are on; set False to disable just the
    DAG walk while keeping the rest of G5's slow tier.
    """

    g5_require_submit_approval: bool = False
    """
    Whether ``submit_hpc_job`` is held for *human* approval after G5's policy
    tiers pass. Default False: this deployment auto-approves tool calls (there
    is no human approver), so the elicitation round-trip is a no-op -- the job
    is gated **autonomously** by G5's deterministic fast tier + slow tier,
    which deny on a policy violation regardless. Set True only for a
    deployment with a real human-in-the-loop approver wired to
    ``PalisadeApprovalCapability``.
    """

    # -----------------------------------------------------------------
    # Retrieval corpora (G3 corpus-integrity pinning)
    # -----------------------------------------------------------------

    knowledge_bases_dir: str = "knowledge_bases"
    """
    Directory holding the served retrieval corpora, one subdirectory per
    knowledge-base slug. G3 recomputes each pinned corpus hash against the
    store found here and denies ``rag_search`` when a corpus no longer
    matches its pin (see ``palisade.corpus_integrity`` and the
    ``pin_corpus`` operator CLI).

    In the reference deployment this is supplied by the host application's
    own settings; standalone it defaults to ``knowledge_bases/`` beside the
    checkout. A missing directory disables the live check rather than
    failing sidecar construction.
    """

    # -----------------------------------------------------------------
    # Contract library
    # -----------------------------------------------------------------

    contracts_dir: str = "palisade_contracts"
    """
    Filesystem path to the scientist-authored contract library. The
    runtime loads contracts from this directory at startup; the
    directory is intended to live in a separate repository so domain
    scientists can contribute via PRs without touching PALISADE
    code. When the directory is empty or missing, PALISADE runs
    with an empty contract registry and gates that consult the
    registry behave as if no contract applies.
    """

    @model_validator(mode="after")
    def _fold_deprecated_g7(self) -> "PalisadeSettings":
        """Fold the deprecated ``g7_enabled`` flag into ``g4_enabled``.

        The former G7 sandbox gate is merged into G4. A deployment that
        still sets ``g7_enabled`` (without ``g4_enabled``) would otherwise
        silently lose the deterministic sandbox checks, since nothing reads
        ``g7_enabled`` anymore. Enabling G4 here preserves that floor and
        warns so the operator migrates the flag.
        """
        if self.g7_enabled and not self.g4_enabled:
            logger.warning(
                "PALISADE_G7_ENABLED is deprecated: the G7 "
                "sandbox gate is merged into G4. Enabling g4_enabled so the "
                "deterministic sandbox checks still run; set g4_enabled "
                "directly and drop g7_enabled."
            )
            self.g4_enabled = True
        return self
