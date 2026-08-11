"""
SIEGE evaluation harness (PALISADE).

This package is the *runnable shape* every SIEGE
template, correctness-contract test, and RL policy plugs into. It
delivers nothing template-specific on its own -- it is the substrate:

- ``schemas`` -- frozen instance / trace / scorer dataclasses plus
  the JSON-Schema files they serialize to. Frozen so template
  authoring doesn't churn the format.
- ``instance_loader`` -- load instance YAML/JSON off disk into
  validated ``Instance`` objects.
- ``turn_buffer`` -- the in-process turn buffer threading values across
  the sessions of one episode. There is **no** persistent cross-session
  memory store on the default path; the durable SQLite/Redis backends
  (and the MINJA/MemoryGraft sequences) are deferred (out of scope for
  this release) under ``siege/deferred/``.
- ``session_runner`` -- the ``SessionRunner`` that threads a
  deterministic session-id through the turn buffer and runs each
  session's turns through the *real* PALISADE gate stack. Extends
  the single-turn ``ProjectAgent.run_stream`` into a multi-session
  episode.
- ``trace_recorder`` -- records per-action gate decisions and
  capability-registry snapshots into a ``Trace`` the scorer (and the
  RL reward) consume.
- ``scorer`` -- the BU / UA / ASR scorer: a per-class programmatic
  check with an LLM-judge fallback, emitting per-(boundary, attack-class)
  cells.
- ``ablation_matrix`` -- the 6-configuration B4-first cumulative ablation
  (baseline -> +G4 -> +G4+G3 -> +G4+G3+G1 -> +G4+G3+G1+G5 ->
  full PALISADE).

The shared CLI lives at
``siege.eval.siege_runner`` and is reachable
as ``python -m siege.eval --gate siege``.
"""

from siege.ablation_matrix import (
    ALL_CONFIGS,
    AUGMENTED_CONFIGS,
    CUMULATIVE_CONFIGS,
    AblationConfig,
    GateLayer,
    GateStack,
    build_gate_stack,
)
from siege.instance_loader import (
    bundled_instances_dir,
    load_instance,
    load_instances,
)
from siege.schemas import (
    Action,
    ActionKind,
    Instance,
    Session,
    SuccessCriterion,
    Turn,
)
from siege.corpus_annotations import (
    AUTH_PRECONDITION,
    CROSS_TENANT_TEMPLATES,
    DEFENSE_TIER,
    MULTI_TENANT_PROFILE_NOTE,
    REALISM_NOTE,
    defense_tier,
    is_cross_tenant,
    realism_note,
    tenancy_note,
)
from siege.report_suites import (
    SUITE_LABELS,
    SUITE_ORDER,
    ReportSuite,
    suite_for,
)
from siege.scorer import (
    METRIC_DEFINITIONS,
    Cell,
    InstanceScore,
    aggregate_cells,
    score_trace,
)
from siege.live_session_runner import (
    AgentDriver,
    AgentTurn,
    LiveSessionRunner,
    ProjectAgentDriver,
    ScriptedAgentDriver,
    build_live_instance,
)
from siege.session_runner import SessionRunner
from siege.trace_recorder import (
    ActionRecord,
    Trace,
    TraceRecorder,
)
from siege.turn_buffer import (
    InProcessTurnBuffer,
    MemoryRecord,
    MemoryStore,
)

__all__ = [
    "AblationConfig",
    "Action",
    "ActionKind",
    "ActionRecord",
    "ALL_CONFIGS",
    "AUGMENTED_CONFIGS",
    "AUTH_PRECONDITION",
    "CROSS_TENANT_TEMPLATES",
    "CUMULATIVE_CONFIGS",
    "Cell",
    "DEFENSE_TIER",
    "GateLayer",
    "MULTI_TENANT_PROFILE_NOTE",
    "REALISM_NOTE",
    "ReportSuite",
    "SUITE_LABELS",
    "SUITE_ORDER",
    "GateStack",
    "InProcessTurnBuffer",
    "Instance",
    "InstanceScore",
    "METRIC_DEFINITIONS",
    "MemoryRecord",
    "MemoryStore",
    "Session",
    "SessionRunner",
    "SuccessCriterion",
    "Trace",
    "TraceRecorder",
    "Turn",
    # WI17 live-agent adapter
    "LiveSessionRunner",
    "AgentDriver",
    "AgentTurn",
    "ScriptedAgentDriver",
    "ProjectAgentDriver",
    "build_live_instance",
    "aggregate_cells",
    "build_gate_stack",
    "bundled_instances_dir",
    "load_instance",
    "load_instances",
    "defense_tier",
    "is_cross_tenant",
    "realism_note",
    "score_trace",
    "suite_for",
    "tenancy_note",
]
