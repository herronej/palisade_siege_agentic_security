"""
Provenance emitter for PALISADE.

"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any
import uuid

from palisade.capabilities import CapabilityTag
from palisade.config import PalisadeSettings
from palisade.gates.base import GateContext, GateDecision
from palisade.incidents import IncidentRecord


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# Event-type constants
# -----------------------------------------------------------------
#
# The two event types emits. Exposed as module-level
# constants so consumers (and the JSONL-tail-reader tests) can
# filter without hard-coding the strings. Flowcept
# topic/routing-key conventions will likely reuse these.

EVENT_GATE_DECISION = "gate_decision"
EVENT_INCIDENT = "incident"


# -----------------------------------------------------------------
# Broker sink
# -----------------------------------------------------------------


class FlowceptSink:
    """
    Ships `ProvenanceEvent`s to a Flowcept broker.

    `flowcept` is an **optional dependency**, imported lazily on the first
    emit so that importing PALISADE -- and constructing a
    `ProvenanceEmitter` -- never requires it. Each provenance event is
    published as a Flowcept *task* grouped under a per-session *workflow*
    (the workflow id is the emitter's `session_id`), so an auditor can
    pull every gate decision and incident for a session as one trace.

    Ingestion uses Flowcept's **documented capture API**: a `FlowceptTask`
    finalized with `.end(...)` publishes the event to the message queue
    (MQ), which Flowcept's `DocumentInserter` consumer persists to the DB
    (MongoDB / LMDB). This is the supported path -- it goes through the
    MQ/pub-sub pipeline Flowcept is built around rather than poking the DB
    directly -- so version skew is unlikely and what we write is what
    Flowcept's own consumers expect. Any failure surfaces as an exception
    the `ProvenanceEmitter` catches and degrades from: provenance must
    never crash the agent run.

    Flowcept resolves its MQ/DB connection from environment variables (or
    a `~/.flowcept/settings.yaml`); see `_apply_endpoint_env` for how the
    operator-supplied endpoint is mapped onto those documented knobs.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        session_id: str,
        workflow_name: str = "palisade",
        start_persistence: bool = True,
    ) -> None:
        self._endpoint = endpoint
        self._session_id = session_id
        self._workflow_name = workflow_name
        self._start_persistence = start_persistence
        self._flowcept: Any | None = None
        self._task_cls: Any | None = None
        self._started = False

    def _apply_endpoint_env(self) -> None:
        """Map the operator endpoint onto Flowcept's documented MQ env knobs.

        Flowcept reads its MQ routing from environment variables (which
        override its settings file): `MQ_URI` is the full connection URI,
        `MQ_ENABLED` toggles publishing, and Redis doubles as the KV store
        (`KVDB_URI`). We fill these in from `flowcept_endpoint` only when
        the operator has not already exported them, so an explicit Flowcept
        configuration always wins. Must run before `import flowcept`, which
        snapshots this configuration at import time.

        Deliberately scoped to the MQ. We do *not* force `MONGO_ENABLED`
        here: enabling a persistent DB via env conflicts with Flowcept's
        default `db_flush_mode: offline` settings file (Flowcept's own
        `validate_config` rejects "offline flush + persistent DB enabled").
        DB persistence is a Flowcept-deployment concern -- govern it via a
        Flowcept settings file (e.g. `flowcept --config-profile full-online`)
        or the `MONGO_*` env. PALISADE's contract is only to publish to
        the broker; `flowcept_start_persistence` then decides whether this
        process also runs the in-process persistence consumer.
        """
        if not self._endpoint:
            return
        # `MQ_URI` overrides host/port and accepts the redis://host:port the
        # operator configures as the broker endpoint.
        os.environ.setdefault("MQ_URI", self._endpoint)
        # Redis serves as Flowcept's KV store too; point it at the same
        # broker unless the deployment split them deliberately.
        os.environ.setdefault("KVDB_URI", self._endpoint)
        # The operator opted into Flowcept by enabling it in PALISADE;
        # MQ publishing defaults to *off* in a fresh Flowcept settings file,
        # so enable it explicitly (operator env still wins). MQ-enabled is
        # safe even under offline flush -- only DB-enabled conflicts there.
        os.environ.setdefault("MQ_ENABLED", "true")

    def _ensure_started(self) -> None:
        """Lazily import flowcept and start the controller once."""
        if self._started:
            return
        # Apply routing env *before* importing flowcept -- it reads its
        # MQ/DB configuration at import time.
        self._apply_endpoint_env()
        try:
            from flowcept import Flowcept # type: ignore
            try:
                from flowcept import FlowceptTask # type: ignore
            except ImportError: # older layout exposes it under instrumentation
                from flowcept.instrumentation.task import FlowceptTask # type: ignore
        except ImportError as exc: # optional dependency not installed
            raise ImportError(
                "PALISADE: flowcept_enabled=True but the 'flowcept' "
                "package is not installed. Install the optional dependency "
                "(e.g. `uv pip install flowcept` or the 'palisade-flowcept' "
                "extra), or set "
                "PALISADE_FLOWCEPT_ENABLED=false."
            ) from exc

        self._task_cls = FlowceptTask
        # The controller establishes the MQ connection and (when
        # start_persistence=True) the DB-persistence consumer. workflow_id
        # is the session id so every task this sink publishes groups under
        # one auditable workflow.
        self._flowcept = Flowcept(
            workflow_name=self._workflow_name,
            workflow_id=self._session_id,
            start_persistence=self._start_persistence,
        )
        self._flowcept.start()
        self._started = True

    def emit(self, event: "ProvenanceEvent") -> None:
        self._ensure_started()
        used = {
            "session_id": event.session_id,
            "event_type": event.event_type,
        }
        custom_metadata = {
            "palisade": True,
            "event_type": event.event_type,
            # Carry PALISADE's own UTC timestamp alongside Flowcept's
            # task timing so the two can be correlated.
            "palisade_timestamp": event.timestamp,
            "endpoint": self._endpoint,
        }
        # Build the task, then finalize with `.end(generated=...)`. `end()`
        # captures status + outputs and publishes the task to the MQ; this
        # is Flowcept's documented "custom task" pattern. workflow_id is
        # inherited from the active controller (our session_id).
        task = self._task_cls(
            activity_id=event.event_type,
            used=used,
            tags=["palisade", event.event_type],
            custom_metadata=custom_metadata,
        )
        task.end(generated=dict(event.payload))

    def close(self) -> None:
        """Best-effort flush/stop of the Flowcept controller."""
        if self._flowcept is not None:
            stop = getattr(self._flowcept, "stop", None)
            if callable(stop):
                try:
                    stop()
                except Exception as exc: # noqa: BLE001 - teardown must not raise
                    logger.warning(
                        "PALISADE: error stopping Flowcept controller (%s: %s)",
                        type(exc).__name__, exc,
                    )
        self._flowcept = None
        self._task_cls = None
        self._started = False


# -----------------------------------------------------------------
# ProvenanceEvent
# -----------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceEvent:
    """
    A single audit event, returned from `ProvenanceEmitter.emit_*`
    and exposed as `ProvenanceEmitter.last_event` for tests and the
    incident-manager rate-limiting path.
    """

    event_type: str
    timestamp: str
    session_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------
# ProvenanceEmitter
# -----------------------------------------------------------------


class ProvenanceEmitter:
    """
    Serializes PALISADE gate decisions and incidents to a JSONL
    log file (or the module logger when no path is configured), and --
    when `flowcept_enabled=True` -- ships them to a Flowcept broker via
    `FlowceptSink`. The broker path is fail-soft: the first sink error
    (flowcept not installed, broker unreachable) logs once and degrades
    to the JSONL log for the rest of the session.

    Not thread-safe by design. See `CapabilityRegistry`'s docstring
    for the rationale (one async task per `ProjectAgent.run_stream`).
    If a future deployment needs cross-task emission, wrap the
    file-write under a lock at the call site -- the 
    no-contention path is the common one.
    """

    def __init__(
        self,
        settings: PalisadeSettings,
        *,
        session_id: str | None = None,
        log_path: str | Path | None = None,
        sink: Any | None = None,
    ) -> None:
        """
        Construct a ProvenanceEmitter.

        Args:
            settings: the PALISADE settings sub-model. The master
                `enabled` flag, the Flowcept toggle/endpoint, and
                the optional `provenance_log_path` are read from
                here.
            session_id: a stable identifier for the session this
                emitter audits. When None, a UUID4 hex is generated
                at construction time so every event from this
                emitter shares an id. does not yet wire
                session ids through `ProjectAgent`; callers should
                pass the eventual session identifier when it is
                available.
            log_path: explicit override for the JSONL destination.
                Takes precedence over `settings.provenance_log_path`.
                Useful for tests; production code typically relies
                on the settings field so all emitters in a
                deployment share a destination.

        Raises:
            ValueError: if `settings.flowcept_enabled is True` and
                `settings.flowcept_endpoint is None`. This is the
                fail-fast contract:
                a Flowcept-enabled deployment without an endpoint
                is mis-configured at boot, and surfacing that here
                is more useful than waiting for the first gate hit.
        """
        if settings.flowcept_enabled and settings.flowcept_endpoint is None:
            raise ValueError(
                "PalisadeSettings: flowcept_enabled=True requires "
                "flowcept_endpoint to be set. Configure "
                "PALISADE_FLOWCEPT_ENDPOINT or leave "
                "PALISADE_FLOWCEPT_ENABLED at its "
                "default (False)."
            )

        self._settings = settings
        self._session_id = (
            session_id if session_id is not None else uuid.uuid4().hex
        )

        # Resolve the log destination. Explicit constructor argument
        # wins, settings second, otherwise we fall back to the module
        # logger.
        resolved_path = log_path if log_path is not None else settings.provenance_log_path
        self._log_path: Path | None = (
            Path(resolved_path) if resolved_path is not None else None
        )

        self._last_event: ProvenanceEvent | None = None

        # broker sink. When Flowcept is enabled we ship events to
        # the broker; an injected `sink` (tests, or an alternate broker)
        # takes precedence over the default `FlowceptSink`. The sink is
        # constructed eagerly but imports `flowcept` lazily, so this does
        # not require the optional dependency at construction time.
        if sink is not None:
            self._sink: Any | None = sink
        elif settings.flowcept_enabled:
            self._sink = FlowceptSink(
                settings.flowcept_endpoint, # validated non-None above
                session_id=self._session_id,
                start_persistence=settings.flowcept_start_persistence,
            )
        else:
            self._sink = None
        # Once the broker sink fails (e.g. flowcept not installed), stop
        # retrying for this session and degrade to the JSONL log so the
        # agent run is never blocked on provenance.
        self._sink_failed = False

    # -----------------------------------------------------------------
    # Read-only state
    # -----------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """The session identifier stamped on every event."""
        return self._session_id

    @property
    def log_path(self) -> Path | None:
        """
        The resolved JSONL destination, or None when events go
        through the module logger.
        """
        return self._log_path

    @property
    def last_event(self) -> ProvenanceEvent | None:
        """
        The most recent event, or None if none have been emitted.
        Useful for tests and for rate-limiting logic in the
        incident manager. Not a full history -- the JSONL file (or
        Flowcept) is the authoritative record.
        """
        return self._last_event

    # -----------------------------------------------------------------
    # Emit methods
    # -----------------------------------------------------------------

    def emit_gate_decision(
        self,
        gate: str,
        decision: GateDecision,
        context: GateContext | None = None,
        *,
        tier: str | None = None,
    ) -> ProvenanceEvent | None:
        """
        Record a gate's `GateDecision` for audit.

        Args:
            gate: short identifier of the gate (e.g., ``"G2"``). By
                convention this is the `Gate.name` attribute of the
                emitting gate.
            decision: the `GateDecision` produced. The summary
                captures `allow`, `reason`, `incident_level`, and
                whether `rewritten_args` was set; the
                `capability_tag` field is serialized in full when
                present.
            context: optional `GateContext`. The serializable
                ``metadata`` annotations are folded into the event so
                audit reviewers can correlate a decision with the
                per-call context (e.g. tool name) without joining
                against a separate log.
            tier: which tier produced this decision -- ``"fast"`` for
                the deterministic check or ``"slow"`` for the
                Q-LLM/contract refinement. A single tool call that runs
                both tiers emits one event per tier, so the audit trail
                shows the fast decision and any slow-tier override.

        Returns:
            The `ProvenanceEvent` that was emitted, or None when
            the master flag is off (no-op).
        """
        if not self._settings.enabled:
            return None

        payload: dict[str, Any] = {
            "gate": gate,
            "decision": _summarize_decision(decision),
        }
        if tier is not None:
            payload["tier"] = tier
        if decision.capability_tag is not None:
            payload["capability_tag"] = _capability_tag_to_dict(
                decision.capability_tag
            )
        # context handling: the GateContext fields that are
        # safe to serialize (none of the registry / scorer / agent
        # references are JSON-compatible) come from the metadata
        # dict, which is documented as gate-set per-call annotations.
        if context is not None and context.metadata:
            payload["context_metadata"] = dict(context.metadata)

        return self._emit(EVENT_GATE_DECISION, payload)

    def emit_incident(self, record: IncidentRecord) -> ProvenanceEvent | None:
        """
        Record an `IncidentRecord` produced by `IncidentManager`.

        The signature accepts the same `IncidentRecord` instance
        `IncidentManager.record()` returns; `IncidentManager` calls
        this as `provenance.emit_incident(record_obj)` once the
        emitter is wired into its constructor.

        Returns:
            The `ProvenanceEvent` that was emitted, or None when
            the master flag is off (no-op).
        """
        if not self._settings.enabled:
            return None

        payload: dict[str, Any] = {
            "level": record.level,
            "severity_name": record.severity_name,
            "gate": record.gate,
            "reason": record.reason,
        }
        if record.capability_tag is not None:
            payload["capability_tag"] = _capability_tag_to_dict(
                record.capability_tag
            )
        if record.metadata:
            payload["metadata"] = dict(record.metadata)

        return self._emit(EVENT_INCIDENT, payload)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> ProvenanceEvent:
        """
        Common emit path used by both `emit_gate_decision` and
        `emit_incident`. Builds the event envelope, dispatches to
        either the Flowcept stub or the JSONL writer, caches as
        `last_event`, and returns the event.

        Assumes the master flag was already checked by the public
        caller -- this is internal-only and not exposed.
        """
        event = ProvenanceEvent(
            event_type=event_type,
            timestamp=_utc_now_iso(),
            session_id=self._session_id,
            payload=dict(payload),
        )

        if self._sink is not None and not self._sink_failed:
            try:
                self._sink.emit(event)
            except Exception as exc: # noqa: BLE001 - provenance must not crash the run
                # First failure (e.g. flowcept missing, broker
                # unreachable): log loudly once, then degrade to the
                # durable JSONL path for the rest of the session rather
                # than spamming the log or blocking the agent.
                self._sink_failed = True
                logger.error(
                    "PALISADE: Flowcept emission failed (%s: %s); "
                    "falling back to the JSONL provenance log for the "
                    "rest of this session.",
                    type(exc).__name__, exc,
                )
                self._write_jsonl(event)
            self._last_event = event
            return event

        self._write_jsonl(event)
        self._last_event = event
        return event

    def close(self) -> None:
        """Flush/stop the broker sink (best-effort). Safe to call when no
        sink is configured."""
        if self._sink is not None:
            close = getattr(self._sink, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc: # noqa: BLE001 - teardown must not raise
                    logger.warning(
                        "PALISADE: error closing provenance sink (%s: %s)",
                        type(exc).__name__, exc,
                    )

    def _write_jsonl(self, event: ProvenanceEvent) -> None:
        """
        Serialize `event` and write it.

        - If a `log_path` is configured, append one JSON line to
          that file (open/append/flush/close per call -- see the
          module docstring for the durability rationale).
        - Otherwise, log at INFO via the module logger with the
          serialized payload in the `extra=` slot so JSON-log
          adapters can consume the structured fields without
          parsing the message.
        """
        record_dict = {
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "session_id": event.session_id,
            **event.payload,
        }
        line = json.dumps(record_dict, sort_keys=True, default=str)

        if self._log_path is not None:
            # `parents=True` so a fresh deployment can write to
            # `logs/palisade/provenance.jsonl` without operators
            # creating the directory by hand. `exist_ok=True` keeps
            # this idempotent across re-entries.
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.write("\n")
                fh.flush()
            return

        logger.info(
            "PALISADE provenance %s session=%s",
            event.event_type,
            event.session_id,
            extra={
                "palisade_provenance_event": record_dict,
            },
        )


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _utc_now_iso() -> str:
    """
    Return the current UTC time as an ISO 8601 string with the
    trailing 'Z' suffix consumers expect for UTC timestamps. The
    explicit `replace(tzinfo=None)` + 'Z' pattern is preferred over
    `isoformat()` of an aware datetime because broker
    consumers vary in how they parse the `+00:00` form.
    """
    return (
        datetime.now(tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%S.%f")
    )[:-3] + "Z"


def _summarize_decision(decision: GateDecision) -> dict[str, Any]:
    """
    Reduce a `GateDecision` to its JSON-safe summary fields. The
    capability tag is serialized separately by the caller because
    it shows up at the top level of the event payload, not nested
    under `decision`.
    """
    return {
        "allow": decision.allow,
        "reason": decision.reason,
        "incident_level": decision.incident_level,
        "has_rewritten_args": decision.rewritten_args is not None,
    }


def _capability_tag_to_dict(tag: CapabilityTag) -> dict[str, Any]:
    """
    Serialize a `CapabilityTag` to a plain dict for inclusion in
    the JSONL payload. This is the same shape `IncidentManager`
    uses for its `extra=` log payloads; kept duplicated rather than
    imported because the two consumers serve different sinks
    (logger `extra=` vs. JSONL file) and may diverge.
    """
    return {
        "source": tag.source,
        "sensitivity": tag.sensitivity.value,
        "dual_use": tag.dual_use.value,
        "taint": tag.taint,
        "provenance_chain": list(tag.provenance_chain),
        "metadata": dict(tag.metadata),
    }
