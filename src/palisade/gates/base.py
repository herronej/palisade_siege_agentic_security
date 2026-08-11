"""
Gate ABC and decision/context dataclasses for PALISADE.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any

from palisade.capabilities import CapabilityRegistry, CapabilityTag
from palisade.trust import TrustScorer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------
# GateDecision
# -----------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    """
    The result of a single gate check.

    """

    allow: bool
    reason: str = ""
    capability_tag: CapabilityTag | None = None
    incident_level: int | None = None
    rewritten_args: dict[str, Any] | None = None
    rewritten_result: str | None = None

    def replace_with(self, **changes: Any) -> "GateDecision":
        """
        Return a new `GateDecision` with the given fields replaced.
        Useful when a slow-tier check wants to refine a fast-tier
        decision without losing fields it doesn't touch.
        """
        from dataclasses import replace
        return replace(self, **changes)


# -----------------------------------------------------------------
# GateContext
# -----------------------------------------------------------------


@dataclass
class GateContext:
    """
    Per-call shared state passed to every gate check.

    """

    capability_registry: CapabilityRegistry
    trust_scorer: TrustScorer
    # field. Will tighten to `ContractLibrary | None` when that
    # class lands.
    contracts: Any | None = None
    # field. Will tighten to `pydantic_ai.Agent | None`.
    quarantine_agent: Any | None = None
    # An optional detection-only second opinion for the slow tier: any
    # object with ``flag(text) -> verdict`` where ``verdict.flagged`` is
    # truthy for "malicious" (``redteam.baselines.detectors.Detector``).
    # Wired only by the judge-augmented ablation config; None everywhere
    # else, so the deployed sidecar path is unchanged.
    judge: Any | None = None
    # (separate issue) field. Will tighten to
    # `ProvenanceEmitter | None`.
    provenance: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------
# Gate ABC
# -----------------------------------------------------------------


class Gate(ABC):
    """
    Base class for G1..G7.
    """

    # Subclasses must override.
    name: str = "Gate"

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    # -----------------------------------------------------------------
    # Public, concrete dispatch -- DO NOT OVERRIDE in subclasses
    # -----------------------------------------------------------------

    async def check_fast(self, payload: Any, ctx: GateContext) -> GateDecision:
        """
        Run the fast-tier check.

        When `self.enabled` is False, returns an allow decision
        without invoking subclass logic.

        Every decision produced by an *enabled* gate is emitted to the
        provenance bus (when one is wired on `ctx`) so the audit bundle
        records allow decisions, not just incidents. The disabled
        pass-through is not provenance-worthy and is skipped.
        """
        if not self.enabled:
            return GateDecision(
                allow=True,
                reason=f"gate {self.name} disabled",
            )
        decision = await self._check_fast_when_enabled(payload, ctx)
        self._emit_decision(ctx, decision, tier="fast")
        return decision

    async def check_slow(
        self,
        payload: Any,
        ctx: GateContext,
        decision: GateDecision,
    ) -> GateDecision:
        """
        Run the slow-tier (Q-LLM-backed) check.

        """
        if not self.enabled:
            return decision
        # The Q-LLM slow tier runs only when a quarantine agent is wired;
        # the contract-registry check needs no model, so it runs whenever
        # the gate is enabled.
        if ctx.quarantine_agent is not None:
            decision = await self._check_slow_when_enabled(payload, ctx, decision)
        decision = self._apply_contract_checks(payload, ctx, decision)
        # The judge runs last so a deterministic contract deny short-circuits
        # its network call; it can only tighten a decision, so order does not
        # change the verdict, only the cost.
        decision = self._apply_judge_check(payload, ctx, decision)
        self._emit_decision(ctx, decision, tier="slow")
        return decision

    # -----------------------------------------------------------------
    # Provenance emission (single chokepoint for all gates)
    # -----------------------------------------------------------------

    def _emit_decision(
        self, ctx: GateContext, decision: GateDecision, *, tier: str
    ) -> None:
        """Emit a gate decision to the provenance bus when one is wired.

        This is the single point every enabled gate's decisions flow
        through (`check_fast` / `check_slow` dispatch), so wiring
        emission here -- rather than at each capability call site --
        guarantees no gate silently skips the audit bundle. A no-op when
        `ctx` carries no provenance emitter (e.g. eval harnesses build a
        bare `GateContext`). Provenance must never break a gate, so any
        emitter error is swallowed with a debug log.
        """
        provenance = getattr(ctx, "provenance", None)
        if provenance is None:
            return
        try:
            provenance.emit_gate_decision(self.name, decision, ctx, tier=tier)
        except Exception: # noqa: BLE001 - provenance must not break a gate
            logger.debug(
                "PALISADE: gate %s provenance emission failed", self.name,
                exc_info=True,
            )

    # -----------------------------------------------------------------
    # Slow-tier contract enforcement
    # -----------------------------------------------------------------

    #: Claim types this gate's slow tier never needs to look past (unused
    #: today; the registry routes by claim ``type``). Kept so a future
    #: gate can scope its contract checks if needed.
    contract_domains: tuple[str,...] = ()

    def _extract_contract_claims(self, payload: Any, ctx: GateContext) -> list[dict]:
        """Claims this gate's slow tier should run through the registry.

        Default: a ``claims`` list on a dict payload, plus any on
        ``ctx.metadata['claims']``. Gates whose claims live elsewhere
        (e.g. inside retrieved text) override this.
        """
        claims: list[dict] = []
        if isinstance(payload, dict) and isinstance(payload.get("claims"), list):
            claims.extend(c for c in payload["claims"] if isinstance(c, dict))
        meta = getattr(ctx, "metadata", None) or {}
        if isinstance(meta.get("claims"), list):
            claims.extend(c for c in meta["claims"] if isinstance(c, dict))
        return claims

    def _apply_contract_checks(
        self, payload: Any, ctx: GateContext, decision: GateDecision
    ) -> GateDecision:
        """Run the contract registry over the gate's claims and fold any
        violation into `decision` (deny + bumped incident level)."""
        registry = getattr(ctx, "contracts", None)
        if registry is None:
            return decision
        claims = self._extract_contract_claims(payload, ctx)
        if not claims:
            return decision
        from palisade.contracts.enforcement import enforce

        outcome = enforce(registry, claims, ctx.trust_scorer)
        if outcome.ok:
            return decision
        level = outcome.incident_level or 2
        bumped = level if decision.incident_level is None else min(
            decision.incident_level, level
        )
        prefix = f"{decision.reason}; " if decision.reason else ""
        return decision.replace_with(
            allow=False,
            incident_level=bumped,
            reason=f"{prefix}{self.name} contract violation -> {outcome.reason}",
        )

    # -----------------------------------------------------------------
    # Slow-tier judge (detection-only second opinion)
    # -----------------------------------------------------------------

    #: Payload keys that carry routing metadata rather than a value the gate
    #: is deciding about. Excluded from the text handed to the judge so a
    #: verdict is never driven by a gate id or tool name.
    _JUDGE_SKIP_KEYS: frozenset[str] = frozenset(
        {"tool_name", "gate", "kb_slug", "namespace", "memory_namespace"}
    )

    def _judge_texts(self, payload: Any, *, _depth: int = 0) -> list[str]:
        """Every string leaf of ``payload`` the judge should read.

        This is the substantive difference between a judge wired *inside*
        the stack and the same model scored as a standalone comparator: as
        a comparator it reads a flat list of an instance's attack strings,
        whereas here it reads what this gate was actually handed -- a whole
        batch script at G5, one retrieved chunk at G3. The framing differs,
        so the two are not interchangeable measurements.
        """
        if _depth > 6:
            return []
        out: list[str] = []
        if isinstance(payload, str):
            if payload.strip():
                out.append(payload)
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if key in self._JUDGE_SKIP_KEYS:
                    continue
                out.extend(self._judge_texts(value, _depth=_depth + 1))
        elif isinstance(payload, (list, tuple)):
            for value in payload:
                out.extend(self._judge_texts(value, _depth=_depth + 1))
        return out

    def _apply_judge_check(
        self, payload: Any, ctx: GateContext, decision: GateDecision
    ) -> GateDecision:
        """Fold an optional detection-only judge verdict into ``decision``.

        Strictly *additive* and strictly one-directional:

        - a MALICIOUS verdict on any text denies (``incident_level=2``);
        - a BENIGN verdict does nothing at all.

        The second half is the load-bearing one. A clean verdict here must
        not clear taint, relax a label, or otherwise widen what the value is
        allowed to do. G2's Sanitize path already learned this: its clean
        verdict used to set ``taint=False``, which made a served, off-host
        model a *declassifier* -- able to erase a recorded provenance fact
        and open a high-privilege sink -- and so silently moved that
        endpoint inside the trusted base. The judge is a detector; it can
        refuse, it cannot license.

        A transport failure is not a verdict. ``Detector.flag`` reports an
        error as ``flagged=False`` rather than raising, so a dead endpoint
        leaves every decision untouched instead of denying the session; the
        caller must read the detector's ``error_rate`` before believing any
        number built from a run. Failing open here is deliberate: an
        additive detector that fails closed becomes a denial-of-service
        vector on the endpoint's availability, and the capability bound --
        which does not call anything -- is what still holds when it is down.
        """
        judge = getattr(ctx, "judge", None)
        if judge is None or not decision.allow:
            return decision
        for text in self._judge_texts(payload):
            try:
                verdict = judge.flag(text)
            except Exception:  # noqa: BLE001 -- a detector must not break a gate
                logger.debug(
                    "PALISADE: judge call failed at gate %s", self.name, exc_info=True
                )
                continue
            if not getattr(verdict, "flagged", False):
                continue
            prefix = f"{decision.reason}; " if decision.reason else ""
            return decision.replace_with(
                allow=False,
                incident_level=2,
                reason=(
                    f"{prefix}{self.name} slow-tier judge: flagged "
                    f"({getattr(verdict, 'reason', 'malicious')})"
                ),
            )
        return decision

    # -----------------------------------------------------------------
    # Subclass hooks
    # -----------------------------------------------------------------

    @abstractmethod
    async def _check_fast_when_enabled(
        self,
        payload: Any,
        ctx: GateContext,
    ) -> GateDecision:
        """
        Subclass-provided fast-tier logic.
        """

    async def _check_slow_when_enabled(
        self,
        payload: Any,
        ctx: GateContext,
        decision: GateDecision,
    ) -> GateDecision:
        """
        Subclass-provided slow-tier logic.
        """
        return decision


# -----------------------------------------------------------------
# PassThroughGate -- a minimal gate used by tests and by the
# disabled-master-flag path of the sidecar.
# -----------------------------------------------------------------


class PassThroughGate(Gate):
    """
    A trivial `Gate` that always allows.

    """

    name = "PassThrough"

    def __init__(
        self,
        *,
        enabled: bool = True,
        reason: str = "pass-through",
    ) -> None:
        super().__init__(enabled=enabled)
        self._reason = reason

    async def _check_fast_when_enabled(
        self,
        payload: Any,
        ctx: GateContext,
    ) -> GateDecision:
        return GateDecision(allow=True, reason=self._reason)
