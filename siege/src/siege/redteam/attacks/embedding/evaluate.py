"""
Shared evaluation for the A family (WI14): ASR-at-budget per access tier.

Reuses the WI13 substrate end to end -- the read-only ``RedTeamEnv`` driver
(``run_attacker``), the three access-tier adapters, and the ``AsrAtBudget``
curve -- so every A-family attacker emits a soft/hard ASR-at-budget *curve*
per tier for free, never a point (the threat model's first-class budget axis).

Two substrate facts shape this module:

- **G3 payload contract.** ``G3RagGate.check_fast`` requires the retrieval
  payload to be ``{"kb_slug": str, "query": str}`` (extra keys are fine, as the
  static ``b3_3`` template's ``claim`` key shows); the embedding backend's
  default artifact carries ``{"user_prompt", "chunk"}`` instead.
  ``rag_attack_instance`` is the instance factory that bridges the two -- the
  poisoned chunk text becomes the ``query`` G3 scans -- and threads a poisoned
  ``claim`` (A3) and ``capability`` through unchanged.
- **Tiers coincide offline.** The deterministic offline gate stack does not
  observe the attacker's access tier, so the black/grey/white-box curves are
  identical here by construction; the access-tier *separation* becomes
  measurable only under the WI17 live agent (WI19). We still emit all three so
  the per-tier reporting shape is in place.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from siege.redteam.access import AccessTier
from siege.redteam.env import Artifact, RedTeamEnv
from siege.redteam.metrics import AsrAtBudget
from siege.redteam.attacks.common import ALL_TIERS, TierCurves
from siege.schemas import (
    Action,
    ActionKind,
    CapabilitySpec,
    Instance,
    Session,
    SuccessCriterion,
    Turn,
)

if TYPE_CHECKING:
    from siege.redteam.attacker import Attacker
    from siege.ablation_matrix import AblationConfig
    from siege.trace_recorder import Trace

__all__ = [
    "DEFAULT_KB_SLUG",
    "rag_attack_instance",
    "TierCurves",
    "evaluate_embedding_attack",
]

#: The mutable literature collection every A-family attack poisons; matches the
#: ``kb_slug`` the static ``b3_*`` templates and the deployed RAG layer use.
DEFAULT_KB_SLUG = "molten-salt-papers"

_ALL_TIERS = ALL_TIERS


def rag_attack_instance(artifact: Artifact, *, instance_id: str = "a_redteam") -> Instance:
    """``Artifact -> Instance`` for the A family: a one-action G3 RAG retrieve.

    Bridges the embedding backend's ``{"user_prompt","chunk"}`` artifact payload
    to the ``{"kb_slug","query"}`` shape ``G3RagGate.check_fast`` demands (the
    poisoned chunk text becomes the ``query`` the gate scans, mirroring how the
    static templates model the retrieved chunk in the query), and threads a
    poisoned ``claim`` (A3) and provenance ``capability`` through unchanged.
    """
    payload = dict(artifact.payload)
    chunk = payload.get("chunk") or payload.get("query") or payload.get("user_prompt", "")
    action_payload: dict = {
        "kb_slug": payload.get("kb_slug", DEFAULT_KB_SLUG),
        "query": chunk,
    }
    if "claim" in payload:
        action_payload["claim"] = payload["claim"]

    cap: CapabilitySpec | None = None
    if artifact.capability is not None:
        c = artifact.capability
        cap = CapabilitySpec(
            value_id=c.get("value_id", f"chunk:{artifact.label}"),
            source=c.get("source", "rag:corpus"),
            dual_use=c.get("dual_use", "none"),
            taint=bool(c.get("taint", True)),
        )

    action = Action(
        kind=ActionKind.RAG_RETRIEVE,
        gate="G3",
        is_attack=True,
        label=artifact.label or "a_redteam",
        payload=action_payload,
        capability=cap,
    )
    return Instance(
        instance_id=instance_id,
        boundary=artifact.boundary,
        template=artifact.template,
        kind="attack",
        sessions=(Session(session_id="s1", turns=(Turn(actions=(action,)),)),),
        success_criterion=SuccessCriterion(check="attack_action_allowed"),
    )


#: A non-empty ``action_space`` placeholder for ``RedTeamEnv``. The A-family
#: backends propose their own ``Artifact`` from ``propose`` (never an int arm),
#: so the space is never indexed -- it only satisfies the constructor.
_PLACEHOLDER = Artifact(
    label="a_seed",
    kind="rag_retrieve",
    gate="G3",
    payload={"kb_slug": DEFAULT_KB_SLUG, "query": ""},
    boundary="B3",
    template="a_redteam",
)


def evaluate_embedding_attack(
    attacker_factory: "Callable[[], Attacker]",
    *,
    budget: int,
    config: "AblationConfig | None" = None,
    adjudicators: "Sequence[Callable[[Instance, Trace], str | None]] | None" = None,
    tiers: Sequence[AccessTier] = _ALL_TIERS,
    instance_factory: Callable[[Artifact], Instance] = rag_attack_instance,
    action_space: Sequence[Artifact] | None = None,
    quarantine_agents: "dict | None" = None,
) -> TierCurves:
    """Drive a fresh attacker per access tier and collect ASR-at-budget curves.

    ``attacker_factory`` must return a *fresh* ``Attacker`` each call (backend
    state is per-tier). ``adjudicators`` is the "+contract" hook -- A3 passes
    ``data_value_adjudicator`` here to express the "+G3 with data-value
    contract" config without touching gate code. ``config`` defaults to
    ``full PALISADE`` (the env default). Every run emits a curve, not a point.
    Pass ``quarantine_agents`` to wire the Q-LLM slow tier.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    space = list(action_space) if action_space else [_PLACEHOLDER]
    curves: dict[AccessTier, AsrAtBudget] = {}
    for tier in tiers:
        env = RedTeamEnv(
            action_space=space,
            instance_factory=instance_factory,
            config=config,
            access_tier=tier,
            adjudicators=list(adjudicators) if adjudicators else None,
            max_steps=1,
            quarantine_agents=quarantine_agents,
        )
        log = env.run_attacker(attacker_factory(), budget)
        curves[tier] = log.asr
    return TierCurves(curves=curves)
