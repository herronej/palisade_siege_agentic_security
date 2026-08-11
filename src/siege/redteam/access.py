"""
Access-tier observation adapters (WI13).

An adaptive adversary is studied at three access tiers -- how much of the
defender's internal state it can observe. Each adapter maps a
``Trace`` to the observation a policy at that tier is allowed to see:

- **black-box** -- only the terminal verdict (allow / deny). The weakest
  attacker; sees whether its artifact got through, nothing else.
- **grey-box** -- + the incident level it triggered and whether the
  session terminated (trust-tier transition). Models an attacker that can
  read coarse telemetry.
- **white-box** -- + per-action gate decisions, the blocking gate, the
  decision reason (a proxy for fired-rule ids), the capability tags
  (``source`` / ``taint`` / ``dual_use``), and the end-of-run tag
  snapshot. Models an insider / full-introspection attacker.

The same episode produces a valid observation at every tier (WI13
acceptance criterion); a stronger tier is a superset of a weaker one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from siege.trace_recorder import Trace


class AccessTier(str, Enum):
    BLACK_BOX = "black_box"
    GREY_BOX = "grey_box"
    WHITE_BOX = "white_box"


# -----------------------------------------------------------------
# Adapters
# -----------------------------------------------------------------


class ObservationAdapter(ABC):
    """Maps a ``Trace`` (or the initial, pre-step state) to an observation dict."""

    tier: AccessTier

    @abstractmethod
    def observe(self, trace: Trace) -> dict[str, Any]:
        """The observation a policy at this tier sees after an episode step."""

    def initial(self) -> dict[str, Any]:
        """The observation returned by ``env.reset()`` before any step."""
        return {"tier": self.tier.value, "started": True}


def _attack_records(trace: Trace) -> list:
    return [a for a in trace.actions if a.is_attack]


class BlackBoxAdapter(ObservationAdapter):
    tier = AccessTier.BLACK_BOX

    def observe(self, trace: Trace) -> dict[str, Any]:
        attacks = _attack_records(trace)
        allowed = any(a.allowed for a in attacks)
        return {
            "tier": self.tier.value,
            "allowed": allowed,
            "blocked": not allowed,
        }


class GreyBoxAdapter(ObservationAdapter):
    tier = AccessTier.GREY_BOX

    def observe(self, trace: Trace) -> dict[str, Any]:
        attacks = _attack_records(trace)
        allowed = any(a.allowed for a in attacks)
        incident = max((a.incident_level or 0 for a in attacks), default=0)
        return {
            "tier": self.tier.value,
            "allowed": allowed,
            "blocked": not allowed,
            "incident_level": incident,
            "terminated": trace.terminated,
            "n_attack_actions": len(attacks),
            "n_blocked": sum(1 for a in attacks if not a.allowed),
        }


class WhiteBoxAdapter(ObservationAdapter):
    tier = AccessTier.WHITE_BOX

    def observe(self, trace: Trace) -> dict[str, Any]:
        grey = GreyBoxAdapter().observe(trace)
        actions = [
            {
                "kind": a.kind,
                "gate": a.gate,
                "defender_live": a.defender_live,
                "is_attack": a.is_attack,
                "allowed": a.allowed,
                "blocked_by": a.blocked_by,
                "reason": a.reason,  # fired-rule-id proxy
                "incident_level": a.incident_level,
                "value_id": a.value_id,
                "capability": dict(a.capability) if a.capability else None,
            }
            for a in trace.actions
        ]
        return {
            **grey,
            "tier": self.tier.value,
            "config": trace.config_name,
            "actions": actions,
            "final_tags": [
                {"value_id": vid, **fields} for vid, fields in trace.final_tags
            ],
        }


_ADAPTERS: dict[AccessTier, type[ObservationAdapter]] = {
    AccessTier.BLACK_BOX: BlackBoxAdapter,
    AccessTier.GREY_BOX: GreyBoxAdapter,
    AccessTier.WHITE_BOX: WhiteBoxAdapter,
}


def make_adapter(tier: AccessTier) -> ObservationAdapter:
    return _ADAPTERS[tier]()


def build_observation(trace: Trace, tier: AccessTier) -> dict[str, Any]:
    """Convenience: observe ``trace`` at ``tier`` without holding an adapter."""
    return make_adapter(tier).observe(trace)


# -----------------------------------------------------------------
# Tier-limited feedback (WI19)
# -----------------------------------------------------------------
#
# The manuscript access-tier result (WI19 ``tab:access``) rests on one
# principle: an attacker can only *learn from what its tier lets it observe*.
# These two helpers make that principle a callable contract so a tier-aware
# attacker (``redteam.manuscript.TierAdaptiveAttacker``) reads exactly the
# fields ``observe`` above exposes at its tier -- nothing more.


def tier_reward(observation: dict[str, Any], tier: AccessTier) -> float:
    """The scalar reward an attacker at ``tier`` can reconstruct from ``observation``.

    Black-box sees only the verdict (allow/deny), so its signal is binary.
    Grey-box additionally sees the incident level, so a *blocked* attempt still
    yields a graded "how close" signal (a lower incident is nearer to evasion).
    White-box additionally sees the capability tags, so it can reward the
    capability-model failure directly -- the hard win it is hunting. The ordering
    ``black <= grey <= white`` in signal richness is the whole reason a stronger
    tier evades faster; the reward here is monotone in that richness.
    """
    allowed = bool(observation.get("allowed", False))
    base = 1.0 if allowed else 0.0
    if tier is AccessTier.BLACK_BOX:
        return base
    # Grey/white: a graded signal from the (grey-visible) incident level.
    incident = int(observation.get("incident_level") or 0)
    if not allowed:
        base += 0.25 * (1.0 - min(incident, 4) / 4.0)
    if tier is AccessTier.GREY_BOX:
        return base
    # White-box: reward a located tainted value riding an allowed sink (the hard
    # win the tag-drop hunt targets) -- visible only via the white-box tags.
    for tag in observation.get("final_tags", ()):  # type: ignore[assignment]
        if isinstance(tag, dict) and tag.get("taint"):
            base += 1.0
            break
    return base


def block_reason(observation: dict[str, Any], tier: AccessTier) -> str | None:
    """The failure signature an attacker at ``tier`` can read from a *blocked* attempt.

    - **black-box**: only that it was blocked (``"blocked"``) -- no structure.
    - **grey-box**: the incident level (``"incident:<n>"``) -- a coarse class.
    - **white-box**: the fired-rule reason string (the real generalization
      signal), else the blocking gate.

    ``None`` when the attempt was allowed. This is what lets white-box generalize
    a single failure across untried candidates while black-box cannot.
    """
    if observation.get("allowed", False):
        return None
    if tier is AccessTier.BLACK_BOX:
        return "blocked"
    if tier is AccessTier.GREY_BOX:
        return f"incident:{int(observation.get('incident_level') or 0)}"
    # white-box: the fired-rule reason (proxy) from the blocking action.
    for action in observation.get("actions", ()):  # type: ignore[assignment]
        if isinstance(action, dict) and action.get("is_attack") and not action.get("allowed"):
            return str(action.get("reason") or action.get("blocked_by") or "blocked")
    return "blocked"
