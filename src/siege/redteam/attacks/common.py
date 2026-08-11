"""
Shared evaluation utilities for the attack families (WI14+).

The per-tier ASR-at-budget container both the A family (embedding-space, WI14)
and the B family (LLM-as-optimizer, WI15) report through. Kept here, above the
family subpackages, so neither family depends on the other for it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from siege.redteam.access import AccessTier
from siege.redteam.env import Artifact, RedTeamEnv
from siege.redteam.metrics import AsrAtBudget

if TYPE_CHECKING:
    from siege.redteam.attacker import Attacker
    from siege.redteam.env import Instance
    from siege.ablation_matrix import AblationConfig

__all__ = ["ALL_TIERS", "TierCurves", "evaluate_attack"]

#: The three access tiers, weakest to strongest.
ALL_TIERS: tuple[AccessTier, ...] = (
    AccessTier.BLACK_BOX,
    AccessTier.GREY_BOX,
    AccessTier.WHITE_BOX,
)


@dataclass(frozen=True)
class TierCurves:
    """The ASR-at-budget curve at each access tier."""

    curves: dict[AccessTier, AsrAtBudget]

    def _pick(self, tier: AccessTier | None) -> AsrAtBudget:
        if tier is not None:
            return self.curves[tier]
        # Default to white-box (the hard-win-hunt tier) when present, else any.
        if AccessTier.WHITE_BOX in self.curves:
            return self.curves[AccessTier.WHITE_BOX]
        return next(iter(self.curves.values()))

    def soft_asr(self, tier: AccessTier | None = None) -> float:
        return self._pick(tier).soft_asr()

    def hard_asr(self, tier: AccessTier | None = None) -> float:
        return self._pick(tier).hard_asr()

    def to_markdown(self) -> str:
        lines = ["| tier | soft ASR | hard ASR |", "|---|---|---|"]
        for tier, asr in self.curves.items():
            lines.append(f"| {tier.value} | {asr.soft_asr():.0%} | {asr.hard_asr():.0%} |")
        lines.append("")
        lines.append(
            "_Offline the gate stack does not observe the attacker's access "
            "tier, so the tiers coincide here by construction; the live agent "
            "(WI17/WI19) differentiates them._"
        )
        return "\n".join(lines)


def evaluate_attack(
    attacker_factory: "Callable[[], Attacker]",
    *,
    action_space: Sequence[Artifact],
    budget: int,
    config: "AblationConfig | None" = None,
    tiers: Sequence[AccessTier] = ALL_TIERS,
    instance_factory: "Callable[[Artifact], Instance] | None" = None,
    adjudicators: "Sequence[Callable[[Instance, object], str | None]] | None" = None,
    use_optimize: bool = True,
) -> TierCurves:
    """Generic per-tier ASR-at-budget driver shared across attack families.

    Drives a fresh ``attacker_factory()`` per access tier against a read-only
    ``RedTeamEnv``; ``adjudicators`` is the "+contract" hook (A3's data-value
    contract, C's correctness contract). Uses the ``LLMOptimizerAttacker``
    ``optimize`` cost-controlled driver when the attacker exposes it, else the
    bare ``run_attacker`` loop. ``config`` defaults to ``full PALISADE``.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    space = list(action_space)
    if not space:
        raise ValueError("action_space must be non-empty")
    curves: dict[AccessTier, AsrAtBudget] = {}
    for tier in tiers:
        env = RedTeamEnv(
            action_space=space,
            instance_factory=instance_factory,
            config=config,
            access_tier=tier,
            adjudicators=list(adjudicators) if adjudicators else None,
            max_steps=1,
        )
        attacker = attacker_factory()
        if use_optimize and hasattr(attacker, "optimize"):
            log = attacker.optimize(env, budget)
        else:
            log = env.run_attacker(attacker, budget)
        curves[tier] = log.asr
    return TierCurves(curves=curves)
