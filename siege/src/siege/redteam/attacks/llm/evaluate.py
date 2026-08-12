"""
Shared evaluation for the B family (WI15): ASR-at-budget per access tier.

Like ``attacks/embedding/evaluate.py`` for the A family, but driven through the
``LLMOptimizerAttacker``'s cost-controlled ``optimize`` entry point (response
cache + token-budget guard) rather than the bare ``run_attacker`` loop. Any
``Attacker`` without an ``optimize`` method falls back to ``run_attacker``, so
the same helper drives a plain bandit or a static baseline too.

``prompt_attack_instance`` is the G1 instance factory: the B1 seed artifact is a
``prompt`` artifact whose ``user_prompt`` payload the env's default factory
already routes to G1, so B1 needs no custom factory -- but the explicit one is
provided for callers that want the untrusted-span ``capability`` threaded onto
the action (the intent-laundering hard-win path).

``live_asr_at_budget`` is the **live** path (B1 step: "LiveSessionRunner runs
each candidate through the real G1 with the Q-LLM extractor active"). It drives
the async attacker loop against a ``LiveSessionRunner`` (a real agent driver +
optional per-gate Q-LLM), reconstructing the ``Transition`` the attacker's
``aobserve`` expects from the substrate reward + access adapter. Offline (no
Q-LLM) it is the same clean-negative-on-hard-win as the deterministic path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from siege.redteam.access import AccessTier, make_adapter
from siege.redteam.env import Artifact, RedTeamEnv, Transition, single_attack_instance
from siege.redteam.metrics import AsrAtBudget
from siege.redteam.reward import CapabilityReward
from siege.redteam.attacks.common import ALL_TIERS, TierCurves

if TYPE_CHECKING:
    from siege.redteam.attacker import Attacker
    from siege.redteam.env import Instance
    from siege.ablation_matrix import AblationConfig

__all__ = ["evaluate_llm_attack", "prompt_attack_instance", "live_asr_at_budget"]


def prompt_attack_instance(artifact: Artifact, *, instance_id: str = "b_redteam") -> "Instance":
    """``Artifact -> Instance`` for a G1 prompt attack.

    Thin wrapper over the env's default ``single_attack_instance`` -- a prompt
    artifact already maps to a one-action G1 ``PROMPT`` instance with the
    untrusted-span ``capability`` copied through. Named so B1 reads explicitly.
    """
    return single_attack_instance(artifact, instance_id=instance_id)


def evaluate_llm_attack(
    attacker_factory: "Callable[[], Attacker]",
    *,
    action_space: Sequence[Artifact],
    budget: int,
    config: "AblationConfig | None" = None,
    tiers: Sequence = ALL_TIERS,
    instance_factory: "Callable[[Artifact], Instance] | None" = None,
    use_optimize: bool = True,
) -> TierCurves:
    """Drive a fresh attacker per access tier and collect ASR-at-budget curves.

    ``attacker_factory`` must return a *fresh* ``Attacker`` each call. When it
    exposes ``optimize`` (the ``LLMOptimizerAttacker`` cost-controlled driver:
    response cache + token budget) that is used; otherwise the bare
    ``RedTeamEnv.run_attacker`` loop. ``config`` defaults to ``full PALISADE``.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    space = list(action_space)
    if not space:
        raise ValueError("action_space must be non-empty")
    curves = {}
    for tier in tiers:
        env = RedTeamEnv(
            action_space=space,
            instance_factory=instance_factory,
            config=config,
            access_tier=tier,
            max_steps=1,
        )
        attacker = attacker_factory()
        if use_optimize and hasattr(attacker, "optimize"):
            log = attacker.optimize(env, budget)
        else:
            log = env.run_attacker(attacker, budget)
        curves[tier] = log.asr
    return TierCurves(curves=curves)


async def live_asr_at_budget(
    attacker: "Attacker",
    *,
    agent_driver: Any,
    budget: int,
    quarantine_agents: "dict[str, Any] | None" = None,
    config: "AblationConfig | None" = None,
    instance_factory: "Callable[[Artifact], Instance]" = single_attack_instance,
    access_tier: AccessTier = AccessTier.WHITE_BOX,
    reward: CapabilityReward | None = None,
) -> AsrAtBudget:
    """Drive the async attacker loop against a live agent + Q-LLM slow tier.

    One query == ``apropose`` -> build instance -> ``LiveSessionRunner.run``
    (real agent driver, optional per-gate Q-LLM via ``quarantine_agents``) ->
    reward/discriminate -> ``aobserve``. Returns the ASR-at-budget curve. This is
    the wiring B1 (and later B3) use when a Q-LLM slow tier is available; offline
    it degrades to the same fast-tier-only verdict as the deterministic path.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")
    from siege.ablation_matrix import CUMULATIVE_CONFIGS
    from siege.live_session_runner import LiveSessionRunner

    runner = LiveSessionRunner(agent_driver=agent_driver, quarantine_agents=quarantine_agents)
    scorer = reward or CapabilityReward()
    adapter = make_adapter(access_tier)
    cfg = config if config is not None else CUMULATIVE_CONFIGS[-1]
    asr = AsrAtBudget()
    for _ in range(budget):
        artifact = await attacker.apropose({}) if hasattr(attacker, "apropose") else attacker.propose({})
        trace = await runner.run(instance_factory(artifact), cfg)
        win = scorer.win_kind(trace)
        transition = Transition(
            action=artifact,
            observation=adapter.observe(trace),
            reward=scorer(trace),
            win_kind=win,
            done=True,
            trace=trace,
            info={"config": cfg.name},
        )
        if hasattr(attacker, "aobserve"):
            await attacker.aobserve(transition)
        else:
            attacker.observe(transition)
        asr.record(win)
    return asr
