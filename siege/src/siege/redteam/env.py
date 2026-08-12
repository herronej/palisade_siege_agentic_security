"""
Gym-style RL environment wrapping the harness (WI13).

``RedTeamEnv`` is the one runnable shape every per-gate adversary policy
(WI14+) plugs into. A ``step`` submits an *artifact* (an attack attempt)
through the **real** PALISADE gate stack via the unmodified
``SessionRunner`` + ablation config, scores the resulting ``Trace`` with
the capability-model reward, and returns ``(observation, reward, done,
info)``. The adversary is strictly read-only against the gates: the env
only ever *calls* ``build_gate_stack`` / ``SessionRunner.run`` -- it never
mutates gate code.

The env is deliberately generic. An ``Artifact`` + an
``instance_factory`` decouple "what the policy emits" from "what the
harness runs", so a token-level G1 policy, a SLURM-rewrite G5 policy, and
the trivial bandit demo all share this env. ``B1FramingSpace`` is the
reference instantiation (the G1 framing operators) used by the bandit
acceptance test.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from siege.ablation_matrix import CUMULATIVE_CONFIGS, AblationConfig
from siege.schemas import (
    Action,
    ActionKind,
    CapabilitySpec,
    Instance,
    Session,
    SuccessCriterion,
    Turn,
)
from siege.session_runner import SessionRunner, TaintBound
from siege.trace_recorder import Trace
from siege.redteam.access import AccessTier, make_adapter
from siege.redteam.metrics import AsrAtBudget
from siege.redteam.reward import CapabilityReward, WinKind

if TYPE_CHECKING:
    # Type-only import: the env *drives* an Attacker but must not import the
    # backend module at runtime -- ``attacker.py`` imports from here, so a
    # runtime import would be circular. The driver is duck-typed (it only
    # calls ``propose`` / ``observe``), so the protocol annotation suffices.
    from siege.redteam.attacker import Attacker


#: What a policy/attacker sees after a step -- the access-tier adapters
#: (``access.py``) return one of these. Kept a plain mapping so every tier
#: (black/grey/white-box) is the same shape to a backend; the protocol in
#: ``attacker.py`` is typed against it.
Observation = dict[str, Any]


# -----------------------------------------------------------------
# Artifact + default instance factory
# -----------------------------------------------------------------


@dataclass(frozen=True)
class Artifact:
    """One attack attempt a policy emits.

    Carries everything the default factory needs to build a single
    attack action: the action ``kind`` (``"prompt"`` / ``"tool_call"`` /
    ``"rag_retrieve"`` ...), the defending ``gate``, the gate-ready
    ``payload``, and an optional provenance ``capability`` (set
    ``taint=True`` to exercise the hard-win path).
    """

    label: str
    payload: dict[str, Any]
    kind: str = "prompt"
    gate: str | None = None
    capability: dict[str, Any] | None = None
    boundary: str = "B1"
    template: str = "redteam"


def single_attack_instance(artifact: Artifact, *, instance_id: str = "redteam") -> Instance:
    """Default factory: a one-action attack ``Instance`` from an ``Artifact``."""
    cap = None
    if artifact.capability is not None:
        cap = CapabilitySpec(
            value_id=artifact.capability.get("value_id", f"art:{artifact.label}"),
            source=artifact.capability.get("source", "user:operator"),
            dual_use=artifact.capability.get("dual_use", "none"),
            taint=bool(artifact.capability.get("taint", True)),
        )
    action = Action(
        kind=ActionKind(artifact.kind),
        gate=artifact.gate,
        is_attack=True,
        label=artifact.label,
        payload=dict(artifact.payload),
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


# -----------------------------------------------------------------
# Step result
# -----------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """One env step, the unit a trainer feeds back to ``Policy.update``."""

    action: Any
    observation: dict[str, Any]
    reward: float
    win_kind: WinKind
    done: bool
    trace: Trace
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeLog:
    """The record of one ``run_attacker`` rollout over a query budget.

    ``transitions`` is every step in order; ``asr`` is the ASR-at-budget
    accumulator (one ``record`` per query) so the run always yields a
    *curve*, not a point; ``queries`` is how many proposals were actually
    submitted (== budget unless the attacker is empty).
    """

    transitions: list[Transition]
    asr: AsrAtBudget
    queries: int

    @property
    def soft_asr(self) -> float:
        return self.asr.soft_asr()

    @property
    def hard_asr(self) -> float:
        return self.asr.hard_asr()

    def wins(self, kind: WinKind) -> int:
        return sum(1 for t in self.transitions if t.win_kind is kind)


# -----------------------------------------------------------------
# Environment
# -----------------------------------------------------------------


class RedTeamEnv:
    """Gym-style env over the read-only PALISADE harness.

    Args:
        action_space: the discrete set of ``Artifact``s a policy chooses
            from (indexable by int).
        instance_factory: ``Artifact -> Instance``. Defaults to
            ``single_attack_instance``.
        config: the ablation config the gates run under. Defaults to
            ``full PALISADE`` (all four gates + trust). Pass an earlier
            cumulative config to ablate.
        access_tier: which observation adapter to expose.
        reward: the ``CapabilityReward``. Defaults to the standard weights.
        max_steps: step budget per episode (1 = single-shot bandit).
        quarantine_agents: optional ``{gate_id: Agent}`` to wire the
            slow-tier Q-LLM (read-only); default offline fast tier only.
        judge: optional slow-tier judge stage (a
            ``redteam.baselines.detectors.Detector``), passed straight to
            ``SessionRunner``. The runner escalates whenever *either* a
            per-gate Q-LLM or a judge is wired, so this can be supplied on
            its own to measure the judge in place of the Q-LLM, or beside it
            for the deployed slow tier. Default: no judge.
    """

    def __init__(
        self,
        *,
        action_space: list[Artifact],
        instance_factory: Callable[[Artifact], Instance] | None = None,
        config: AblationConfig | None = None,
        access_tier: AccessTier = AccessTier.BLACK_BOX,
        reward: CapabilityReward | None = None,
        max_steps: int = 1,
        quarantine_agents: dict[str, Any] | None = None,
        judge: Any | None = None,
        adjudicators: list[Callable[[Instance, Trace], str | None]] | None = None,
        bound: TaintBound = "declarative",
    ) -> None:
        if not action_space:
            raise ValueError("RedTeamEnv requires a non-empty action_space")
        self.action_space = list(action_space)
        self._factory = instance_factory or single_attack_instance
        self._config = config or CUMULATIVE_CONFIGS[-1]
        self._adapter = make_adapter(access_tier)
        self._reward = reward or CapabilityReward()
        self._max_steps = max(1, int(max_steps))
        self._quarantine_agents = quarantine_agents
        self._judge = judge
        # Which label the capability bound reads, passed straight to
        # ``SessionRunner``. ``declarative`` (the default, and what every
        # pre-existing adaptive result is scored under) trusts the corpus-
        # recorded oracle tag; ``production`` reconstructs it by content match,
        # which is the deployed predicate. Exposed so an adaptive driver can be
        # scored under both without reimplementing the runner loop.
        self._bound: TaintBound = bound
        # Extra out-of-band adjudicators (e.g. the correctness
        # contract layer): each maps (instance, trace) -> a block reason or
        # None. Any reason downgrades the episode to blocked. This is how a
        # "+contract" ablation config is expressed without touching gates.
        self._adjudicators = list(adjudicators or [])
        self._steps = 0

    @property
    def n_actions(self) -> int:
        return len(self.action_space)

    @property
    def config(self) -> AblationConfig:
        return self._config

    def reset(self) -> dict[str, Any]:
        self._steps = 0
        return self._adapter.initial()

    def _resolve(self, action: Any) -> Artifact:
        if isinstance(action, Artifact):
            return action
        if isinstance(action, int):
            return self.action_space[action]
        raise TypeError(f"action must be int index or Artifact, got {type(action)!r}")

    async def astep(self, action: Any) -> Transition:
        artifact = self._resolve(action)
        instance = self._factory(artifact)
        runner = SessionRunner(
            quarantine_agents=self._quarantine_agents,
            judge=self._judge,
            bound=self._bound,
        )
        trace = await runner.run(instance, self._config)

        reward = self._reward(trace)
        win = self._reward.win_kind(trace)
        obs = self._adapter.observe(trace)

        # Out-of-band adjudication (correctness contracts etc.): a violation
        # downgrades the episode to blocked even if the gate fast-tier let it
        # through -- the value got retrieved/written but the contract caught it.
        contract_reason: str | None = None
        for adjudicate in self._adjudicators:
            contract_reason = adjudicate(instance, trace)
            if contract_reason:
                break
        if contract_reason:
            win = WinKind.NONE
            reward = self._reward.w_block
            obs = dict(obs)
            obs["allowed"] = False
            obs["blocked"] = True
            obs["contract_blocked"] = contract_reason

        self._steps += 1
        done = (
            self._steps >= self._max_steps
            or trace.terminated
            # single-shot adjudication: a 1-budget episode ends after the verdict
            or self._max_steps == 1
        )
        info = {
            "config": self._config.name,
            "win_kind": win,
            "blocked": not obs.get("allowed", False),
            "label": artifact.label,
            "contract_blocked": contract_reason,
        }
        return Transition(
            action=action,
            observation=obs,
            reward=reward,
            win_kind=win,
            done=done,
            trace=trace,
            info=info,
        )

    def step(self, action: Any) -> Transition:
        """Synchronous step (drives ``astep`` on a private event loop)."""
        return asyncio.run(self.astep(action))

    # -------------------------------------------------------------
    # Attacker driver (WI13a generalization)
    # -------------------------------------------------------------
    #
    # v0.2 left the rollout to each ``trainers/`` subclass (the discrete
    # single-step loop in ``trainers/base.py``). To drive the WI13b/13c
    # backends -- an LLM-optimizer that refines a free-form prompt, an
    # embedding search that realizes a candidate -- the env owns one generic
    # loop over the ``Attacker`` protocol instead: reset, ``propose``,
    # ``astep``, ``observe``, record, repeat for a fixed query budget. This
    # is what makes the env drive LLM/embedding attackers, not only RL
    # trainers, and it ties ASR-at-budget logging to the protocol so every
    # backend's run emits a curve for free.

    async def arun_attacker(
        self,
        attacker: "Attacker",
        budget: int,
        *,
        asr: AsrAtBudget | None = None,
    ) -> EpisodeLog:
        """Drive ``attacker`` for ``budget`` queries against the read-only stack.

        One query == one ``propose`` -> ``astep`` -> ``observe`` cycle. An
        episode runs until it is ``done`` (single-shot for ``max_steps==1``,
        multi-turn otherwise); a fresh episode starts until the query budget
        is spent. ``budget`` is the threat model's first-class axis, so it
        bounds *queries*, not episodes.
        """
        if budget < 1:
            raise ValueError("budget must be >= 1")
        log = asr if asr is not None else AsrAtBudget()
        transitions: list[Transition] = []
        queries = 0
        while queries < budget:
            obs = self.reset()
            done = False
            while not done and queries < budget:
                artifact = attacker.propose(obs)
                transition = await self.astep(artifact)
                attacker.observe(transition)
                log.record(transition.win_kind)
                transitions.append(transition)
                obs = transition.observation
                done = transition.done
                queries += 1
        return EpisodeLog(transitions=transitions, asr=log, queries=queries)

    def run_attacker(
        self,
        attacker: "Attacker",
        budget: int,
        *,
        asr: AsrAtBudget | None = None,
    ) -> EpisodeLog:
        """Synchronous ``arun_attacker`` (drives it on a private event loop)."""
        return asyncio.run(self.arun_attacker(attacker, budget, asr=asr))


# -----------------------------------------------------------------
# B1 framing-operator action space (the reference instantiation)
# -----------------------------------------------------------------


class B1FramingSpace:
    """The G1 framing operators as a discrete ``RedTeamEnv`` action space.

    Wraps ``eval/g1_attacks.ATTACK_GENERATORS`` so each operator
    (instruction-override, role-impersonation, gcg-suffix, ...) is one
    ``Artifact``. Offline, most are caught by the G1 regex while the
    ``gcg_suffix`` operator evades it -- the mixed-reward space the bandit
    demo learns over. The G1 policy reuses this space.
    """

    def __init__(self, operators: tuple[str, ...] | None = None, seed: int = 0) -> None:
        import random

        from siege.eval.g1_attacks import ATTACK_GENERATORS

        ops = operators or (
            "instruction_override",
            "system_prompt_exfil",
            "role_impersonation",
            "dan_family",
            "scientific_framing",
            "output_formatting_hijack",
            "gcg_suffix",
        )
        rng = random.Random(seed)
        self.artifacts: list[Artifact] = []
        for op in ops:
            scenario = ATTACK_GENERATORS[op](rng)
            self.artifacts.append(
                Artifact(
                    label=op,
                    kind="prompt",
                    gate="G1",
                    payload={"user_prompt": scenario.user_prompt},
                    boundary="B1",
                    template="b1_redteam",
                )
            )

    def action_space(self) -> list[Artifact]:
        return list(self.artifacts)
