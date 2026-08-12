"""
The ``Attacker`` protocol -- the core adaptive-attack substrate (WI13a).

This is the one interface every attack *backend* plugs into. v0.2 organized
the harness around a single RL ``Policy`` (``policy.py``): map an
observation to a discrete action index, then ``update`` from the
transition. That contract is RL-shaped -- it assumes a fixed, indexable
action space and a learner that updates value estimates per step. The
adaptive-adversary program (integration plan v0.5 §2) needs three backends
that are *not* all RL:

- **LLM-as-optimizer** (PAIR/TAP): proposes a free-form artifact (a prompt,
  a poisoned chunk, a rewritten SLURM body), then *refines* it from judge
  feedback. There is no discrete action index; ``observe`` appends feedback
  to the proposer's context. **Primary backend. No GPU.** -> lands in WI13b
  (``llm_optimizer.py``).
- **embedding-optimizer**: a gradient-free search in EmbeddingGemma space
  whose ``propose`` realizes a candidate embedding back into text. -> lands
  in WI13c (``embedding_optimizer.py``).
- **trained-RL**: wraps a ``trainers/`` policy (GRPO token attacker). The
  v0.2 shape, retained for B1 Path B only; GPU. -> lands in WI13b.

So WI13a *generalizes* ``Policy`` into ``Attacker``: ``propose(obs) ->
Artifact`` (always an artifact, never an opaque index) and ``observe(
transition) -> None`` (refine / update / no-op -- whatever the backend
does between queries). ``Artifact`` and ``Transition`` are unchanged from
``env.py``; the env drives any ``Attacker`` over a query budget via
``env.run_attacker``.

**Scope (WI13a): the protocol and the substrate bridge only.** No attack
family and no optimizer backend ships here -- those are WI13b/13c/14/15/16.
What *does* ship is the bridge that makes the generalization real and
testable: ``PolicyAttacker`` re-expresses the v0.2 reference ``Policy``
objects (``ScriptedPolicy`` / ``RandomPolicy`` / ``EpsilonGreedyBandit``)
behind the new protocol, and ``StaticAttacker`` is the non-adaptive floor.
Both are pure glue -- zero attack logic -- so the protocol is exercised
end-to-end (acceptance: a black-box and a white-box attacker each yield a
valid observation on one episode and emit an ASR-at-budget curve) before a
single backend lands.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

from siege.redteam.env import Artifact, Observation, Transition
from siege.redteam.policy import Policy

__all__ = [
    "Attacker",
    "PolicyAttacker",
    "StaticAttacker",
]


@runtime_checkable
class Attacker(Protocol):
    """One interface for the LLM-optimizer, embedding-optimizer, and trained-RL backends.

    ``propose`` emits the next attack ``Artifact`` to submit through the
    read-only gate stack; ``observe`` is the post-query hook the backend
    uses to refine (LLM-optimizer), update (trained-RL), or do nothing
    (a static/scripted attacker). The env (``env.run_attacker``) calls them
    in lockstep -- ``propose`` then ``step`` then ``observe`` -- for a fixed
    query budget, which is the first-class axis of the threat model.

    Being ``@runtime_checkable`` lets the harness assert that a backend
    satisfies the contract (``isinstance(x, Attacker)``) without importing
    the concrete backend -- the WI13b/13c backends are validated against
    this protocol the moment they land.
    """

    def propose(self, obs: Observation) -> Artifact:
        """Choose the next attack artifact, given what this tier can observe."""
        ...

    def observe(self, transition: Transition) -> None:
        """Refine / update / no-op from the transition the env returned."""
        ...


class PolicyAttacker:
    """Bridges a v0.2 discrete ``Policy`` to the ``Attacker`` protocol.

    This is the concrete proof that ``Attacker`` *generalizes* ``Policy``:
    every v0.2 reference policy (``ScriptedPolicy``, ``RandomPolicy``,
    ``EpsilonGreedyBandit``, and the policy-gradient/GFlowNet learners)
    drops in unchanged behind the new protocol.

    A discrete ``Policy`` chooses an ``int`` arm into a fixed action space
    and learns over those ints (e.g. the bandit's per-arm value table). The
    protocol, by contrast, speaks ``Artifact``s. The bridge therefore:

    - ``propose``: calls ``policy.propose(obs)``; if the policy returned an
      ``int`` arm, it resolves it to ``action_space[arm]`` and remembers the
      arm; an ``Artifact`` is passed through untouched.
    - ``observe``: forwards the transition to ``policy.update`` -- but first
      rewrites ``transition.action`` back to the remembered ``int`` arm, so
      the discrete learner still sees the index it actually chose (the env
      records the resolved ``Artifact`` as the action). Without this rewrite
      a value-table learner like ``EpsilonGreedyBandit`` -- which ignores a
      non-``int`` action -- would never learn through the bridge.

    Note this is distinct from WI13b's ``TrainedAttacker``: that wraps a
    *trained, GPU* attacker LM; this re-expresses the already-present
    in-process reference policies and carries no backend of its own.
    """

    def __init__(self, policy: Policy, action_space: list[Artifact]) -> None:
        if not action_space:
            raise ValueError("PolicyAttacker requires a non-empty action_space")
        self.policy = policy
        self.action_space = list(action_space)
        self._last_arm: int | None = None

    def propose(self, obs: Observation) -> Artifact:
        choice = self.policy.propose(obs)
        if isinstance(choice, Artifact):
            self._last_arm = None
            return choice
        if isinstance(choice, int):
            if not 0 <= choice < len(self.action_space):
                raise IndexError(
                    f"policy proposed arm {choice} outside [0,{len(self.action_space)})"
                )
            self._last_arm = choice
            return self.action_space[choice]
        raise TypeError(
            f"wrapped Policy.propose must return an int arm or an Artifact, "
            f"got {type(choice)!r}"
        )

    def observe(self, transition: Transition) -> None:
        # Hand the discrete learner back the int arm it chose, not the
        # resolved Artifact the env recorded -- otherwise an int-indexed
        # value table never updates.
        if self._last_arm is not None and not isinstance(transition.action, int):
            transition = replace(transition, action=self._last_arm)
        self.policy.update(transition)


class StaticAttacker:
    """The non-adaptive floor as an ``Attacker``: always proposes one artifact.

    ``observe`` is a no-op -- a static attacker never refines. This is the
    baseline an adaptive backend must beat on the ASR-at-budget curve (the
    "does the attacker do better than firing the same payload every time?"
    reference), and the simplest possible conformance check for the
    protocol and the ``env.run_attacker`` driver.
    """

    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact

    def propose(self, obs: Observation) -> Artifact:
        return self.artifact

    def observe(self, transition: Transition) -> None:
        """Static: no refinement."""
