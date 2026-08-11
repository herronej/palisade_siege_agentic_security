"""
Common policy interface for the adaptive-adversary trainers (WI13).

Every trainer (bandit, GRPO, HRL, GFlowNet) drives the same ``Policy``
contract: ``propose`` an action given an observation, then ``update`` from
the transition the env returns. WI13 ships the trivial reference policies
the bandit acceptance demo needs; WI14+ subclass ``Policy`` per gate.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

from siege.redteam.env import Transition


class Policy(ABC):
    """Maps an observation to an action and learns from transitions."""

    @abstractmethod
    def propose(self, observation: dict[str, Any]) -> Any:
        """Choose an action (an int index into the env action space, or an Artifact)."""

    def update(self, transition: Transition) -> None:  # noqa: B027 - optional hook
        """Learn from one step. Default: stateless (no-op)."""


class ScriptedPolicy(Policy):
    """Always proposes a fixed action -- the 'trivial scripted policy'."""

    def __init__(self, action: Any) -> None:
        self._action = action

    def propose(self, observation: dict[str, Any]) -> Any:
        return self._action


class RandomPolicy(Policy):
    """Uniform-random over a discrete action space -- the static floor."""

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self._n = n_actions
        self._rng = random.Random(seed)

    def propose(self, observation: dict[str, Any]) -> int:
        return self._rng.randrange(self._n)


class EpsilonGreedyBandit(Policy):
    """ε-greedy value-table bandit over a discrete action space.

    The single-step learner the WI13 acceptance demo uses: it estimates a
    mean reward per arm and concentrates on the best one, so soft-win ASR
    climbs above the uniform static floor as it learns which framing
    operator evades the gate.
    """

    def __init__(self, n_actions: int, epsilon: float = 0.1, seed: int = 0) -> None:
        self._n = n_actions
        self._epsilon = epsilon
        self._rng = random.Random(seed)
        self._counts = [0] * n_actions
        self._values = [0.0] * n_actions

    def propose(self, observation: dict[str, Any]) -> int:
        if self._rng.random() < self._epsilon:
            return self._rng.randrange(self._n)
        best = max(self._values)
        # Break ties randomly among the best arms.
        candidates = [i for i, v in enumerate(self._values) if v == best]
        return self._rng.choice(candidates)

    def update(self, transition: Transition) -> None:
        a = transition.action
        if not isinstance(a, int):
            return
        self._counts[a] += 1
        # Incremental sample-average update.
        self._values[a] += (transition.reward - self._values[a]) / self._counts[a]

    @property
    def values(self) -> list[float]:
        return list(self._values)

    def greedy_action(self) -> int:
        return max(range(self._n), key=lambda i: self._values[i])


def _softmax(logits: list[float]) -> list[float]:
    """Numerically-stable softmax over a list of logits."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


class SoftmaxPolicy(Policy):
    """Categorical policy over a discrete action space, trained by policy gradient.

    ``propose`` samples an arm ~ ``softmax(logits)``; ``update`` is a no-op
    because GRPO updates per *group*, not per step. A GRPO-style trainer
    calls ``update_with_advantage`` once per group member with the
    group-relative advantage (reward - group mean), doing the REINFORCE
    step ``logits += lr * advantage * (onehot(a) - softmax)``.

    This is the contextless, discrete-action realization of the GRPO loop
    (the framing-operator space). The token-level attacker-LM variant swaps
    this for an LM policy behind the same interface; the group-relative
    advantage math here is identical.
    """

    def __init__(self, n_actions: int, lr: float = 0.3, seed: int = 0) -> None:
        self._n = n_actions
        self._lr = lr
        self._rng = random.Random(seed)
        self._logits = [0.0] * n_actions

    def probs(self) -> list[float]:
        return _softmax(self._logits)

    def propose(self, observation: dict[str, Any]) -> int:
        p = self.probs()
        # Inverse-CDF sample without numpy.
        r = self._rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if r <= acc:
                return i
        return self._n - 1

    def update_with_advantage(self, action: int, advantage: float) -> None:
        """One REINFORCE step: ``∇ log π(a) = onehot(a) - softmax``."""
        p = self.probs()
        for i in range(self._n):
            grad = (1.0 if i == action else 0.0) - p[i]
            self._logits[i] += self._lr * advantage * grad

    def greedy_action(self) -> int:
        return max(range(self._n), key=lambda i: self._logits[i])


class GFlowNetPolicy(Policy):
    """Tabular trajectory-balance GFlowNet over a discrete action space.

    Each episode is a one-step trajectory ``s0 -> x_a`` (pick one artifact).
    Trajectory balance for a one-step terminal trajectory reduces to
    minimizing ``(logZ + log P_F(a) - log R(x_a))**2`` where the flow reward
    is ``R = exp(beta * reward)`` (strictly positive; ``beta`` is the
    reward temperature). At the optimum ``P_F(a) ∝ exp(beta * reward_a)`` --
    i.e. the policy samples arms *in proportion to reward* rather than
    collapsing onto the single best one. That reward-proportional spread is
    the diversity property the WI16 instance generator wants: over multiple
    equal-reward evaders, the GFlowNet keeps mass on *all* of them, where the
    ε-greedy bandit concentrates on one.

    ``propose`` samples ~ ``softmax(logits)``; a GFlowNet (trajectory-balance)
    trainer calls ``tb_update`` per step with the realized reward.
    """

    def __init__(self, n_actions: int, lr: float = 0.1, beta: float = 2.0, seed: int = 0) -> None:
        self._n = n_actions
        self._lr = lr
        self._beta = beta
        self._rng = random.Random(seed)
        self._logits = [0.0] * n_actions
        self._log_z = 0.0

    def probs(self) -> list[float]:
        return _softmax(self._logits)

    @property
    def log_z(self) -> float:
        return self._log_z

    def propose(self, observation: dict[str, Any]) -> int:
        p = self.probs()
        r = self._rng.random()
        acc = 0.0
        for i, pi in enumerate(p):
            acc += pi
            if r <= acc:
                return i
        return self._n - 1

    def tb_update(self, action: int, reward: float) -> None:
        """One trajectory-balance gradient step for the sampled action."""
        p = self.probs()
        log_pf = math.log(p[action])
        log_r = self._beta * reward
        delta = self._log_z + log_pf - log_r
        # dL/dlogZ = 2*delta ; dL/dlogits_i = 2*delta*(1[i==a] - p_i)
        self._log_z -= self._lr * 2.0 * delta
        for i in range(self._n):
            grad = (1.0 if i == action else 0.0) - p[i]
            self._logits[i] -= self._lr * 2.0 * delta * grad

    def greedy_action(self) -> int:
        return max(range(self._n), key=lambda i: self._logits[i])
