"""
Correctness-sabotage attacks -- the C family (PALISADE WI16).

The attack category with **no direct precedent in the literature**: code that
passes AST/Semgrep and the Q-LLM intent check but emits subtly-wrong science --
a unit error, a sign-flipped coefficient, a wrong-form correlation, an
off-by-one constant -- that a scientist would act on, and that **only the
Phase-11 contracts can catch because static analysis structurally cannot**.

Two sub-objectives, reported separately (``correctness_sabotage.py``):

- **C-validate** -- sabotage the contracts *should* catch (a value driven
  outside the physical-bounds / data-value envelope): the cell drops between
  ``+G4`` (AST/Semgrep only) and ``+G4+contracts``. Validates the contract
  library.
- **C-evade** -- sabotage that *evades* the contracts (a K-vs-degC unit error, a
  wrong correlation form, an unbounded quantity -- none has a distinct
  contract): the residual-risk finding that quantifies the uncovered tail.

Read-only: builds ``create_file`` ``Artifact``s run through the read-only env;
the effect-equivalence oracle execs only this module's own controlled sabotage
templates in a restricted namespace, never attacker-supplied code.
"""

from siege.redteam.attacks.code.correctness_sabotage import (
    CorrectnessSabotage,
    CorrectnessSabotageResult,
    EffectEquivalenceOracle,
    SabotageClass,
    SabotageInstance,
    correctness_adjudicator,
)

__all__ = [
    "CorrectnessSabotage",
    "CorrectnessSabotageResult",
    "EffectEquivalenceOracle",
    "SabotageClass",
    "SabotageInstance",
    "correctness_adjudicator",
]
