"""
LLM-as-optimizer attacks -- the B family (PALISADE WI15).

The family the ``LLMOptimizerAttacker`` (WI13b) primary backend was built for.
All four attacks treat an LLM as a gradient-free optimizer over attack text and
run **read-only** against the gate stack (no ``palisade.gates`` import):

- **B1 -- dual-LLM intent-laundering at G1** (``intent_laundering.py``,
  headline): refine a prompt until the G1 fast tier admits it while it carries
  a user-untrusted instruction. The soft-vs-hard separation is the
  methodological centerpiece -- the hard win (an untrusted span re-tagged to
  ``user-trusted`` intent) is realizable only against the **live** Q-LLM
  extractor, so offline it is an honest clean negative.
- **B2 -- stateful multi-turn / trust-oscillation** (``multiturn_oscillation.py``):
  a crescendo of benign turns then a dual-use pivot, and an oscillation policy
  that probes trust down then strikes during recovery -- the optimization-backed
  test of the sticky high-stakes cap (C4).
- **B3 -- cross-gate chaining** (``cross_gate.py`` + the top-level
  ``redteam/orchestrator.py``): the decisive composition experiment. A
  hierarchical manager chains per-gate workers (the A1 poisoned-chunk worker at
  G3 -> a G4 code worker -> a G5 SLURM worker); the reward is the end-to-end
  hard win, measured across the six-config ablation with transfer.
- **B4 -- scientific strategy library** (``strategy_library.py``): an
  AutoDAN-Turbo-style lifelong library that discovers, stores, retrieves, and
  recombines scientific-agent attack strategies; doubles as an instance
  generator (the template-mutation + curation fallback is the offline core).
"""

from siege.redteam.attacks.llm.cross_gate import CrossGateChaining, CrossGateResult
from siege.redteam.attacks.llm.evaluate import evaluate_llm_attack, live_asr_at_budget, prompt_attack_instance
from siege.redteam.attacks.llm.intent_laundering import (
    IntentLaunderingAttack,
    IntentLaunderingResult,
    default_launder_proposer,
)
from siege.redteam.attacks.llm.multiturn_oscillation import (
    CrescendoResult,
    MultiTurnOscillation,
    OscillationResult,
)
from siege.redteam.attacks.llm.strategy_library import ScientificStrategyLibrary, Strategy

__all__ = [
    # shared evaluation
    "evaluate_llm_attack",
    "prompt_attack_instance",
    "live_asr_at_budget",
    # B1
    "IntentLaunderingAttack",
    "IntentLaunderingResult",
    "default_launder_proposer",
    # B2
    "MultiTurnOscillation",
    "OscillationResult",
    "CrescendoResult",
    # B3
    "CrossGateChaining",
    "CrossGateResult",
    # B4
    "ScientificStrategyLibrary",
    "Strategy",
]
