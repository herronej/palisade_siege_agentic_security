"""
Core adaptive-attack substrate (PALISADE WI13a).

The shared core every attack backend and family plugs into -- and nothing
more. A Gym-style environment wrapping the **read-only** harness
(``SessionRunner`` + the six-config ablation), a capability-model reward
with a soft/hard-win discriminator, three access-tier observation adapters,
the ASR-at-budget curve, and the backend-agnostic ``Attacker`` protocol
that generalizes v0.2's ``Policy``.

**Strictly read-only against the deployed gates.** This package only
*calls* ``build_gate_stack`` / ``SessionRunner.run``; it never imports or
mutates ``palisade.gates`` internals or ``agents.py``.

Surface:

- ``RedTeamEnv`` / ``Artifact`` / ``Transition`` / ``B1FramingSpace`` -- the env.
- ``Attacker`` / ``PolicyAttacker`` / ``StaticAttacker`` -- the backend-agnostic
  attacker protocol (WI13a) and its substrate bridges; ``RedTeamEnv.run_attacker``
  drives any of them over a query budget (``EpisodeLog``).
- ``CapabilityReward`` / ``discriminate`` / ``WinKind`` -- the reward + discriminator.
- ``AccessTier`` / ``build_observation`` -- the observation adapters.
- ``Policy`` / ``EpsilonGreedyBandit`` / ``ScriptedPolicy`` / ``RandomPolicy`` -- policies.
- ``AsrAtBudget`` -- the ASR-at-budget curve.

The attack *backends* (LLM-optimizer -> WI13b, embedding-optimizer -> WI13c,
trained-RL -> WI13b Path B) and the attack *families* (A/B/C -> WI14/15/16)
plug in behind the ``Attacker`` protocol; they are not part of this substrate.
"""

from __future__ import annotations

from siege.redteam.access import (
    AccessTier,
    BlackBoxAdapter,
    GreyBoxAdapter,
    ObservationAdapter,
    WhiteBoxAdapter,
    block_reason,
    build_observation,
    make_adapter,
    tier_reward,
)
from siege.redteam.attacker import (
    Attacker,
    PolicyAttacker,
    StaticAttacker,
)
from siege.redteam.llm_optimizer import (
    AgentProposer,
    CallableProposer,
    Candidate,
    Judge,
    LLMOptimizerAttacker,
    LLMOptimizerConfig,
    LlmJudge,
    ProgrammaticJudge,
    ProposalRequest,
    Proposer,
    ResponseCache,
    Strategy,
    TokenBudget,
    build_judge_agent,
    build_proposer_agent,
)
from siege.redteam.manifold import BenignManifold, ManifoldReport
from siege.redteam.embedding_optimizer import (
    ChromaRankProbe,
    EmbeddingCandidate,
    EmbeddingOptimizerAttacker,
    EmbeddingOptimizerConfig,
    Encoder,
    HashingEncoder,
    InMemoryRankProbe,
    RankProbe,
    RankResult,
    SentenceTransformerEncoder,
    live_rank_probe,
    served_collection,
)
from siege.redteam.realizer import (
    ParaphraseRealizer,
    Paraphraser,
    Realizer,
    RealizeResult,
    Vec2TextRealizer,
    agent_paraphraser,
    build_paraphraser_agent,
)
from siege.redteam.env import (
    Artifact,
    B1FramingSpace,
    EpisodeLog,
    Observation,
    RedTeamEnv,
    Transition,
    single_attack_instance,
)
from siege.redteam.metrics import AsrAtBudget, AsrPoint
from siege.redteam.policy import (
    EpsilonGreedyBandit,
    Policy,
    RandomPolicy,
    ScriptedPolicy,
)
from siege.redteam.reward import (
    CapabilityReward,
    CuriosityBonus,
    RewardShaper,
    WinKind,
    discriminate,
    incident_progress_shaping,
)
from siege.redteam.manuscript import (
    G1BudgetCurve,
    G3Sweep,
    ManuscriptEvalResult,
    TierAdaptiveAttacker,
    g1_access_tier_table,
    g1_access_tier_table_live,
    g1_budget_curve,
    g3_embedding_sweep,
    run_manuscript_eval,
    structural_vs_detection_statement,
)
from siege.redteam.baselines import (
    DenylistDetector,
    Detector,
    DetectorVerdict,
    PromptGuardAdapter,
    ScreenResult,
    screen_texts,
)

__all__ = [
    # env
    "RedTeamEnv",
    "Artifact",
    "Observation",
    "Transition",
    "EpisodeLog",
    "B1FramingSpace",
    "single_attack_instance",
    # attacker protocol (WI13a) + substrate bridges
    "Attacker",
    "PolicyAttacker",
    "StaticAttacker",
    # LLM-optimizer backend (WI13b)
    "LLMOptimizerAttacker",
    "LLMOptimizerConfig",
    "Strategy",
    "Proposer",
    "CallableProposer",
    "AgentProposer",
    "build_proposer_agent",
    "Judge",
    "ProgrammaticJudge",
    "LlmJudge",
    "build_judge_agent",
    "ProposalRequest",
    "Candidate",
    "ResponseCache",
    "TokenBudget",
    # embedding-space backend (WI13c)
    "BenignManifold",
    "ManifoldReport",
    "Encoder",
    "SentenceTransformerEncoder",
    "HashingEncoder",
    "RankProbe",
    "InMemoryRankProbe",
    "ChromaRankProbe",
    "served_collection",
    "live_rank_probe",
    "RankResult",
    "EmbeddingOptimizerAttacker",
    "EmbeddingOptimizerConfig",
    "EmbeddingCandidate",
    "Realizer",
    "ParaphraseRealizer",
    "Vec2TextRealizer",
    "RealizeResult",
    "Paraphraser",
    "build_paraphraser_agent",
    "agent_paraphraser",
    # reward
    "CapabilityReward",
    "discriminate",
    "WinKind",
    "RewardShaper",
    "CuriosityBonus",
    "incident_progress_shaping",
    # access
    "AccessTier",
    "ObservationAdapter",
    "BlackBoxAdapter",
    "GreyBoxAdapter",
    "WhiteBoxAdapter",
    "build_observation",
    "make_adapter",
    "tier_reward",
    "block_reason",
    # manuscript adaptive eval (WI19)
    "TierAdaptiveAttacker",
    "G1BudgetCurve",
    "G3Sweep",
    "ManuscriptEvalResult",
    "g1_budget_curve",
    "g1_access_tier_table",
    "g1_access_tier_table_live",
    "g3_embedding_sweep",
    "structural_vs_detection_statement",
    "run_manuscript_eval",
    # external-defense baselines (WI20)
    "Detector",
    "DetectorVerdict",
    "DenylistDetector",
    "PromptGuardAdapter",
    "ScreenResult",
    "screen_texts",
    # policy
    "Policy",
    "EpsilonGreedyBandit",
    "ScriptedPolicy",
    "RandomPolicy",
    # metrics
    "AsrAtBudget",
    "AsrPoint",
]
