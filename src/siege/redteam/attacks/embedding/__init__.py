"""
Embedding-space attacks -- the A family (PALISADE WI14).

The G3 RAG-boundary family. All three attacks are **gradient-free** and run
against the deployed EmbeddingGemma-300M encoder (white-box-on-the-encoder
per the adaptive threat model), composing the WI13c substrate
(``embedding_optimizer`` / ``manifold`` / ``realizer``) with, for A3, the
Phase-11 contract library:

- **A1 -- natural-norm corpus poisoning** (``natural_norm.py``, headline):
  poison a chunk that retrieves in top-k while its norm and local density sit
  inside the benign manifold, so the G3 cluster detector's outlier test never
  fires -- the "attack-the-detector-you-built" result.
- **A2 -- hybrid-retrieval seam** (``hybrid_seam.py``): a chunk surfaced by
  the vector path but ranked low by the structured/lexical path, so the
  fusion still returns it; or a chunk whose value contradicts the structured
  path.
- **A3 -- contract-aware data-value poisoning** (``data_value.py``): a
  poisoned value that retrieves for the target query and steers the answer
  while round-tripping around the data-value contract's tolerance -- the
  first measurement of what that contract actually buys.

Everything here is read-only against the gates (no ``palisade.gates``
import); the G3 detector geometry is re-derived by ``BenignManifold`` and the
structured-path geometry by the local lexical surrogate.
"""

from siege.redteam.attacks.embedding.natural_norm import (
    NaturalNormPoisoning,
    NaturalNormResult,
    SignedManifestPrecondition,
)
from siege.redteam.attacks.embedding.hybrid_seam import HybridSeamAttack, HybridSeamResult, LexicalRankProbe
from siege.redteam.attacks.embedding.data_value import (
    DataValuePoisoning,
    DataValueResult,
    PoisonedValue,
    data_value_adjudicator,
)
from siege.redteam.attacks.embedding.evaluate import TierCurves, evaluate_embedding_attack

__all__ = [
    # A1
    "NaturalNormPoisoning",
    "NaturalNormResult",
    "SignedManifestPrecondition",
    # A2
    "HybridSeamAttack",
    "HybridSeamResult",
    "LexicalRankProbe",
    # A3
    "DataValuePoisoning",
    "DataValueResult",
    "PoisonedValue",
    "data_value_adjudicator",
    # shared evaluation (ASR-at-budget per tier)
    "evaluate_embedding_attack",
    "TierCurves",
]
