"""
A1 -- natural-norm corpus poisoning (PALISADE WI14, headline).

The "attack-the-detector-you-built" result. An indirect-injection attacker
plants a chunk into the literature collection. In principle the G3
embedding-cluster detector + signed manifest catch it; A1 searches text space
for a poisoned chunk that retrieves in the top-k for a target query set **while
its norm and local density sit inside the benign manifold**, so the cluster
detector's outlier test never fires (its documented blind spot). Realizing the
payload as fluent text defeats the perplexity checks for free.

This is pure composition over the WI13c substrate -- ``EmbeddingOptimizerAttacker``
(the objective), ``BenignManifold`` (the norm shell + density surrogate for the
G3 detector, re-derived from the public encoder so no gate is imported), and a
``Realizer`` (paraphrase / vec2text). A1 adds only the query-set centroid target
(plan step 2), the embedded payload (step 5), and the signed-manifest
precondition record (step 6, R-Int-7) so the soft win (detector evasion) is
never conflated with the hard precondition (injection admission).

Lineage: AgentPoison (Chen et al., arXiv 2407.12784) and PoisonedRAG (Zou et
al., arXiv 2402.07867) form the tight/lone embedding clusters the G3 detector
catches and that A1 evades by staying on the benign manifold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from siege.redteam.env import Artifact
from siege.redteam.embedding_optimizer import (
    EmbeddingCandidate,
    EmbeddingOptimizerAttacker,
    EmbeddingOptimizerConfig,
    Encoder,
    RankProbe,
)
from siege.redteam.manifold import BenignManifold, ManifoldReport
from siege.redteam.realizer import Realizer
from siege.redteam.attacks.embedding.evaluate import DEFAULT_KB_SLUG

if TYPE_CHECKING:
    from collections.abc import Callable

    from siege.redteam.attacks.embedding.evaluate import TierCurves

__all__ = [
    "SignedManifestPrecondition",
    "NaturalNormResult",
    "NaturalNormPoisoning",
]


@dataclass(frozen=True)
class SignedManifestPrecondition:
    """The supply-chain precondition A1's *hard* win depends on (R-Int-7).

    A1's detector-evasion (the soft win) stands on its own; whether the poisoned
    chunk is actually *admitted* to the index is a separate, honestly-stated
    precondition. The signed-manifest hash check rejects a chunk iff the check
    is enforced **and** the chunk was injected after signing (not present when
    the manifest was computed). Reported separately so A1 is never over-credited.
    """

    manifest_enforced: bool
    present_before_signing: bool

    @property
    def hash_check_fires(self) -> bool:
        """Would the signed-manifest hash check reject this injection?"""
        return self.manifest_enforced and not self.present_before_signing

    @property
    def injection_admitted(self) -> bool:
        """Does the poisoned chunk reach the live index?"""
        return not self.hash_check_fires

    def to_markdown(self) -> str:
        return (
            "| signed-manifest precondition | value |\n|---|---|\n"
            f"| manifest hash check enforced | {self.manifest_enforced} |\n"
            f"| chunk present before signing | {self.present_before_signing} |\n"
            f"| **hash check fires (injection blocked)** | **{self.hash_check_fires}** |\n"
            f"| injection admitted | {self.injection_admitted} |"
        )


@dataclass(frozen=True)
class NaturalNormResult:
    """The A1 outcome: the best poisoned chunk and its evasion verdict."""

    query_set: tuple[str, ...]
    best: EmbeddingCandidate
    budget: int
    manifest: SignedManifestPrecondition
    manifold_report: ManifoldReport | None = None
    history: tuple[EmbeddingCandidate, ...] = ()

    @property
    def retrieves_top_k(self) -> bool:
        return self.best.in_top_k

    @property
    def inside_norm_shell(self) -> bool:
        return self.best.in_shell

    @property
    def evades_cluster_detector(self) -> bool:
        """Retrieves top-k AND inside the manifold AND under the detect threshold."""
        return self.best.evades

    @property
    def soft_win(self) -> bool:
        """Detector-evasion soft win: the chunk retrieves while invisible to G3."""
        return self.best.evades

    def to_markdown(self) -> str:
        head = (
            "| A1 natural-norm poisoning | value |\n|---|---|\n"
            f"| target queries | {len(self.query_set)} |\n"
            f"| query budget | {self.budget} |\n"
            f"| retrieval rank (1-based) | {self.best.rank} |\n"
            f"| retrieves top-k | {self.best.in_top_k} |\n"
            f"| inside benign norm shell | {self.best.in_shell} |\n"
            f"| cluster detectability |z| | {self.best.detectability:.3f} |\n"
            f"| **evades cluster detector** | **{self.best.evades}** |"
        )
        parts = [head, "", self.manifest.to_markdown()]
        if self.manifold_report is not None:
            parts += ["", self.manifold_report.to_markdown()]
        return "\n".join(parts)


class NaturalNormPoisoning:
    """A1 -- natural-norm corpus poisoning over the WI13c embedding backend.

    Args:
        encoder: read-only handle to the deployed EmbeddingGemma-300M (or the
            offline ``HashingEncoder`` in CI).
        manifold: a fitted ``BenignManifold`` (norm shell + density surrogate).
        rank_probe: ``InMemoryRankProbe`` (CI) or ``ChromaRankProbe`` (live).
        realizer: paraphrase/inversion loop that turns the target into text.
        query_set: the target queries the poisoned chunk must retrieve for; the
            attack targets their **centroid** in embedding space (plan step 2).
        payload_instruction: the instruction/data-value payload embedded into
            the realized fluent text (plan step 5).
        signed_manifest_enforced / present_before_signing: the R-Int-7
            precondition recorded on every result.
    """

    def __init__(
        self,
        *,
        encoder: Encoder,
        manifold: BenignManifold,
        rank_probe: RankProbe,
        realizer: Realizer,
        query_set: Sequence[str],
        payload_instruction: str = "",
        config: EmbeddingOptimizerConfig | None = None,
        kb_slug: str = DEFAULT_KB_SLUG,
        signed_manifest_enforced: bool = False,
        present_before_signing: bool = True,
    ) -> None:
        if not query_set:
            raise ValueError("query_set must be non-empty")
        self._encoder = encoder
        self._manifold = manifold
        self._rank_probe = rank_probe
        self._realizer = realizer
        self._query_set = tuple(query_set)
        self._config = config or EmbeddingOptimizerConfig()
        self._kb_slug = kb_slug
        # Payload embedded into fluent text (plan step 5): the chunk carries the
        # instruction while its embedding is steered to the query-set centroid.
        self._seed_text = (self._query_set[0] + " " + payload_instruction).strip()
        # Target = query-set centroid in EmbeddingGemma space (plan step 2).
        mat = np.asarray(self._encoder.encode(list(self._query_set)), dtype=np.float64)
        self._target = mat.mean(axis=0)
        self._manifest = SignedManifestPrecondition(
            manifest_enforced=bool(signed_manifest_enforced),
            present_before_signing=bool(present_before_signing),
        )

    @property
    def precondition(self) -> SignedManifestPrecondition:
        return self._manifest

    def new_attacker(self) -> EmbeddingOptimizerAttacker:
        """A fresh embedding-optimizer attacker (state is per-run/per-tier)."""
        seed_artifact = Artifact(
            label="a1_natural_norm",
            kind="rag_retrieve",
            gate="G3",
            payload={"kb_slug": self._kb_slug, "query": self._seed_text},
            boundary="B3",
            template="a1_natural_norm",
        )
        return EmbeddingOptimizerAttacker(
            encoder=self._encoder,
            manifold=self._manifold,
            rank_probe=self._rank_probe,
            realizer=self._realizer,
            query=self._query_set[0],
            target=self._target,
            seed_text=self._seed_text,
            seed_artifact=seed_artifact,
            config=self._config,
        )

    def run(
        self,
        budget: int,
        *,
        held_out: Sequence[Sequence[float]] | np.ndarray | None = None,
    ) -> NaturalNormResult:
        """Search ``budget`` candidates and return the best, with the FP report.

        When ``held_out`` benign embeddings are supplied, the result carries the
        surrogate G3 cluster-detector's false-positive rate on that held-out
        split (R-Int-6/8) -- the number that grounds the cluster-evasion claim.
        """
        attacker = self.new_attacker()
        best = attacker.search(budget)
        report = self._manifold.report(held_out) if held_out is not None else None
        return NaturalNormResult(
            query_set=self._query_set,
            best=best,
            budget=int(budget),
            manifest=self._manifest,
            manifold_report=report,
            history=tuple(attacker.history),
        )

    def asr_at_budget(self, budget: int, **kwargs) -> "TierCurves":
        """ASR-at-budget curves per access tier (drives the read-only env)."""
        from siege.redteam.attacks.embedding.evaluate import evaluate_embedding_attack

        return evaluate_embedding_attack(self.new_attacker, budget=budget, **kwargs)
