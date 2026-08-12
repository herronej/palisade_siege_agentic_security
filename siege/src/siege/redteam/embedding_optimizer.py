"""
Embedding-space attack backend (PALISADE WI13c).

The gradient-free EmbeddingGemma backend the entire A family depends on. It
runs independently of the LLM-optimizer (WI13b) and wires three pieces behind
the WI13a ``Attacker`` protocol:

- an **encoder** -- a read-only handle to the *deployed* EmbeddingGemma-300M
  (the realism argument: the encoder is public, so this is legitimately
  white-box-on-the-encoder);
- a **rank probe** -- submit a candidate to the live ChromaDB and read back its
  retrieval rank for the attack query;
- a **realizer** -- the paraphrase/inversion loop (``realizer.py``) that turns a
  target embedding into fluent text;

against a **benign manifold** (``manifold.py``) that keeps the search inside the
natural-norm region the G3 cluster detector is blind to.

The A1 objective (integration plan §4), made concrete here:

    maximize   retrieval_rank_topk(chunk)        # the chunk is actually retrieved
    minus      cluster_detectability(chunk)       # ... while invisible to G3's z-test
    subject to norm(chunk) in benign_shell        # ... with a natural norm

Each external dependency has a real implementation and a deterministic offline
one so the loop runs in CI with no model and no network:
``SentenceTransformerEncoder`` / ``HashingEncoder`` and ``ChromaRankProbe`` /
``InMemoryRankProbe``.

Lineage: AgentPoison (Chen et al., arXiv 2407.12784) and PoisonedRAG (Zou et
al., arXiv 2402.07867) are the attacks whose tight-cluster / lone-outlier
signatures the G3 detector catches and that A1 is built to evade by staying on
the benign manifold; vec2text (Morris et al., arXiv 2310.06816) is the optional
inversion realizer.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from siege.redteam.env import Artifact, Observation, Transition
from siege.redteam.manifold import BenignManifold
from siege.redteam.realizer import Realizer
from siege.redteam.reward import WinKind

__all__ = [
    "Encoder",
    "SentenceTransformerEncoder",
    "HashingEncoder",
    "RankResult",
    "RankProbe",
    "InMemoryRankProbe",
    "ChromaRankProbe",
    "served_collection",
    "live_rank_probe",
    "EmbeddingCandidate",
    "EmbeddingOptimizerConfig",
    "EmbeddingOptimizerAttacker",
]

_EPS = 1e-12


# =================================================================
# Encoder -- the EmbeddingGemma handle (real) / a deterministic stub (CI)
# =================================================================


@runtime_checkable
class Encoder(Protocol):
    """Anything that maps texts to an ``(n, dim)`` embedding matrix."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Read-only handle to the deployed EmbeddingGemma-300M (sentence-transformers).

    Mirrors the RAG layer (``rag_mcp.py``):
    ``SentenceTransformer("google/embeddinggemma-300m").encode(..., convert_to_numpy=True)``.
    The model is loaded lazily so importing this module never pulls in torch.
    """

    def __init__(
        self, model_name: str = "google/embeddinggemma-300m", *, device: str = "cpu"
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy

            self._model = SentenceTransformer(self._model_name, device=self._device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure()
        return np.asarray(
            self._model.encode(list(texts), convert_to_numpy=True), dtype=np.float64
        )


class HashingEncoder:
    """Deterministic, offline, signed feature-hashing encoder for CI.

    No model download, no network, reproducible across processes (uses
    ``hashlib``, not the salted built-in ``hash``). Token overlap drives cosine
    similarity, and raw L2 norm grows with token count -- enough geometry to
    exercise the rank probe, the manifold, and the realizer end to end. Not a
    semantic encoder; the real ``SentenceTransformerEncoder`` is for that.
    """

    _TOKEN = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 64) -> None:
        self._dim = int(dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in self._TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
                bucket = h % self._dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                out[i, bucket] += sign
        return out


# =================================================================
# Rank probe -- where does a candidate land for the attack query?
# =================================================================


@dataclass(frozen=True)
class RankResult:
    """The candidate's retrieval position for the attack query."""

    rank: int  # 1-based: 1 == more similar to the query than every corpus doc
    in_top_k: bool
    k: int
    similarity: float  # candidate cosine similarity to the query
    n_corpus: int


class RankProbe(ABC):
    """Returns where a candidate chunk would rank for the attack query."""

    @abstractmethod
    def rank(self, candidate_text: str, *, query: str, k: int) -> RankResult: ...


class InMemoryRankProbe(RankProbe):
    """Offline rank probe: cosine rank of the candidate among an in-memory corpus.

    The CI-side equivalent of ``ChromaRankProbe`` -- same contract (a valid
    top-k rank for a seeded candidate), no ChromaDB.
    """

    def __init__(self, encoder: Encoder, corpus_texts: Sequence[str]) -> None:
        self._encoder = encoder
        self._corpus = list(corpus_texts)
        self._corpus_unit = _unit_rows(
            np.asarray(encoder.encode(self._corpus), dtype=np.float64)
        )

    def rank(self, candidate_text: str, *, query: str, k: int) -> RankResult:
        q = _unit(np.asarray(self._encoder.encode([query])[0], dtype=np.float64))
        cand = _unit(np.asarray(self._encoder.encode([candidate_text])[0], dtype=np.float64))
        corpus_sims = self._corpus_unit @ q
        cand_sim = float(cand @ q)
        rank = 1 + int(np.sum(corpus_sims > cand_sim))
        return RankResult(
            rank=rank, in_top_k=rank <= k, k=k, similarity=cand_sim,
            n_corpus=len(self._corpus),
        )


class ChromaRankProbe(RankProbe):
    """Live rank probe over a deployed ChromaDB collection (read-only).

    Mirrors ``rag_mcp``'s ``collection.query(query_embeddings=[v],
    n_results=...)``. It ranks the candidate among the collection's nearest
    neighbours for the query **without inserting it** -- the probe is
    non-destructive; the actual A1 injection happens only at final submission
    through the signed-manifest path. Assumes a cosine collection
    (``hnsw:space="cosine"``), matching the deployed RAG.
    """

    def __init__(self, collection: Any, encoder: Encoder, *, over_k: int = 20) -> None:
        self._collection = collection
        self._encoder = encoder
        self._over_k = int(over_k)

    def rank(self, candidate_text: str, *, query: str, k: int) -> RankResult:
        q = np.asarray(self._encoder.encode([query])[0], dtype=np.float64)
        n_results = max(self._over_k, k)
        res = self._collection.query(
            query_embeddings=[q.tolist()], n_results=n_results, include=["distances"]
        )
        neighbour_dists = list(res["distances"][0]) if res.get("distances") else []
        cand = np.asarray(self._encoder.encode([candidate_text])[0], dtype=np.float64)
        cand_sim = float(_unit(cand) @ _unit(q))
        cand_dist = 1.0 - cand_sim
        rank = 1 + sum(1 for d in neighbour_dists if d < cand_dist)
        return RankResult(
            rank=rank, in_top_k=rank <= k, k=k, similarity=cand_sim,
            n_corpus=len(neighbour_dists),
        )


# =================================================================
# Live-index wiring (WI19) -- connect the rank probe to the served ChromaDB
# =================================================================
#
# The offline sweep uses ``InMemoryRankProbe`` over a seeded corpus (the CI
# path, R-Int-17). The *reported* manuscript number wants the rank probe run
# against the deployed index the agent actually retrieves from. These two
# helpers do exactly that -- open the served ChromaDB collection read-only and
# wrap it in a ``ChromaRankProbe`` -- mirroring ``rag_mcp`` (a
# ``PersistentClient`` over the ``text_chunks`` cosine collection) without
# importing the MCP server. ``chromadb`` is imported lazily so the offline path
# never pulls it in.


def served_collection(
    persist_dir: str, *, collection_name: str = "text_chunks"
) -> Any:
    """Open the deployed ChromaDB collection at ``persist_dir`` read-only.

    Mirrors ``rag_mcp``'s ``chromadb.PersistentClient(path).get_collection(
    "text_chunks")`` (the cosine collection the RAG layer serves). Raises a
    clear error if ``chromadb`` is absent or the collection is missing, so the
    caller can fall back to the seeded ``InMemoryRankProbe`` CI path.
    """
    try:
        import chromadb  # lazy: the offline path never imports it
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "chromadb is required for the live G3 sweep; use InMemoryRankProbe "
            "for the offline CI path (R-Int-17)."
        ) from exc
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(name=collection_name)


def live_rank_probe(
    *,
    persist_dir: str,
    encoder: Encoder | None = None,
    collection_name: str = "text_chunks",
    over_k: int = 20,
    model_name: str = "google/embeddinggemma-300m",
) -> ChromaRankProbe:
    """Build a ``ChromaRankProbe`` over the served index (the WI19 live wiring).

    ``encoder`` defaults to the deployed ``SentenceTransformerEncoder`` (the
    public EmbeddingGemma-300M -- the same model the RAG layer embeds with, so
    ranks match what the agent sees). Pass an explicit ``encoder`` (e.g. a
    ``HashingEncoder``) only for a probe smoke test.
    """
    collection = served_collection(persist_dir, collection_name=collection_name)
    enc = encoder or SentenceTransformerEncoder(model_name)
    return ChromaRankProbe(collection, enc, over_k=over_k)


# =================================================================
# The attacker
# =================================================================


@dataclass(frozen=True)
class EmbeddingCandidate:
    """One scored poisoned-chunk candidate under the A1 objective."""

    text: str
    rank: int
    in_top_k: bool
    detectability: float  # manifold KL-style density surprise
    in_shell: bool  # raw norm inside the benign band
    evades: bool  # retrieves top-k AND inside the manifold (a soft evasion)
    score: float


@dataclass(frozen=True)
class EmbeddingOptimizerConfig:
    k: int = 5  # top-k retrieval horizon
    w_detect: float = 1.0  # weight on cluster_detectability in the objective
    shell_penalty: float = 1.0  # penalty for falling outside the norm shell


class EmbeddingOptimizerAttacker:
    """Gradient-free embedding-space search behind the WI13a ``Attacker`` protocol.

    Wires the four components from the spec -- ``encoder``, ``manifold``,
    ``rank_probe``, ``realizer`` -- plus the attack ``query`` (what the poisoned
    chunk must be retrieved for). ``search`` runs the A1 objective; ``propose`` /
    ``observe`` make it drivable by the env (the live path) too.
    """

    def __init__(
        self,
        *,
        encoder: Encoder,
        manifold: BenignManifold,
        rank_probe: RankProbe,
        realizer: Realizer,
        query: str,
        target: np.ndarray | None = None,
        seed_text: str = "",
        seed_artifact: Artifact | None = None,
        config: EmbeddingOptimizerConfig | None = None,
    ) -> None:
        self._encoder = encoder
        self._manifold = manifold
        self._rank_probe = rank_probe
        self._realizer = realizer
        self._query = query
        # The target the chunk should embed near: an explicit centroid (A1 uses
        # the query-set centroid) or, by default, the query's own embedding.
        self._target = (
            np.asarray(target, dtype=np.float64)
            if target is not None
            else np.asarray(encoder.encode([query])[0], dtype=np.float64)
        )
        self._seed_text = seed_text or query
        self._cfg = config or EmbeddingOptimizerConfig()
        self._seed_artifact = seed_artifact or Artifact(
            label="a1_natural_norm",
            kind="rag_retrieve",
            gate="G3",
            payload={"user_prompt": self._seed_text},
            boundary="B3",
            template="a1_redteam",
        )

        self._history: list[EmbeddingCandidate] = []
        self._best: EmbeddingCandidate | None = None
        self._pool: list[str] = []
        self._pool_i = 0
        self._last_text: str | None = None

    # -- introspection -------------------------------------------------
    @property
    def best(self) -> EmbeddingCandidate | None:
        return self._best

    @property
    def history(self) -> list[EmbeddingCandidate]:
        return list(self._history)

    # -- the embedding-space optimizer (the headline) ------------------
    def score(self, text: str) -> EmbeddingCandidate:
        """Score one candidate chunk under the A1 objective."""
        emb = np.asarray(self._encoder.encode([text])[0], dtype=np.float64)
        rr = self._rank_probe.rank(text, query=self._query, k=self._cfg.k)
        detect = self._manifold.cluster_detectability(emb)
        in_shell = self._manifold.in_shell(emb)
        evades = rr.in_top_k and in_shell and detect <= self._manifold.detect_threshold
        retrieval = 1.0 if rr.in_top_k else 1.0 / max(rr.rank, 1)
        score = (
            retrieval
            - self._cfg.w_detect * detect
            - (0.0 if in_shell else self._cfg.shell_penalty)
        )
        return EmbeddingCandidate(
            text=text, rank=rr.rank, in_top_k=rr.in_top_k, detectability=detect,
            in_shell=in_shell, evades=evades, score=score,
        )

    def search(self, budget: int, *, seed_text: str | None = None) -> EmbeddingCandidate:
        """Realize ``budget`` candidates toward the target and keep the best.

        Returns the highest-objective candidate -- ideally one that retrieves in
        the top-k while sitting inside the benign manifold (``evades=True``).
        """
        if budget < 1:
            raise ValueError("budget must be >= 1")
        realized = self._realizer.realize_many(
            self._target, seed_text=seed_text or self._seed_text, budget=budget
        )
        for r in realized:
            cand = self.score(r.text)
            self._history.append(cand)
            if self._best is None or cand.score > self._best.score:
                self._best = cand
        if self._best is None:  # realizer returned nothing usable
            self._best = self.score(self._seed_text)
            self._history.append(self._best)
        return self._best

    # -- Attacker protocol (env / live path) ------------------
    def propose(self, obs: Observation) -> Artifact:
        if not self._pool:
            self._pool = [
                r.text
                for r in self._realizer.realize_many(
                    self._target, seed_text=self._seed_text, budget=max(1, self._cfg.k)
                )
            ] or [self._seed_text]
        self._last_text = self._pool[min(self._pool_i, len(self._pool) - 1)]
        self._pool_i += 1
        return self._artifact(self._last_text)

    def observe(self, transition: Transition) -> None:
        # The authoritative embedding signal comes from search(); here we just
        # fold the env's retrieval outcome so the live path can refine.
        if self._last_text is not None:
            cand = self.score(self._last_text)
            self._history.append(cand)
            if self._best is None or cand.score > self._best.score:
                self._best = cand

    def _artifact(self, text: str) -> Artifact:
        payload = {**self._seed_artifact.payload, "user_prompt": text, "chunk": text}
        return replace(self._seed_artifact, payload=payload, label=f"{self._seed_artifact.label}:opt")


# -----------------------------------------------------------------
# helpers
# -----------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), _EPS)


def _unit_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, _EPS, None)
