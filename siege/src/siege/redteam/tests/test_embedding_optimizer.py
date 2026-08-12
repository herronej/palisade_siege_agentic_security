"""
WI13c acceptance (rank-probe half) + encoder/attacker units.

Acceptance: the rank probe returns a valid top-k rank for a seeded candidate
against ChromaDB. CI uses an in-process **ephemeral ChromaDB** (the real
``ChromaRankProbe`` code path, no network) and the offline ``InMemoryRankProbe``.
"""

from __future__ import annotations

import numpy as np
import pytest

from siege.redteam.attacker import Attacker
from siege.redteam.embedding_optimizer import (
    ChromaRankProbe,
    EmbeddingOptimizerAttacker,
    EmbeddingOptimizerConfig,
    HashingEncoder,
    InMemoryRankProbe,
    SentenceTransformerEncoder,
)
from siege.redteam.manifold import BenignManifold
from siege.redteam.realizer import ParaphraseRealizer

_QUERY = "melting point of flibe salt"
_CORPUS = [
    "quarterly budget review and staffing plan",
    "reactor maintenance scheduling notes",
    "cafeteria menu for the week",
    "the density of sodium chloride at high temperature",
    "travel reimbursement policy update",
]
_GOOD = "flibe salt melting point measurement"  # echoes the query terms
_BAD = "cafeteria menu staffing budget"  # echoes the corpus, not the query


def _paraphraser(seed: str, n: int) -> list[str]:
    """Deterministic stand-in for the paraphraser LLM."""
    variants = [_BAD, _GOOD, "flibe melting point thermophysical data", seed, "unrelated text"]
    return variants[:n]


# -----------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------


def test_hashing_encoder_is_deterministic_and_overlap_sensitive():
    enc = HashingEncoder(dim=64)
    a1 = enc.encode([_GOOD])[0]
    a2 = enc.encode([_GOOD])[0]
    assert np.allclose(a1, a2)  # deterministic across calls
    q = enc.encode([_QUERY])[0]
    cos = lambda x, y: float(x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
    # The query-echoing text is closer to the query than the corpus-echoing one.
    assert cos(enc.encode([_GOOD])[0], q) > cos(enc.encode([_BAD])[0], q)


def test_real_encoder_is_lazy_and_does_not_load_torch_on_import():
    enc = SentenceTransformerEncoder()  # constructing must not load the model
    assert enc._model is None


# -----------------------------------------------------------------
# Rank probe (acceptance) -- InMemory and real ephemeral ChromaDB
# -----------------------------------------------------------------


def test_inmemory_rank_probe_returns_valid_top_k_rank():
    enc = HashingEncoder(dim=128)
    probe = InMemoryRankProbe(enc, _CORPUS)
    res = probe.rank(_GOOD, query=_QUERY, k=3)
    assert 1 <= res.rank <= len(_CORPUS) + 1  # a valid rank
    assert res.in_top_k  # the seeded query-echoing candidate retrieves top-k
    # An off-query candidate ranks worse than the seeded one.
    assert probe.rank(_BAD, query=_QUERY, k=3).rank > res.rank


def test_chroma_rank_probe_returns_valid_top_k_rank_against_ephemeral_db():
    chromadb = pytest.importorskip("chromadb")
    enc = HashingEncoder(dim=128)
    client = chromadb.EphemeralClient()
    col = client.create_collection("redteam_rank_probe", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"doc{i}" for i in range(len(_CORPUS))],
        embeddings=[enc.encode([t])[0].tolist() for t in _CORPUS],
        documents=list(_CORPUS),
    )
    probe = ChromaRankProbe(col, enc, over_k=10)
    res = probe.rank(_GOOD, query=_QUERY, k=3)
    assert 1 <= res.rank  # a valid rank against the live (ephemeral) ChromaDB
    assert res.in_top_k
    assert res.rank < probe.rank(_BAD, query=_QUERY, k=3).rank


# -----------------------------------------------------------------
# The attacker
# -----------------------------------------------------------------


def _attacker(k: int = 3) -> EmbeddingOptimizerAttacker:
    enc = HashingEncoder(dim=128)
    manifold = BenignManifold(k=3, target_fp=0.1).fit(enc.encode(_CORPUS + [_GOOD, _QUERY]))
    return EmbeddingOptimizerAttacker(
        encoder=enc,
        manifold=manifold,
        rank_probe=InMemoryRankProbe(enc, _CORPUS),
        realizer=ParaphraseRealizer(enc, _paraphraser, manifold=manifold),
        query=_QUERY,
        seed_text=_QUERY,
        config=EmbeddingOptimizerConfig(k=k),
    )


def test_search_finds_a_top_k_retrieving_candidate():
    attacker = _attacker(k=3)
    best = attacker.search(budget=5)
    assert best.in_top_k  # the search surfaced a chunk that retrieves in top-k
    assert best.score == max(c.score for c in attacker.history)  # best by the A1 objective
    # The rank probe correctly ranks the query-echoing chunk in top-k (the
    # retrieval half of the objective is wired), even if the objective ends up
    # preferring a lower-detectability in-manifold chunk (the evasion half).
    good = [c for c in attacker.history if c.text == _GOOD]
    assert good and good[0].in_top_k
    assert all(c.detectability >= 0.0 for c in attacker.history)


def test_attacker_conforms_to_protocol_and_emits_a_g3_artifact():
    attacker = _attacker()
    assert isinstance(attacker, Attacker)
    art = attacker.propose({})
    assert art.kind == "rag_retrieve" and art.gate == "G3"
    assert "user_prompt" in art.payload and "chunk" in art.payload


def test_search_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        _attacker().search(budget=0)
