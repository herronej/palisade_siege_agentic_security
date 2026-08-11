"""
Unit tests for the backend-side G3 hybrid-retrieval helpers
(`gates/g3_hybrid.py`).

Covers the backend's load-bearing pieces of the work item
`Add hybrid BM25 + vector retrieval mode (Semantic
Chameleon arXiv 2603.18034)`:

- Decision predicate `should_inject_hybrid(settings, tool_name)`
  fires only when `g3_enabled=True`, `g3_hybrid_retrieval=True`,
  and the tool is `rag_search`. Covers the AC "When `g3_enabled=true`
  and `g3_hybrid_retrieval=true` in settings, the backend agent
  always invokes `rag_search` with `hybrid=true`."
- Args injection: `inject_hybrid_args(args, alpha)` overrides
  any existing `hybrid` key and doesn't mutate the input dict.
- Mirrored merge math: `merge_retrievals(...)` produces the
  expected merged ordering for synthetic inputs. Includes a
  cross-module pin against the MCP server's copy of the function.
- Default-behavior compatibility: settings default to
  `g3_hybrid_retrieval=False`, so the gate is a no-op for
  unmigrated deployments. Covers the AC "Default behavior
  (`hybrid=false`) unchanged -- backwards compatible."

The PoisonedRAG-style demotion test pins the merge-behavior end of
the Semantic Chameleon defense. A full ASR evaluation against the
PoisonedRAG corpus is an evaluation deliverable (the work
item AC #5), not a unit test.
"""

from __future__ import annotations

from typing import Any

import pytest

from palisade.config import PalisadeSettings
from palisade.gates.g3_hybrid import (
    MirroredRetrievalResult,
    _min_max_normalize,
    inject_hybrid_args,
    merge_retrievals,
    should_inject_hybrid,
)


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------


def _r(chunk_id: str, score: float, *, doc: str = "", **meta: Any) -> MirroredRetrievalResult:
    """Build a `MirroredRetrievalResult` with concise positional args."""
    return MirroredRetrievalResult(
        chunk_id=chunk_id,
        document=doc or chunk_id,
        metadata=dict(meta),
        score=score,
    )


# -----------------------------------------------------------------
# should_inject_hybrid -- the gate's decision predicate
# -----------------------------------------------------------------


def test_should_inject_hybrid_true_when_all_flags_on() -> None:
    """The acceptance criterion in its happy-path."""
    settings = PalisadeSettings(
        enabled=True,
        g3_enabled=True,
        g3_hybrid_retrieval=True,
    )
    assert should_inject_hybrid(settings, "rag_search") is True


def test_should_inject_hybrid_false_when_g3_disabled() -> None:
    """`g3_enabled=False` -> never hybridize."""
    settings = PalisadeSettings(
        enabled=True,
        g3_enabled=False,
        g3_hybrid_retrieval=True,  # would-be on, but g3 master is off
    )
    assert should_inject_hybrid(settings, "rag_search") is False


def test_should_inject_hybrid_false_when_hybrid_flag_off() -> None:
    """
    Default `g3_hybrid_retrieval=False` -> never hybridize. This
    is the load-bearing backwards-compat property: a deployment
    that turns G3 on without explicitly opting into hybrid sees
    no change to its RAG behavior.
    """
    settings = PalisadeSettings(
        enabled=True,
        g3_enabled=True,
        g3_hybrid_retrieval=False,  # default
    )
    assert should_inject_hybrid(settings, "rag_search") is False


def test_should_inject_hybrid_false_for_non_rag_tool() -> None:
    """Hybrid only applies to `rag_search`. Other tools are passed through."""
    settings = PalisadeSettings(
        enabled=True,
        g3_enabled=True,
        g3_hybrid_retrieval=True,
    )
    for name in ("run_bash", "submit_hpc_job", "create_file", "display_file"):
        assert should_inject_hybrid(settings, name) is False


def test_settings_default_disables_hybrid() -> None:
    """
    AC: "Default behavior (`hybrid=false`) unchanged -- backwards
    compatible." Pin the defaults so a future settings change
    that flips them on by accident fires this test.
    """
    settings = PalisadeSettings()
    assert settings.g3_hybrid_retrieval is False
    assert settings.g3_hybrid_alpha == 0.5  # used only when retrieval is on


# -----------------------------------------------------------------
# inject_hybrid_args -- the gate's mutation surface
# -----------------------------------------------------------------


def test_inject_hybrid_args_adds_flags_to_empty_args() -> None:
    out = inject_hybrid_args({}, alpha=0.5)
    assert out == {"hybrid": True, "alpha": 0.5}


def test_inject_hybrid_args_preserves_other_args() -> None:
    out = inject_hybrid_args(
        {"query": "salt corrosion", "kb_slug": "salts", "n_results": 8},
        alpha=0.7,
    )
    assert out == {
        "query": "salt corrosion",
        "kb_slug": "salts",
        "n_results": 8,
        "hybrid": True,
        "alpha": 0.7,
    }


def test_inject_hybrid_args_overrides_caller_supplied_hybrid_flag() -> None:
    """
    The model is not authoritative on security. If the LLM
    spuriously sets `hybrid=False`, the sidecar overrides it
    when the configured policy says hybridize.
    """
    out = inject_hybrid_args(
        {"query": "q", "hybrid": False, "alpha": 0.9},
        alpha=0.3,
    )
    assert out["hybrid"] is True
    assert out["alpha"] == 0.3


def test_inject_hybrid_args_does_not_mutate_input() -> None:
    """
    PALISADE's `process_tool_call` chain reuses argument dicts
    across hooks; silent mutation could corrupt downstream gates.
    """
    original = {"query": "q"}
    out = inject_hybrid_args(original, alpha=0.5)
    assert original == {"query": "q"}
    assert out is not original


# -----------------------------------------------------------------
# Mirrored merge math
# -----------------------------------------------------------------


def test_min_max_normalize_empty() -> None:
    assert _min_max_normalize([]) == []


def test_min_max_normalize_single_item_returns_one() -> None:
    assert _min_max_normalize([0.7]) == [1.0]


def test_min_max_normalize_all_equal_returns_ones() -> None:
    assert _min_max_normalize([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]


def test_min_max_normalize_typical_case() -> None:
    norm = _min_max_normalize([0.0, 0.5, 1.0])
    assert norm == [0.0, 0.5, 1.0]


def test_merge_rejects_alpha_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        merge_retrievals([_r("a", 1.0)], [], alpha=1.5)
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        merge_retrievals([_r("a", 1.0)], [], alpha=-0.1)


def test_merge_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        merge_retrievals([_r("a", 1.0)], [], top_k=0)


def test_merge_empty_inputs_returns_empty() -> None:
    assert merge_retrievals([], []) == []


def test_merge_vector_only_passes_through_at_alpha_one() -> None:
    """`alpha=1.0` ignores BM25 entirely -> vector ordering preserved."""
    vec = [_r("a", 0.9), _r("b", 0.7), _r("c", 0.5)]
    bm25 = [_r("z", 10.0), _r("y", 5.0)]
    merged = merge_retrievals(vec, bm25, alpha=1.0, top_k=5)
    ids = [r.chunk_id for r in merged]
    # Top three are the vector list in its original order; the
    # BM25-only chunks land at the bottom (their vector score is 0).
    assert ids[:3] == ["a", "b", "c"]


def test_merge_bm25_only_passes_through_at_alpha_zero() -> None:
    """
    `alpha=0.0` zeroes the vector contribution -> BM25's *relative*
    ordering survives at the top of the merged list.

    Vector-only chunks and BM25-bottom chunks both end up with
    merged_score=0.0 (the floor of min-max normalization), so we
    don't pin their relative order -- only that the BM25 top is
    rank 1 and the second BM25 chunk is rank 2.
    """
    vec = [_r("a", 0.9), _r("b", 0.7)]
    bm25 = [_r("z", 10.0), _r("y", 5.0), _r("x", 1.0)]
    merged = merge_retrievals(vec, bm25, alpha=0.0, top_k=5)
    ids = [r.chunk_id for r in merged]
    assert ids[:2] == ["z", "y"]


def test_merge_alpha_half_balances_modalities() -> None:
    """
    A chunk that appears top-1 in BOTH modalities beats one that
    appears top-1 in only one.
    """
    vec = [
        _r("agreed", 0.9),
        _r("vector_only", 0.88),  # high vector, missing from BM25
    ]
    bm25 = [
        _r("agreed", 10.0),
        _r("bm25_only", 9.0),  # high BM25, missing from vector
    ]
    merged = merge_retrievals(vec, bm25, alpha=0.5, top_k=3)
    assert merged[0].chunk_id == "agreed"


def test_merge_demotes_gradient_crafted_chunk(
) -> None:
    """
    PoisonedRAG-style attack scenario. A poisoned chunk has the
    top vector similarity (gradient-optimized embedding) but the
    text is nonsense, so its BM25 score is near zero. The merge
    demotes the poisoned chunk *out of rank 1* -- the
    load-bearing security property: an attacker's chunk no longer
    sits at the top of the LLM's retrieved context.

    Setup:
    - `poisoned`: vector top-1 (0.95), BM25 worst (0.1).
    - `legit_a`: vector mid (0.7), BM25 top (10.0).
    - `legit_b`: vector mid (0.6), BM25 second (8.0).

    Without the BM25 fusion, `poisoned` would be the LLM's #1
    retrieved chunk. After the fusion, `legit_a` rises to #1 and
    `poisoned` drops. The exact ordering of `legit_b` vs
    `poisoned` is sensitive to the min-max normalization tail
    (`legit_b`'s vector score is the worst of its modality, so
    it normalizes to 0.0), so we don't pin it -- this test
    documents the structural reality that BM25 fusion doesn't
    eliminate the poisoned chunk, it demotes it to a position
    where the LLM's "use the top-ranked source" heuristic no
    longer points at it.
    """
    vec = [
        _r("poisoned", 0.95),
        _r("legit_a", 0.7),
        _r("legit_b", 0.6),
    ]
    bm25 = [
        _r("legit_a", 10.0),
        _r("legit_b", 8.0),
        _r("poisoned", 0.1),
    ]
    merged = merge_retrievals(vec, bm25, alpha=0.5, top_k=3)
    ids = [r.chunk_id for r in merged]
    # Without hybrid, "poisoned" would be rank 0 (top vector).
    # After hybrid, the rank-1 spot is a legit chunk.
    assert ids[0] != "poisoned"
    assert ids[0] in {"legit_a", "legit_b"}
    # The poisoned chunk is demoted at least to rank 1 (out of
    # the top-1 spot it would otherwise have occupied).
    assert ids.index("poisoned") >= 1


def test_merge_top_k_caps_output() -> None:
    vec = [_r(f"v{i}", 1.0 - i * 0.1) for i in range(10)]
    bm25 = [_r(f"b{i}", 1.0 - i * 0.1) for i in range(10)]
    merged = merge_retrievals(vec, bm25, alpha=0.5, top_k=4)
    assert len(merged) == 4


def test_merge_carries_through_metadata() -> None:
    """
    The merged result inherits the vector-side metadata (richer
    in this codebase). BM25-only chunks carry their own metadata.
    """
    vec = [_r("v1", 0.9, source="paper_a.pdf", page=1)]
    bm25 = [_r("b1", 5.0, source="paper_b.pdf", page=2)]
    merged = merge_retrievals(vec, bm25, alpha=0.5, top_k=2)
    by_id = {r.chunk_id: r for r in merged}
    assert by_id["v1"].metadata == {"source": "paper_a.pdf", "page": 1}
    assert by_id["b1"].metadata == {"source": "paper_b.pdf", "page": 2}


def test_merge_records_merged_score() -> None:
    """The merged_score field is populated on every returned result."""
    merged = merge_retrievals(
        [_r("a", 0.9), _r("b", 0.5)],
        [_r("a", 5.0)],
        alpha=0.5,
    )
    for r in merged:
        assert r.merged_score is not None
        assert 0.0 <= r.merged_score <= 1.0


# -----------------------------------------------------------------
# Cross-module pin: the backend mirror matches the MCP server's
# canonical implementation. Drift fires here.
# -----------------------------------------------------------------


def test_merge_matches_mcp_server_module() -> None:
    """
    Pin the merge formula across the two duplicate copies.
    Imports both modules, runs identical inputs through each,
    and asserts the produced (chunk_id, merged_score) sequence
    is the same to within floating-point tolerance.

    If this test fires, you've drifted one copy of the merge
    formula and not the other. Update both.
    """
    try:
        from vista_mcp_server.hybrid_search import (
            RetrievalResult as ServerResult,
            merge_retrievals as server_merge,
        )
    except ImportError:
        pytest.skip("vista_mcp_server not importable in this environment")

    backend_vec = [
        _r("a", 0.9, source="p", page=1),
        _r("b", 0.7),
        _r("c", 0.5),
    ]
    backend_bm25 = [
        _r("b", 5.0),
        _r("c", 3.0),
        _r("d", 1.0),
    ]

    server_vec = [
        ServerResult(
            chunk_id=r.chunk_id,
            document=r.document,
            metadata=dict(r.metadata),
            score=r.score,
        )
        for r in backend_vec
    ]
    server_bm25 = [
        ServerResult(
            chunk_id=r.chunk_id,
            document=r.document,
            metadata=dict(r.metadata),
            score=r.score,
        )
        for r in backend_bm25
    ]

    backend_merged = merge_retrievals(backend_vec, backend_bm25, alpha=0.5, top_k=5)
    server_merged = server_merge(server_vec, server_bm25, alpha=0.5, top_k=5)

    backend_seq = [(r.chunk_id, r.merged_score) for r in backend_merged]
    server_seq = [(r.chunk_id, r.merged_score) for r in server_merged]

    assert len(backend_seq) == len(server_seq)
    for (bid, bs), (sid, ss) in zip(backend_seq, server_seq):
        assert bid == sid, f"chunk-id drift: backend={bid} server={sid}"
        # Floats can land slightly differently across copies if
        # somebody refactors; compare with tolerance.
        assert bs is not None and ss is not None
        assert abs(bs - ss) < 1e-12, f"merged_score drift: {bs} vs {ss}"
