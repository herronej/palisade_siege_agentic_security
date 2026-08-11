"""
Unit tests for the G3 evaluation harness.

The harness's job is to produce reproducible numbers under a
fixed seed; the load-bearing tests pin determinism and the
overall shape of the result tables. Spot-checks on the per-cell
numerics protect against silent drift if anyone refactors the
merge / vector / BM25 logic in the runner.

Tests are organized by module:

- `corpus.py`: seeded determinism, query/chunk shape.
- `attacks.py`: attack-profile output shape, RNG threading.
- `runner.py`: end-to-end matrix produces the right number of
  cells, ASR/FPR are in [0, 1], known ASR ceilings hold.
- `report.py`: headline + tables are present, no missing
  placeholders.
"""

from __future__ import annotations

import random

import pytest

from siege.eval import (
    ATTACK_PROFILES,
    EvaluationConfig,
    format_report,
    generate_attack,
    generate_corpus,
    run_evaluation,
)
from siege.eval.runner import (
    DEFAULT_CONFIGS,
    _build_retrieval_results,
    _simulate_bm25_score,
    _simulate_vector_similarity,
)


# -----------------------------------------------------------------
# corpus.py
# -----------------------------------------------------------------


def test_corpus_is_deterministic_under_fixed_seed() -> None:
    """Pin reproducibility: same seed -> same chunks + queries."""
    a = generate_corpus(seed=42)
    b = generate_corpus(seed=42)
    assert a.chunks == b.chunks
    assert a.queries == b.queries
    assert a.seed == b.seed == 42


def test_corpus_changes_with_different_seed() -> None:
    """Different seeds -> different content (at least at the chunk_id level)."""
    a = generate_corpus(seed=42)
    b = generate_corpus(seed=43)
    # We don't pin exact difference but at least one query should
    # differ at this corpus size.
    a_queries = {q.query for q in a.queries}
    b_queries = {q.query for q in b.queries}
    assert a_queries != b_queries


def test_corpus_default_counts() -> None:
    corpus = generate_corpus(seed=42)
    assert len(corpus.chunks) == 100
    assert len(corpus.queries) == 20


def test_corpus_custom_counts_respected() -> None:
    corpus = generate_corpus(seed=42, n_chunks=50, n_queries=10)
    assert len(corpus.chunks) == 50
    assert len(corpus.queries) == 10


def test_corpus_rejects_more_queries_than_chunks() -> None:
    with pytest.raises(ValueError, match="n_queries"):
        generate_corpus(seed=42, n_chunks=5, n_queries=10)


def test_corpus_gold_chunks_exist_in_corpus() -> None:
    """Each query's gold_chunk_id is the id of an actual chunk."""
    corpus = generate_corpus(seed=42)
    chunk_ids = {c.chunk_id for c in corpus.chunks}
    for q in corpus.queries:
        assert q.gold_chunk_id in chunk_ids


def test_corpus_gold_chunks_are_distinct_across_queries() -> None:
    """No two queries share a gold chunk."""
    corpus = generate_corpus(seed=42)
    gold_ids = [q.gold_chunk_id for q in corpus.queries]
    assert len(set(gold_ids)) == len(gold_ids)


# -----------------------------------------------------------------
# attacks.py
# -----------------------------------------------------------------


def test_attack_profiles_registry_has_expected_names() -> None:
    """Pin the registry's keys so a future refactor that drops a
    family fires here."""
    assert set(ATTACK_PROFILES.keys()) == {
        "poisoned_rag",
        "agent_poison",
        "memory_graft",
        "joint_optimization",
    }


def test_attack_generator_produces_n_chunks() -> None:
    """`generate_attack(..., n=k)` returns exactly k chunks."""
    corpus = generate_corpus(seed=42)
    query = corpus.queries[0]
    rng = random.Random(0)
    for profile_name in ATTACK_PROFILES:
        out = generate_attack(profile_name, query, rng, n=3)
        assert len(out) == 3
        for p in out:
            assert p.family == profile_name
            assert p.target_query == query.query


def test_attack_chunks_are_deterministic_under_rng() -> None:
    """Same RNG state -> same poisoned chunks."""
    corpus = generate_corpus(seed=42)
    query = corpus.queries[0]
    a = generate_attack("agent_poison", query, random.Random(0), n=2)
    b = generate_attack("agent_poison", query, random.Random(0), n=2)
    assert [p.text for p in a] == [p.text for p in b]


def test_poisoned_rag_chunks_have_no_query_token_overlap() -> None:
    """Gradient-gibberish should NOT contain query tokens. This is
    what BM25 can demote."""
    corpus = generate_corpus(seed=42)
    query = corpus.queries[0]
    poisoned = generate_attack("poisoned_rag", query, random.Random(0), n=5)
    q_tokens = set(query.query.lower().split())
    for p in poisoned:
        c_tokens = set(p.text.lower().split())
        # Random gibberish vocabulary; the intersection with the
        # query's natural-English tokens should be zero.
        assert not (q_tokens & c_tokens), (
            f"PoisonedRAG chunk leaked query tokens: {q_tokens & c_tokens}"
        )


def test_joint_optimization_chunks_DO_overlap_query_tokens() -> None:
    """
    Joint-optimization is the adaptive attack: it must include
    query-token overlap so BM25 scores it well. Pinning this
    keeps the attack honest -- a future refactor that
    accidentally drops the overlap would silently improve the
    headline.
    """
    corpus = generate_corpus(seed=42)
    query = corpus.queries[0]
    poisoned = generate_attack("joint_optimization", query, random.Random(0), n=3)
    q_tokens = set(query.query.lower().split())
    for p in poisoned:
        c_tokens = set(p.text.lower().split())
        # At minimum the salt name and a property word should be
        # present.
        assert (q_tokens & c_tokens), (
            "JointOptimization chunk has no query-token overlap; "
            "BM25 will trivially demote it, which defeats the point "
            "of the attack profile"
        )


def test_attack_simulated_vector_similarity_is_high() -> None:
    """Every attack profile claims simulated vector similarity > 0.8."""
    corpus = generate_corpus(seed=42)
    query = corpus.queries[0]
    rng = random.Random(0)
    for profile_name in ATTACK_PROFILES:
        chunks = generate_attack(profile_name, query, rng, n=1)
        assert chunks[0].simulated_vector_similarity > 0.8


# -----------------------------------------------------------------
# runner.py -- primitives
# -----------------------------------------------------------------


def test_simulate_vector_similarity_in_unit_interval() -> None:
    """Jaccard proxy stays in [0, 1]."""
    s = _simulate_vector_similarity("salt corrosion", "salt corrosion happens")
    assert 0.0 <= s <= 1.0


def test_simulate_vector_similarity_is_zero_for_disjoint() -> None:
    """No shared tokens -> 0."""
    assert _simulate_vector_similarity("foo bar", "baz qux") == 0.0


def test_simulate_bm25_score_zero_for_no_overlap() -> None:
    """No query terms in chunk -> 0."""
    assert _simulate_bm25_score("salt", "completely different text") == 0.0


def test_simulate_bm25_score_positive_for_overlap() -> None:
    """Query terms present -> positive."""
    score = _simulate_bm25_score("salt corrosion", "salt corrosion in pipes")
    assert score > 0.0


def test_build_retrieval_results_ranks_vector_descending() -> None:
    """Vector list comes out in vector-score descending order."""
    corpus = generate_corpus(seed=42, n_chunks=20, n_queries=4)
    vec, bm25 = _build_retrieval_results(
        corpus.queries[0].query, list(corpus.chunks), [], top_k_per_modality=10,
    )
    for i in range(len(vec) - 1):
        assert vec[i].score >= vec[i + 1].score
    for i in range(len(bm25) - 1):
        assert bm25[i].score >= bm25[i + 1].score


# -----------------------------------------------------------------
# runner.py -- end to end
# -----------------------------------------------------------------


def test_run_evaluation_produces_expected_cell_count() -> None:
    """4 attack profiles x 4 configs = 16 ASR cells; 4 FPR cells."""
    corpus = generate_corpus(seed=42, n_chunks=30, n_queries=10)
    result = run_evaluation(corpus, rng_seed=42)
    assert len(result.asr_cells) == len(ATTACK_PROFILES) * len(DEFAULT_CONFIGS)
    assert len(result.fpr_cells) == len(DEFAULT_CONFIGS)


def test_run_evaluation_asr_in_unit_interval() -> None:
    """Every ASR cell is in [0, 1]."""
    corpus = generate_corpus(seed=42, n_chunks=30, n_queries=10)
    result = run_evaluation(corpus, rng_seed=42)
    for cell in result.asr_cells:
        assert 0.0 <= cell.asr_top1 <= 1.0
        assert 0.0 <= cell.asr_top5 <= 1.0
        # top-1 ASR is bounded above by top-5 by definition.
        assert cell.asr_top1 <= cell.asr_top5


def test_run_evaluation_fpr_in_unit_interval() -> None:
    corpus = generate_corpus(seed=42, n_chunks=30, n_queries=10)
    result = run_evaluation(corpus, rng_seed=42)
    for fpr in result.fpr_cells:
        assert 0.0 <= fpr.fpr <= 1.0


def test_run_evaluation_records_corpus_and_rng_seeds() -> None:
    corpus = generate_corpus(seed=7, n_chunks=30, n_queries=5)
    result = run_evaluation(corpus, rng_seed=11)
    assert result.corpus_seed == 7
    assert result.rng_seed == 11


def test_vector_only_baseline_has_zero_fpr() -> None:
    """
    The vector-only configuration is the FPR baseline -- by
    construction `_evaluate_fpr` compares against vector-only's
    top-1, so vector-only itself never regresses.
    """
    corpus = generate_corpus(seed=42)
    result = run_evaluation(corpus, rng_seed=42)
    for fpr in result.fpr_cells:
        if fpr.config_name == "vector-only":
            assert fpr.fpr == 0.0


def test_vector_only_baseline_has_high_poisoned_rag_asr() -> None:
    """
    Sanity check: vector-only retrieval is *vulnerable* to
    PoisonedRAG (the attacker's whole point is to compromise
    vector similarity). The headline number for vector-only on
    PoisonedRAG should be near 100% top-1; if it's not, the
    attack profile or the simulation is broken.
    """
    corpus = generate_corpus(seed=42)
    result = run_evaluation(corpus, rng_seed=42)
    pr_vec_only = next(
        c for c in result.asr_cells
        if c.attack_family == "poisoned_rag" and c.config_name == "vector-only"
    )
    assert pr_vec_only.asr_top1 >= 0.9, (
        f"vector-only PoisonedRAG ASR top-1 unexpectedly low: "
        f"{pr_vec_only.asr_top1:.2f}; check that "
        f"simulated_vector_similarity is dominating the ranking"
    )


def test_hybrid_substantially_reduces_poisoned_rag_asr() -> None:
    """
    The headline Semantic Chameleon result: hybrid retrieval
    drives gradient-guided ASR (PoisonedRAG) to near zero. We
    pin the work-item AC of < 5% here so a regression in the
    merge logic or the BM25 proxy fires this test.
    """
    corpus = generate_corpus(seed=42)
    result = run_evaluation(corpus, rng_seed=42)
    pr_hybrid = next(
        c for c in result.asr_cells
        if c.attack_family == "poisoned_rag" and c.config_name == "hybrid"
    )
    assert pr_hybrid.asr_top1 < 0.05, (
        f"hybrid PoisonedRAG ASR top-1 above 5%: {pr_hybrid.asr_top1:.2f}; "
        f"the Semantic Chameleon defense has regressed"
    )


def test_joint_optimization_is_hardest_for_hybrid() -> None:
    """
    Honest-failure pin: JointOptimization's best ASR is worse
    than PoisonedRAG's best. The adaptive attack defeats the
    hybrid defense as documented in the report.
    """
    corpus = generate_corpus(seed=42)
    result = run_evaluation(corpus, rng_seed=42)
    pr_best = min(
        c.asr_top1 for c in result.asr_cells if c.attack_family == "poisoned_rag"
    )
    joint_best = min(
        c.asr_top1 for c in result.asr_cells
        if c.attack_family == "joint_optimization"
    )
    assert joint_best >= pr_best, (
        f"JointOptimization best ASR ({joint_best:.2f}) should be at "
        f"least PoisonedRAG best ({pr_best:.2f}); if not, the "
        f"adaptive-attack model is too weak"
    )


def test_run_evaluation_with_custom_configs() -> None:
    """Configurable: a caller can pass a different config tuple."""
    corpus = generate_corpus(seed=42, n_chunks=20, n_queries=5)
    custom = (
        EvaluationConfig(name="only-vector", alpha=1.0, slow_tier_enabled=False),
        EvaluationConfig(name="custom-blend", alpha=0.7, slow_tier_enabled=False),
    )
    result = run_evaluation(corpus, configs=custom, rng_seed=42)
    assert len(result.fpr_cells) == 2
    assert {f.config_name for f in result.fpr_cells} == {"only-vector", "custom-blend"}
    assert len(result.asr_cells) == len(ATTACK_PROFILES) * 2


# -----------------------------------------------------------------
# report.py
# -----------------------------------------------------------------


def test_format_report_includes_all_required_sections() -> None:
    """
    Pin the structural shape of the markdown report. A
    refactor that drops a section fires this test before the
    docs/ file goes out of date.
    """
    corpus = generate_corpus(seed=42, n_chunks=30, n_queries=5)
    result = run_evaluation(corpus, rng_seed=42)
    md = format_report(result)
    assert "# G3 evaluation" in md
    assert "## Headline" in md
    assert "## Methodology" in md
    assert "## ASR results" in md
    assert "## False-positive rate" in md
    assert "## Honest limits" in md
    # Each attack family's display name appears.
    for profile in ATTACK_PROFILES.values():
        assert profile.display_name in md


def test_format_report_headline_includes_gradient_guided_number() -> None:
    """The work-item AC asks for the gradient-guided ASR; the
    headline must surface it explicitly."""
    corpus = generate_corpus(seed=42)
    result = run_evaluation(corpus, rng_seed=42)
    md = format_report(result)
    # Headline phrasing pinned so a future change is intentional.
    assert "Gradient-guided ASR top-1" in md
    assert "PoisonedRAG" in md


def test_format_report_is_pure_text() -> None:
    """No leftover Python-format placeholders in the output."""
    corpus = generate_corpus(seed=42, n_chunks=20, n_queries=5)
    result = run_evaluation(corpus, rng_seed=42)
    md = format_report(result)
    # `format_report` uses f-strings; an unfilled `{key}` would be a
    # template bug. We don't catch every escape but the common
    # signature is enough.
    assert "{" not in md.replace("`{", "")  # allow markdown code-fence braces
