"""
G3 evaluation runner.

Threads the synthetic corpus + attack profiles through the
backend's hybrid-retrieval merge logic and produces structured
per-attack, per-configuration ASR / FPR numbers. Outputs an
`EvaluationResult` the `report.py` module formats into markdown.

## Configuration matrix

Four configurations, the same set the work item names:

- **vector-only, no slow tier** (`alpha=1.0`, Q-LLM off): the
  baseline. This is what VISTA does today.
- **vector-only + slow tier**: vector retrieval, then simulated
  Q-LLM scan demotes detected chunks. The Q-LLM detection rate
  is per-family (see `attacks.py:ATTACK_PROFILES`).
- **hybrid, no slow tier** (`alpha=0.5`): the Semantic Chameleon
  defense in its cheap form.
- **hybrid + slow tier**: the full G3 path.

## Metrics

For each (attack family, configuration) cell:

- **ASR top-1** = fraction of attacked queries where the
  poisoned chunk lands at rank 1.
- **ASR top-5** = fraction where the poisoned chunk lands in
  the top 5.

For the benign workload (no attacks):

- **FPR** = fraction of benign queries where the gold chunk
  drops out of rank 1 under hybrid retrieval compared to
  vector-only. Negative numbers (hybrid is BETTER than
  vector-only) are clamped to zero in the report.

## Simulated vector similarity

We don't have an embedding model. For benign chunks we use a
Jaccard-style token overlap (a cheap proxy for what a real
SentenceTransformer would produce -- correlated for short
domain-specific text). For poisoned chunks we use the
`simulated_vector_similarity` field directly, modeling the
attacker's gradient-optimization gain.

The proxy is honest about what it can and can't say: it captures
the *gap* between BM25 and vector signal (which is what hybrid
retrieval exploits) but not the absolute calibration. A real
deployment's BM25-vs-vector tradeoff will differ in detail but
not in shape.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from palisade.gates.g3_hybrid import (
    MirroredRetrievalResult,
    merge_retrievals,
)
from siege.eval.attacks import (
    ATTACK_PROFILES,
    AttackProfile,
    PoisonedChunk,
    generate_attack,
)
from siege.eval.corpus import BenignQuery, Chunk, Corpus


# -----------------------------------------------------------------
# Config + result dataclasses
# -----------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationConfig:
    """
    A single G3 configuration to evaluate.

    Fields:
        name: short identifier for the column in the report.
        alpha: weight on the vector modality (1.0 = vector-only,
            0.5 = the spec's hybrid default).
        slow_tier_enabled: when True, the Q-LLM scan runs on
            retrieved chunks and demotes those it (probabilistic-
            ally) detects. Per-attack detection rate from
            `AttackProfile.simulated_qllm_detection_rate`.
    """

    name: str
    alpha: float
    slow_tier_enabled: bool


# The four configurations the work item names. Pinned here as
# module constants so tests, runners, and the report all read
# from the same source.
DEFAULT_CONFIGS: tuple[EvaluationConfig, ...] = (
    EvaluationConfig(name="vector-only", alpha=1.0, slow_tier_enabled=False),
    EvaluationConfig(name="vector + slow", alpha=1.0, slow_tier_enabled=True),
    EvaluationConfig(name="hybrid", alpha=0.5, slow_tier_enabled=False),
    EvaluationConfig(name="hybrid + slow", alpha=0.5, slow_tier_enabled=True),
)


@dataclass(frozen=True)
class CellResult:
    """ASR for one (attack family, configuration) cell."""

    attack_family: str
    config_name: str
    n_attacks: int
    asr_top1: float
    asr_top5: float


@dataclass(frozen=True)
class FprResult:
    """FPR for one configuration against the benign workload."""

    config_name: str
    n_queries: int
    fpr: float


@dataclass(frozen=True)
class EvaluationResult:
    """
    Output of `run_evaluation`. The report formatter consumes
    this directly.

    Fields:
        corpus_seed: the seed used (recorded for report
            reproducibility).
        configs: the configurations evaluated.
        attack_profiles: the attack profiles evaluated.
        asr_cells: per-(attack, config) ASR cells. Indexed as a
            list rather than a nested dict so the report
            formatter can group flexibly.
        fpr_cells: per-config FPR cells.
        n_corpus_chunks: bookkeeping for the report.
        n_benign_queries: bookkeeping for the report.
        rng_seed: the seed forwarded into the attack generators.
    """

    corpus_seed: int
    configs: tuple[EvaluationConfig, ...]
    attack_profiles: tuple[AttackProfile, ...]
    asr_cells: tuple[CellResult, ...]
    fpr_cells: tuple[FprResult, ...]
    n_corpus_chunks: int
    n_benign_queries: int
    rng_seed: int


# -----------------------------------------------------------------
# Retrieval primitives (simulated)
# -----------------------------------------------------------------


# English stopwords filtered from both modalities' stand-ins so
# short benign queries (e.g. "What is the density of FLiBe at
# 873 K?") don't have their score dominated by common-word
# overlap. Real BM25 with IDF weighting downweights these
# automatically; the cheap stand-ins fake the same effect by
# filtering at tokenization time. Real SentenceTransformer
# similarity is also less sensitive to stopwords than raw
# Jaccard.
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for",
    "from", "has", "have", "he", "in", "is", "it", "its", "of",
    "on", "or", "she", "that", "the", "they", "this", "to",
    "was", "were", "what", "when", "where", "which", "who",
    "with",
})


def _tokenize(text: str) -> set[str]:
    """Cheap tokenizer: lowercase + whitespace split, stopwords
    filtered. Sets so we can compute Jaccard quickly."""
    return {t for t in text.lower().split() if t not in _STOPWORDS}


def _simulate_vector_similarity(query: str, chunk_text: str) -> float:
    """
    Token-Jaccard proxy (over content words; stopwords filtered)
    for what a real SentenceTransformer would score. Correlated
    with embedding similarity on short domain-specific text; we
    use it for benign chunks so the benign-vs-poisoned gap
    matches what an attacker would face in a real deployment.

    Range [0, 1].
    """
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(chunk_text)
    if not q_tokens or not c_tokens:
        return 0.0
    intersection = len(q_tokens & c_tokens)
    union = len(q_tokens | c_tokens)
    return intersection / union if union > 0 else 0.0


def _simulate_bm25_score(query: str, chunk_text: str) -> float:
    """
    Cheap BM25 stand-in for the runner -- counts query *content*
    terms (stopwords filtered) appearing in the chunk weighted by
    inverse chunk length. Faithful enough to the Okapi-BM25 +
    IDF signal for the small corpora in this harness; the
    production code path uses the full BM25Okapi in mcp-server.
    We don't reuse the full implementation here because we want
    the runner to stay in the backend package and have no
    mcp-server dependency.
    """
    q_tokens = _tokenize(query)
    c_tokens = [t for t in chunk_text.lower().split() if t not in _STOPWORDS]
    if not q_tokens or not c_tokens:
        return 0.0
    matches = sum(1 for t in c_tokens if t in q_tokens)
    # Length normalization so a poisoned chunk that pads with
    # query terms doesn't get an unbounded score.
    return matches / (1.0 + 0.1 * len(c_tokens))


# -----------------------------------------------------------------
# Per-query evaluation
# -----------------------------------------------------------------


def _build_retrieval_results(
    query: str,
    chunks: list[Chunk],
    poisoned: list[PoisonedChunk],
    top_k_per_modality: int,
) -> tuple[list[MirroredRetrievalResult], list[MirroredRetrievalResult]]:
    """
    Build (vector_results, bm25_results) for a query against the
    union of corpus chunks + poisoned chunks.

    Vector scoring:
    - Corpus chunks: Jaccard proxy.
    - Poisoned chunks: the attacker's pre-baked
      `simulated_vector_similarity` (models the gradient gain).

    BM25 scoring:
    - Same simple BM25 stand-in for both -- no attacker
      manipulation because BM25 reads the chunk text the gates
      see.

    Returns the top-K per modality sorted descending. The merge
    function takes care of normalization.
    """
    all_items: list[tuple[str, str, float, float]] = []
    # (chunk_id, text, vector_score, bm25_score)
    for chunk in chunks:
        vec = _simulate_vector_similarity(query, chunk.text)
        bm25 = _simulate_bm25_score(query, chunk.text)
        all_items.append((chunk.chunk_id, chunk.text, vec, bm25))
    for p in poisoned:
        vec = p.simulated_vector_similarity
        bm25 = _simulate_bm25_score(query, p.text)
        all_items.append((p.chunk_id, p.text, vec, bm25))

    # Vector ranking.
    vec_sorted = sorted(all_items, key=lambda t: t[2], reverse=True)[:top_k_per_modality]
    vector_results = [
        MirroredRetrievalResult(chunk_id=cid, document=text, score=v)
        for cid, text, v, _ in vec_sorted
    ]

    # BM25 ranking.
    bm25_sorted = sorted(all_items, key=lambda t: t[3], reverse=True)[:top_k_per_modality]
    bm25_results = [
        MirroredRetrievalResult(chunk_id=cid, document=text, score=b)
        for cid, text, _, b in bm25_sorted
    ]
    return vector_results, bm25_results


def _apply_slow_tier(
    merged: list[MirroredRetrievalResult],
    poisoned_ids: set[str],
    detection_rates: dict[str, float],
    poisoned_id_to_family: dict[str, str],
    rng: random.Random,
) -> list[MirroredRetrievalResult]:
    """
    Simulate the Q-LLM slow-tier scan.

    For each merged result that is a poisoned chunk, the Q-LLM
    detects it with probability `detection_rates[family]` and
    demotes it (removes from the list). Non-poisoned chunks are
    never affected -- this is the optimistic-for-Q-LLM case (no
    false positives). The report says this loud.
    """
    survivors: list[MirroredRetrievalResult] = []
    for r in merged:
        if r.chunk_id in poisoned_ids:
            family = poisoned_id_to_family[r.chunk_id]
            detect = rng.random() < detection_rates.get(family, 0.0)
            if detect:
                continue  # detected and demoted
        survivors.append(r)
    return survivors


def _evaluate_one_attack(
    profile: AttackProfile,
    config: EvaluationConfig,
    corpus: Corpus,
    rng: random.Random,
    top_k_per_modality: int,
    top_k_final: int,
) -> CellResult:
    """
    Compute ASR for one (attack family, config) cell.

    For each benign query in the corpus, generate one poisoned
    chunk from `profile`, run retrieval, and check whether the
    poisoned chunk surfaces at rank 1 or in top-5.

    ASR is the fraction of queries where the poisoned chunk
    appeared.
    """
    n_top1 = 0
    n_top5 = 0
    detection_rates = {
        name: p.simulated_qllm_detection_rate
        for name, p in ATTACK_PROFILES.items()
    }

    for query_obj in corpus.queries:
        # One poisoned chunk per query.
        poisoned = profile.generator(query_obj, rng, 1)
        poisoned_ids = {p.chunk_id for p in poisoned}
        poisoned_id_to_family = {p.chunk_id: p.family for p in poisoned}

        vector_results, bm25_results = _build_retrieval_results(
            query_obj.query,
            list(corpus.chunks),
            poisoned,
            top_k_per_modality,
        )

        if config.alpha == 1.0:
            # Vector-only: skip the merge math and use the vector
            # ranking directly. (The merge function would produce
            # the same ordering at alpha=1.0 but we model the
            # "vector-only path" as the existing code path -- no
            # BM25 retrieval, no merge step.)
            merged = vector_results[:top_k_final]
        else:
            merged = merge_retrievals(
                vector_results,
                bm25_results,
                alpha=config.alpha,
                top_k=top_k_final,
            )

        if config.slow_tier_enabled:
            merged = _apply_slow_tier(
                merged,
                poisoned_ids,
                detection_rates,
                poisoned_id_to_family,
                rng,
            )

        ids = [r.chunk_id for r in merged]
        if ids and ids[0] in poisoned_ids:
            n_top1 += 1
        if any(cid in poisoned_ids for cid in ids[:5]):
            n_top5 += 1

    n = len(corpus.queries)
    return CellResult(
        attack_family=profile.name,
        config_name=config.name,
        n_attacks=n,
        asr_top1=n_top1 / n if n else 0.0,
        asr_top5=n_top5 / n if n else 0.0,
    )


def _evaluate_fpr(
    config: EvaluationConfig,
    corpus: Corpus,
    top_k_per_modality: int,
    top_k_final: int,
) -> FprResult:
    """
    Measure the false-positive rate against the benign workload.

    For each benign query, compute:
    - vector-only top-1 (baseline)
    - this-config top-1
    FPR = fraction of queries where this-config demoted the
    gold chunk out of top-1 relative to the baseline.

    A config that NEVER demotes the gold chunk gets FPR = 0. A
    config that always does gets FPR = 1.
    """
    regressed = 0
    n = len(corpus.queries)
    for query_obj in corpus.queries:
        # No poisoned chunks for the benign-workload measurement.
        vec, bm25 = _build_retrieval_results(
            query_obj.query, list(corpus.chunks), [], top_k_per_modality,
        )
        baseline_top1 = vec[0].chunk_id if vec else None
        if config.alpha == 1.0:
            this_top1 = vec[0].chunk_id if vec else None
        else:
            merged = merge_retrievals(
                vec, bm25, alpha=config.alpha, top_k=top_k_final,
            )
            this_top1 = merged[0].chunk_id if merged else None
        if baseline_top1 == query_obj.gold_chunk_id and this_top1 != query_obj.gold_chunk_id:
            regressed += 1
    return FprResult(
        config_name=config.name,
        n_queries=n,
        fpr=regressed / n if n else 0.0,
    )


# -----------------------------------------------------------------
# Public runner
# -----------------------------------------------------------------


def run_evaluation(
    corpus: Corpus,
    *,
    configs: tuple[EvaluationConfig, ...] = DEFAULT_CONFIGS,
    rng_seed: int = 42,
    top_k_per_modality: int = 20,
    top_k_final: int = 5,
) -> EvaluationResult:
    """
    Run the full evaluation matrix against `corpus`.

    Args:
        corpus: from `corpus.generate_corpus(seed=...)`.
        configs: the configurations to measure. Defaults to the
            four configurations the work item names.
        rng_seed: seeds the attack generators and the slow-tier
            detection coin flips. Pinned to 42 in the shipped
            report.
        top_k_per_modality: over-retrieve factor (each modality
            returns this many before the merge clips). Matches
            the MCP server's behavior.
        top_k_final: number of chunks the merged result returns.
            Matches the `rag_search` default.

    Returns:
        An `EvaluationResult` the report formatter consumes.
    """
    rng = random.Random(rng_seed)

    asr_cells: list[CellResult] = []
    for profile in ATTACK_PROFILES.values():
        for config in configs:
            cell = _evaluate_one_attack(
                profile=profile,
                config=config,
                corpus=corpus,
                rng=rng,
                top_k_per_modality=top_k_per_modality,
                top_k_final=top_k_final,
            )
            asr_cells.append(cell)

    fpr_cells: list[FprResult] = []
    for config in configs:
        fpr_cells.append(
            _evaluate_fpr(
                config=config,
                corpus=corpus,
                top_k_per_modality=top_k_per_modality,
                top_k_final=top_k_final,
            )
        )

    return EvaluationResult(
        corpus_seed=corpus.seed,
        configs=tuple(configs),
        attack_profiles=tuple(ATTACK_PROFILES.values()),
        asr_cells=tuple(asr_cells),
        fpr_cells=tuple(fpr_cells),
        n_corpus_chunks=len(corpus.chunks),
        n_benign_queries=len(corpus.queries),
        rng_seed=rng_seed,
    )
