"""
Unit tests for the G3 embedding-cluster anomaly detector
(`gates/g3_anomaly.py`).

Covers all four acceptance criteria of the work item
`Add embedding-cluster anomaly detector for
AgentPoison/PoisonedRAG attack detection`:

1. Detector consumes ChromaDB-shaped embeddings (numpy
   ndarray, list of lists, etc.).
2. z-score thresholding flags outliers; threshold is a
   per-call parameter (per-deployment configurable).
3. Synthetic AgentPoison-style outlier embedding: detector
   flags.
4. Natural-norm crafted embedding: detector misses, and this
   is documented in the test as the expected gap that the
   semantic-entropy probe covers.

Test fixtures use a seeded `np.random.default_rng` so the
assertions are deterministic. Embedding dimensions are kept
small (16-128) so the tests are fast; the algebra doesn't change
at higher dimensions but the test wall-clock would.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from palisade.config import PalisadeSettings
from palisade.gates.g3_anomaly import (
    AnomalyResult,
    _coerce_embeddings,
    _l2_normalize,
    detect_embedding_anomalies,
)


# -----------------------------------------------------------------
# Fixtures -- synthetic embeddings
# -----------------------------------------------------------------


def _normal_cluster(
    n: int,
    *,
    center: np.ndarray,
    sigma: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate `n` embeddings drawn from a Gaussian around `center`
    with std `sigma`, then L2-normalize each row so the result
    looks like real sentence-transformer outputs (unit-norm
    embeddings on a hypersphere).

    Tight `sigma` produces a tight cluster; loosen it to
    simulate corpus-natural spread.
    """
    rng = np.random.default_rng(seed)
    d = center.shape[0]
    points = center + rng.standard_normal((n, d)) * sigma
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _spread_unit_vectors(n: int, d: int, *, seed: int = 0) -> np.ndarray:
    """
    Generate `n` unit vectors drawn uniformly on the d-dimensional
    hypersphere. Models a maximally diverse retrieval batch (the
    legitimate-corpus baseline used for the PoisonedRAG cluster
    test).
    """
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n, d))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


# -----------------------------------------------------------------
# AC 1 -- input-shape acceptance
# -----------------------------------------------------------------


def test_accepts_numpy_ndarray_input() -> None:
    """ChromaDB returns numpy arrays; the detector accepts them directly."""
    embeddings = np.eye(4, dtype=np.float64)
    result = detect_embedding_anomalies(embeddings)
    assert isinstance(result, AnomalyResult)
    assert len(result.z_scores) == 4


def test_accepts_list_of_lists_input() -> None:
    """The JSON wire format -- list[list[float]] -- is accepted."""
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    result = detect_embedding_anomalies(embeddings)
    assert isinstance(result, AnomalyResult)
    assert len(result.z_scores) == 4


def test_accepts_chromadb_style_batch_payload() -> None:
    """
    ChromaDB's `query(..., include=["embeddings"])` returns a
    `list[list[list[float]]]` whose outer length is the number of
    queries (1 in our setup). The G3 gate will index into
    `[0]` before passing here, so the contract for THIS function
    is the inner `list[list[float]]`.
    """
    chromadb_payload = [
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    ]
    # Simulate the gate's indexing.
    result = detect_embedding_anomalies(chromadb_payload[0])
    assert isinstance(result, AnomalyResult)


def test_rejects_1d_input_with_clear_message() -> None:
    """1-D input is a programming error -- raise with guidance."""
    with pytest.raises(ValueError, match="2-D"):
        detect_embedding_anomalies(np.array([1.0, 0.0, 0.0]))


def test_rejects_3d_input() -> None:
    """A `list[list[list[float]]]` (the raw ChromaDB return) is
    not what we want -- the gate should index first."""
    with pytest.raises(ValueError, match="2-D"):
        detect_embedding_anomalies(np.zeros((1, 4, 3)))


def test_rejects_non_numeric_input() -> None:
    """Strings or other non-float entries fail at coercion."""
    with pytest.raises(ValueError):
        detect_embedding_anomalies([["a", "b"], ["c", "d"]])


def test_rejects_non_positive_z_threshold() -> None:
    """Zero / negative threshold is a config bug -- raise loudly."""
    embeddings = np.eye(3)
    with pytest.raises(ValueError, match="positive"):
        detect_embedding_anomalies(embeddings, z_threshold=0.0)
    with pytest.raises(ValueError, match="positive"):
        detect_embedding_anomalies(embeddings, z_threshold=-1.5)


# -----------------------------------------------------------------
# AC 3 -- AgentPoison-style outlier is flagged
# -----------------------------------------------------------------


def test_agentpoison_style_outlier_is_flagged_at_default_threshold() -> None:
    """
    AC3: a single chunk far from the natural cluster (the
    AgentPoison constrained-optimization-trigger signature) is
    flagged at the default z_threshold=3.0.

    Setup: 10 corpus-natural chunks tightly clustered around
    [1, 0, ..., 0]; one attacker chunk near [-1, 0, ..., 0]. The
    attacker's mean pairwise distance is much larger than the
    cluster's, so |z| > 3.0 for the attacker.
    """
    d = 32
    center_legit = np.zeros(d)
    center_legit[0] = 1.0

    legit = _normal_cluster(10, center=center_legit, sigma=0.03, seed=1)

    # Attacker far away on the antipodal direction.
    attacker_center = np.zeros(d)
    attacker_center[0] = -1.0
    attacker = (attacker_center / np.linalg.norm(attacker_center)).reshape(1, -1)

    batch = np.vstack([legit, attacker])
    result = detect_embedding_anomalies(batch, z_threshold=3.0)

    assert result.ran is True
    # The attacker is the last row.
    assert 10 in result.flagged_indices, (
        f"expected attacker (index 10) flagged; got {result.flagged_indices}, "
        f"z_scores={result.z_scores}"
    )
    # The attacker's z-score is the largest.
    z_attacker = result.z_scores[10]
    z_others = [result.z_scores[i] for i in range(10)]
    assert abs(z_attacker) > max(abs(z) for z in z_others)


def test_agentpoison_outlier_reason_mentions_flagged_count() -> None:
    """The reason string carries the count for the audit trail."""
    d = 16
    center = np.zeros(d)
    center[0] = 1.0
    legit = _normal_cluster(8, center=center, sigma=0.02, seed=2)

    attacker = np.zeros((1, d))
    attacker[0, 5] = 1.0  # orthogonal direction
    batch = np.vstack([legit, attacker])

    result = detect_embedding_anomalies(batch, z_threshold=2.0)
    assert result.flagged_indices  # at least one flagged
    assert "flagged" in result.reason
    assert str(len(result.flagged_indices)) in result.reason


# -----------------------------------------------------------------
# AC 4 -- natural-norm crafted embedding is NOT flagged
# (documented gap)
# -----------------------------------------------------------------


def test_natural_norm_crafted_attack_is_missed_documented_gap() -> None:
    """
    AC4: an attacker who crafts a chunk inside the natural
    distribution slips past this detector. This is the documented
    gap from the work item -- the semantic-entropy probe (separate
    issue) is the intended coverage. We pin the limit here so a
    future refactor that accidentally claims to defend against
    natural-norm attacks breaks this test.

    Setup: 10 corpus-natural chunks AND 1 attacker chunk all
    drawn from the same Gaussian. The detector sees them as
    statistically indistinguishable.
    """
    d = 64
    center = np.zeros(d)
    center[0] = 1.0
    # Use a fresh seed so the attacker is sampled from the same
    # distribution but is a distinct sample (not identical).
    legit = _normal_cluster(10, center=center, sigma=0.05, seed=3)
    attacker = _normal_cluster(1, center=center, sigma=0.05, seed=4)

    batch = np.vstack([legit, attacker])
    result = detect_embedding_anomalies(batch, z_threshold=3.0)

    assert result.ran is True
    # The expected behavior: NO flag. The attacker is
    # indistinguishable from the cluster in mean pairwise
    # distance.
    assert result.flagged_indices == (), (
        f"natural-norm attack unexpectedly flagged; "
        f"flagged_indices={result.flagged_indices}, "
        f"z_scores={result.z_scores}"
    )
    # And every chunk's |z| is moderate -- below 3.0.
    assert all(abs(z) < 3.0 for z in result.z_scores)


# -----------------------------------------------------------------
# PoisonedRAG cluster signature: not a strict AC but pinned so
# the cluster-side mechanism is exercised
# -----------------------------------------------------------------


def test_poisonedrag_minority_cluster_has_lower_z_scores_than_diverse_chunks() -> None:
    """
    PoisonedRAG's tight injected cluster is harder than
    AgentPoison's lone outlier: the cluster chunks have lower
    mean pairwise distances than the diverse chunks, manifesting
    as *negative* z-scores. Pinned here as a directional signal;
    whether they cross threshold 3.0 depends on cluster vs.
    legitimate ratio (the dominant-cluster degenerate case is
    documented in the module docstring as out of scope).

    Setup: 9 maximally-diverse legitimate chunks + 3 tightly-
    clustered poisoned chunks. The 3 cluster chunks should have
    smaller mean pairwise distance than the average.
    """
    d = 64
    legit = _spread_unit_vectors(9, d, seed=5)

    center = np.zeros(d)
    center[0] = 1.0
    cluster = _normal_cluster(3, center=center, sigma=0.01, seed=6)

    batch = np.vstack([legit, cluster])
    # Use a sensitive threshold so we can pin the mechanism's
    # behavior; production defaults to 3.0 and would not flag a
    # 3-in-12 cluster.
    result = detect_embedding_anomalies(batch, z_threshold=1.0)

    # Cluster indices are 9, 10, 11.
    cluster_mean_dists = [result.mean_pairwise_distances[i] for i in (9, 10, 11)]
    legit_mean_dists = [result.mean_pairwise_distances[i] for i in range(9)]
    # The cluster's mean pairwise distance is smaller than the
    # legitimate chunks' on average -- the structural signal.
    assert max(cluster_mean_dists) < min(legit_mean_dists), (
        f"cluster vs legit means: cluster={cluster_mean_dists}, "
        f"legit={legit_mean_dists}"
    )


# -----------------------------------------------------------------
# AC 2 -- threshold tunability
# -----------------------------------------------------------------


def test_lower_threshold_flags_more_chunks() -> None:
    """
    z_threshold is per-call configurable; a lower threshold
    flags more chunks for the same input.
    """
    d = 16
    center = np.zeros(d)
    center[0] = 1.0
    legit = _normal_cluster(10, center=center, sigma=0.05, seed=7)
    attacker = np.zeros((1, d))
    attacker[0, 1] = 1.0  # mildly off-axis
    batch = np.vstack([legit, attacker])

    strict = detect_embedding_anomalies(batch, z_threshold=4.0)
    lenient = detect_embedding_anomalies(batch, z_threshold=1.0)
    assert len(lenient.flagged_indices) >= len(strict.flagged_indices)


def test_threshold_records_in_result() -> None:
    """The threshold used appears on the result so audit can capture it."""
    result = detect_embedding_anomalies(
        np.eye(4), z_threshold=2.5
    )
    assert result.z_threshold == 2.5


def test_default_threshold_matches_settings() -> None:
    """
    The function default matches the settings default so a
    deployment using the defaults end-to-end gets consistent
    behavior. Pinned so a future settings tweak that drifts from
    the function default fires this test.
    """
    settings = PalisadeSettings()
    # Call with no threshold argument and compare reported value.
    result = detect_embedding_anomalies(np.eye(4))
    assert result.z_threshold == settings.g3_anomaly_z_threshold


# -----------------------------------------------------------------
# Edge cases -- batch-too-small, all-identical, zero rows
# -----------------------------------------------------------------


def test_empty_batch_returns_benign_result() -> None:
    """No chunks -> no anomaly, no crash."""
    result = detect_embedding_anomalies(np.zeros((0, 8)))
    assert result.ran is False
    assert result.flagged_indices == ()
    assert result.z_scores == ()


def test_single_chunk_batch_returns_benign_result() -> None:
    """n=1 has no pairwise distances; detector skips."""
    result = detect_embedding_anomalies(np.array([[1.0, 0.0, 0.0]]))
    assert result.ran is False
    assert result.flagged_indices == ()
    assert len(result.z_scores) == 1
    assert math.isnan(result.z_scores[0])
    assert "batch too small" in result.reason.lower()


def test_two_chunk_batch_below_default_min_returns_benign() -> None:
    """
    n=2 yields a single pairwise distance, so the z-score
    distribution has zero std -- below the default
    min_batch_size of 3.
    """
    result = detect_embedding_anomalies(
        np.array([[1.0, 0.0], [0.0, 1.0]])
    )
    assert result.ran is False
    assert result.flagged_indices == ()


def test_min_batch_size_is_configurable() -> None:
    """`min_batch_size` is per-call configurable."""
    # n=3 with min_batch_size=5 -> skipped.
    result = detect_embedding_anomalies(
        np.eye(3), min_batch_size=5
    )
    assert result.ran is False
    assert "batch too small" in result.reason.lower()


def test_all_identical_embeddings_return_benign_result_with_explanation() -> None:
    """
    When every chunk's embedding is identical, all pairwise
    distances are 0 so the std is 0 and z-scores are undefined.
    The detector returns a documented benign result rather than
    NaN-propagating or dividing by zero.
    """
    identical = np.tile([1.0, 0.0, 0.0, 0.0], (5, 1))
    result = detect_embedding_anomalies(identical)
    assert result.ran is False
    assert result.flagged_indices == ()
    assert all(math.isnan(z) for z in result.z_scores)
    assert "std is zero" in result.reason.lower() or "identical" in result.reason.lower()


def test_zero_norm_row_does_not_crash() -> None:
    """
    A zero-vector embedding (degenerate) doesn't divide by zero
    during L2 normalization. The zero row is treated as
    maximally far from every other chunk (cosine sim with zero
    is 0, distance 1) and naturally flagged as an outlier in
    most setups.
    """
    d = 8
    legit = _normal_cluster(
        5, center=np.array([1.0] + [0.0] * (d - 1)), sigma=0.02, seed=8
    )
    zero_row = np.zeros((1, d))
    batch = np.vstack([legit, zero_row])

    # Just verify no crash; the zero row is row index 5.
    result = detect_embedding_anomalies(batch)
    assert isinstance(result, AnomalyResult)
    # No assertion on flagged_indices because the all-zero
    # behavior is well-defined but secondary to the no-crash
    # property.
    assert result.ran is True


def test_clean_batch_records_max_z_in_reason() -> None:
    """Clean-batch reason includes the max z-score so ops can spot near-threshold cases."""
    d = 16
    center = np.zeros(d)
    center[0] = 1.0
    legit = _normal_cluster(10, center=center, sigma=0.05, seed=9)
    result = detect_embedding_anomalies(legit, z_threshold=3.0)
    assert result.flagged_indices == ()
    assert result.ran is True
    assert "max|z|" in result.reason


# -----------------------------------------------------------------
# Determinism + result-immutability properties
# -----------------------------------------------------------------


def test_detector_is_deterministic_for_same_input() -> None:
    """Pure function -- two calls with the same input return equal results."""
    rng = np.random.default_rng(123)
    embeddings = rng.standard_normal((10, 32))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    a = detect_embedding_anomalies(embeddings)
    b = detect_embedding_anomalies(embeddings)
    assert a.flagged_indices == b.flagged_indices
    assert a.z_scores == b.z_scores
    assert a.mean_pairwise_distances == b.mean_pairwise_distances


def test_anomaly_result_is_frozen() -> None:
    """Audit consumers can stash the result without fear of mutation."""
    result = detect_embedding_anomalies(np.eye(4))
    with pytest.raises(Exception):
        result.flagged_indices = (1,)  # type: ignore[misc]


# -----------------------------------------------------------------
# Internal helpers -- minimal coverage so a refactor that breaks
# them surfaces here, not deep in a higher-level test
# -----------------------------------------------------------------


def test_coerce_embeddings_returns_float64_2d_array() -> None:
    arr = _coerce_embeddings([[1, 2], [3, 4]])
    assert arr.dtype == np.float64
    assert arr.shape == (2, 2)


def test_l2_normalize_handles_zero_rows() -> None:
    """Zero row stays zero; non-zero row is normalized to unit length."""
    arr = np.array([[3.0, 4.0], [0.0, 0.0]])
    norm = _l2_normalize(arr)
    assert math.isclose(np.linalg.norm(norm[0]), 1.0, abs_tol=1e-9)
    assert np.array_equal(norm[1], [0.0, 0.0])
