"""
Benign-embedding manifold model (PALISADE WI13c).

``BenignManifold`` is the attacker's *model* of where the benign corpus lives
in EmbeddingGemma space. It is the core of the A family (embedding-space
attacks): A1 "natural-norm corpus poisoning" searches for a poisoned chunk
that retrieves in the top-k **while staying inside this manifold**, so the G3
cluster detector's outlier test never fires.

Why a model and not the detector itself. The read-only rule (threat model §6)
forbids any ``redteam`` module from importing ``palisade.gates``. So this
module does not call the real G3 detector -- it *re-derives* the same geometry
the detector keys on, from the public encoder (the realism argument: the
EmbeddingGemma-300M encoder is public, so the embedding attacks are
legitimately white-box-on-the-encoder). The detector
(``gates/g3_anomaly.detect_embedding_anomalies``) flags, within a retrieval
*batch*, the per-chunk ``|z|`` of mean pairwise cosine distance -- symmetric,
so it catches both a lone optimized outlier (AgentPoison, arXiv 2407.12784)
and a tight injected cluster (PoisonedRAG, arXiv 2402.07867). Its documented
blind spot, verbatim from ``g3_anomaly.py``: *"an attacker who crafts a
poisoned chunk whose embedding lives inside the natural distribution (matching
its norm and direction statistics) is invisible to this detector by
construction."* That blind spot is exactly what this manifold maps.

Two complementary constraints, matching the two things the detector keys on:

- **Norm shell** -- the ``[low, high]`` band of *raw* L2 norms over the benign
  split. The detector L2-normalizes (so it ignores norm), but a poisoned chunk
  with an unnatural norm is conspicuous to the other G3 checks and to a
  corpus-reference baseline; constraining the search to the natural-norm band
  keeps the payload unremarkable.
- **kNN local density** -- in cosine geometry (L2-normalized, like the
  detector). ``cluster_detectability`` is the *symmetric* standardized surprise
  of the candidate's local density relative to the benign density: a candidate
  in a too-sparse region (a lone outlier, +z) or a too-dense region (a tight
  cluster, -z) both score high, mirroring the detector's symmetric ``|z|``.

The detectability threshold is **calibrated on the fit split** to a target
false-positive rate; ``false_positive_rate`` then reports the FP rate on a
**held-out** split, so the cluster-evasion claim is grounded rather than
overfit (the R-Int-6 mitigation).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["BenignManifold", "ManifoldReport"]

_EPS = 1e-12


@dataclass(frozen=True)
class ManifoldReport:
    """Held-out validation summary for a fitted ``BenignManifold``."""

    n_fit: int
    n_held_out: int
    norm_shell: tuple[float, float]
    detect_threshold: float
    target_fp: float
    held_out_fp_rate: float  # the headline: G3-cluster-detector FP rate on held-out

    def to_markdown(self) -> str:
        lo, hi = self.norm_shell
        return (
            "| metric | value |\n|---|---|\n"
            f"| fit / held-out size | {self.n_fit} / {self.n_held_out} |\n"
            f"| norm shell | [{lo:.3f}, {hi:.3f}] |\n"
            f"| detectability threshold | {self.detect_threshold:.3f} |\n"
            f"| target FP | {self.target_fp:.1%} |\n"
            f"| **held-out FP rate** | **{self.held_out_fp_rate:.1%}** |"
        )


class BenignManifold:
    """Norm shell + kNN local-density model of the benign embedding distribution.

    Args:
        k: neighbours for the local-density estimate.
        norm_quantiles: ``(low, high)`` quantiles defining the norm shell.
        target_fp: the benign false-positive rate the detectability threshold
            is calibrated to on the fit split.
    """

    def __init__(
        self,
        *,
        k: int = 5,
        norm_quantiles: tuple[float, float] = (0.01, 0.99),
        target_fp: float = 0.05,
    ) -> None:
        if not 0 < target_fp < 1:
            raise ValueError(f"target_fp must be in (0,1), got {target_fp!r}")
        if not (0.0 <= norm_quantiles[0] < norm_quantiles[1] <= 1.0):
            raise ValueError(f"bad norm_quantiles {norm_quantiles!r}")
        self._k = max(1, int(k))
        self._norm_q = norm_quantiles
        self._target_fp = float(target_fp)

        self._fitted = False
        self._unit: np.ndarray | None = None  # L2-normalized benign rows (cosine geometry)
        self._norm_lo = 0.0
        self._norm_hi = 0.0
        self._mu = 0.0  # mean of benign log local-density distances
        self._sigma = 1.0  # std of the same
        self._threshold = 0.0  # detectability cut at target_fp on the fit split

    # -----------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------
    def fit(self, benign_embeddings: Sequence[Sequence[float]] | np.ndarray) -> "BenignManifold":
        """Fit the norm shell + density model on a benign embedding split."""
        e = _coerce(benign_embeddings)
        n = e.shape[0]
        if n < 2:
            raise ValueError(f"need >= 2 benign embeddings to fit, got {n}")

        norms = np.linalg.norm(e, axis=1)
        self._norm_lo = float(np.quantile(norms, self._norm_q[0]))
        self._norm_hi = float(np.quantile(norms, self._norm_q[1]))
        self._unit = e / np.clip(norms, _EPS, None)[:, None]

        # Leave-one-out kNN cosine distance per benign point -> benign density.
        loo = self._knn_distances(self._unit, exclude_self=True)
        log_loo = np.log(loo + _EPS)
        self._mu = float(np.mean(log_loo))
        self._sigma = float(np.std(log_loo)) or 1.0

        # Calibrate the detectability cut so ~target_fp of the fit split flags.
        det = np.abs((log_loo - self._mu) / self._sigma)
        self._threshold = float(np.quantile(det, 1.0 - self._target_fp))
        self._fitted = True
        return self

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------
    def norm_shell(self) -> tuple[float, float]:
        self._check()
        return (self._norm_lo, self._norm_hi)

    def in_shell(self, embedding: Sequence[float] | np.ndarray) -> bool:
        """Is the candidate's raw L2 norm inside the benign norm band?"""
        self._check()
        norm = float(np.linalg.norm(np.asarray(embedding, dtype=np.float64)))
        return self._norm_lo <= norm <= self._norm_hi

    def cluster_detectability(self, embedding: Sequence[float] | np.ndarray) -> float:
        """KL-style density surprise vs the benign distribution (symmetric).

        Models benign log-local-density as ``Gaussian(mu, sigma)``; returns the
        candidate's standardized surprise ``|z|`` -- the per-point KL/NLL proxy.
        Low = the candidate sits in a typical benign neighbourhood (the natural-
        norm region the G3 detector is blind to); high = a lone outlier (sparse)
        or a tight cluster (over-dense), both of which the detector's symmetric
        ``|z|`` test flags. This is the term the A1 search minimizes.
        """
        self._check()
        d = self._candidate_knn_distance(embedding)
        z = (float(np.log(d + _EPS)) - self._mu) / self._sigma
        return abs(z)

    def flags(self, embedding: Sequence[float] | np.ndarray) -> bool:
        """The surrogate detector's verdict: outside the norm shell OR too anomalous."""
        self._check()
        return (not self.in_shell(embedding)) or (
            self.cluster_detectability(embedding) > self._threshold
        )

    @property
    def detect_threshold(self) -> float:
        self._check()
        return self._threshold

    # -----------------------------------------------------------------
    # Held-out FP-rate report (acceptance criterion)
    # -----------------------------------------------------------------
    def false_positive_rate(
        self, held_out: Sequence[Sequence[float]] | np.ndarray
    ) -> float:
        """Fraction of held-out *benign* embeddings the surrogate detector flags."""
        self._check()
        e = _coerce(held_out)
        if e.shape[0] == 0:
            return 0.0
        return float(np.mean([self.flags(row) for row in e]))

    def report(self, held_out: Sequence[Sequence[float]] | np.ndarray) -> ManifoldReport:
        """Bundle the norm shell, threshold, and held-out FP rate (the R-Int-6 report)."""
        self._check()
        e = _coerce(held_out)
        return ManifoldReport(
            n_fit=0 if self._unit is None else self._unit.shape[0],
            n_held_out=e.shape[0],
            norm_shell=(self._norm_lo, self._norm_hi),
            detect_threshold=self._threshold,
            target_fp=self._target_fp,
            held_out_fp_rate=self.false_positive_rate(e),
        )

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _knn_distances(self, unit: np.ndarray, *, exclude_self: bool) -> np.ndarray:
        """Mean cosine distance to the k nearest benign neighbours, per row."""
        sim = unit @ unit.T
        np.clip(sim, -1.0, 1.0, out=sim)
        dist = 1.0 - sim
        if exclude_self:
            np.fill_diagonal(dist, np.inf)
        n = dist.shape[0]
        k_eff = min(self._k, n - (1 if exclude_self else 0))
        k_eff = max(1, k_eff)
        # Partial sort: the k smallest distances per row.
        nearest = np.partition(dist, kth=k_eff - 1, axis=1)[:, :k_eff]
        return nearest.mean(axis=1)

    def _candidate_knn_distance(self, embedding: Sequence[float] | np.ndarray) -> float:
        assert self._unit is not None
        v = np.asarray(embedding, dtype=np.float64)
        v = v / max(float(np.linalg.norm(v)), _EPS)
        dist = 1.0 - np.clip(self._unit @ v, -1.0, 1.0)
        k_eff = max(1, min(self._k, dist.shape[0]))
        nearest = np.partition(dist, kth=k_eff - 1)[:k_eff]
        return float(nearest.mean())

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("BenignManifold.fit() must be called before use")


def _coerce(embeddings: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(embeddings, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D (n, dim); got {arr.ndim}-D shape {arr.shape}"
        )
    return arr
