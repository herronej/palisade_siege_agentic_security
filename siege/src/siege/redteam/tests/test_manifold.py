"""
WI13c acceptance (manifold half): ``BenignManifold.fit`` runs on a held-out
benign split and reports the G3 cluster-detector FP rate on it; the natural-norm
region it accepts is the region the *real* G3 detector is blind to.

The benign data is synthetic but geometrically controlled (a tight cluster
around a base direction, with a norm band) so the assertions are deterministic.
The grounding test imports the real detector (tests may; non-test redteam
modules may not).
"""

from __future__ import annotations

import numpy as np

from palisade.gates.g3_anomaly import detect_embedding_anomalies
from siege.redteam.manifold import BenignManifold

_DIM = 32


def _benign(rng: np.random.Generator, n: int, base: np.ndarray) -> np.ndarray:
    """A tight benign cluster: near ``base`` direction, norm ~ N(10, 1)."""
    dirs = base + 0.12 * rng.standard_normal((n, _DIM))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = 10.0 + rng.standard_normal(n)
    return dirs * norms[:, None]


def _fit_held():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(_DIM)
    base /= np.linalg.norm(base)
    e = _benign(rng, 400, base)
    return e[:300], e[300:], base, rng


def test_fit_runs_and_reports_held_out_fp_rate():
    fit, held, _base, _rng = _fit_held()
    m = BenignManifold(k=5, target_fp=0.05).fit(fit)

    report = m.report(held)
    # The headline acceptance: a held-out FP rate is produced and is sane
    # (calibrated to target on the fit split; not wildly overfit -- R-Int-6).
    assert 0.0 <= report.held_out_fp_rate <= 0.15
    assert report.held_out_fp_rate == m.false_positive_rate(held)
    lo, hi = m.norm_shell()
    assert 0.0 < lo < hi
    assert "held-out FP rate" in report.to_markdown()


def test_natural_norm_point_is_accepted_outlier_is_flagged():
    fit, _held, base, rng = _fit_held()
    m = BenignManifold(k=5, target_fp=0.05).fit(fit)

    # A natural-norm candidate (on the benign manifold) -> accepted.
    natural = _benign(rng, 1, base)[0]
    assert m.in_shell(natural)
    assert not m.flags(natural)
    assert m.cluster_detectability(natural) <= m.detect_threshold

    # A lone outlier (orthogonal direction, natural norm) -> flagged by the
    # symmetric density |z|, even though its norm is in the shell.
    orth = rng.standard_normal(_DIM)
    orth -= orth @ base * base
    orth /= np.linalg.norm(orth)
    outlier = orth * 10.0
    assert m.in_shell(outlier)  # norm alone does not save it...
    assert m.flags(outlier)  # ...the density test catches the lone outlier
    assert m.cluster_detectability(outlier) > m.detect_threshold


def test_grounds_against_the_real_g3_detector():
    """The manifold's accept/reject lines up with the real G3 z-test:
    a natural-norm chunk is invisible (the documented blind spot A1 exploits);
    a lone outlier is caught."""
    fit, held, base, rng = _fit_held()
    m = BenignManifold(k=5, target_fp=0.05).fit(fit)

    natural = _benign(rng, 1, base)[0]
    orth = rng.standard_normal(_DIM)
    orth -= orth @ base * base
    orth /= np.linalg.norm(orth)
    outlier = orth * 10.0

    benign_batch = held[:15]

    # Natural-norm candidate appended to a benign retrieval batch: NOT flagged
    # by the real detector (invisible by construction).
    batch_nat = np.vstack([benign_batch, natural[None, :]])
    res_nat = detect_embedding_anomalies(batch_nat, z_threshold=3.0)
    assert (len(batch_nat) - 1) not in res_nat.flagged_indices

    # Lone outlier appended: the real detector flags it (high +z).
    batch_out = np.vstack([benign_batch, outlier[None, :]])
    res_out = detect_embedding_anomalies(batch_out, z_threshold=3.0)
    assert (len(batch_out) - 1) in res_out.flagged_indices


def test_requires_fit_before_use():
    import pytest

    m = BenignManifold()
    with pytest.raises(RuntimeError):
        m.norm_shell()
    with pytest.raises(ValueError):
        BenignManifold().fit(np.zeros((1, _DIM)))  # need >= 2 rows
