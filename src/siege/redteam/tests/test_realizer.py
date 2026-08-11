"""
WI13c realizer tests: the paraphrase-and-score loop picks the candidate whose
embedding is closest to the target (penalized for leaving the benign norm
shell); the optional vec2text realizer refuses to be a silent no-op.
"""

from __future__ import annotations

import numpy as np
import pytest

from siege.redteam.embedding_optimizer import HashingEncoder
from siege.redteam.manifold import BenignManifold
from siege.redteam.realizer import ParaphraseRealizer, RealizeResult, Vec2TextRealizer

_TARGET_TEXT = "flibe salt melting point"
_NEAR = "melting point of flibe salt"  # high cosine to target
_FAR = "quarterly cafeteria budget meeting"  # low cosine


def _paraphraser(seed: str, n: int) -> list[str]:
    return [_FAR, _NEAR, "flibe melting point data", "unrelated note"][:n]


def test_paraphrase_realizer_picks_the_closest_to_target():
    enc = HashingEncoder(dim=128)
    target = enc.encode([_TARGET_TEXT])[0]
    realizer = ParaphraseRealizer(enc, _paraphraser)

    best = realizer.realize(target, seed_text=_TARGET_TEXT, budget=4)
    assert isinstance(best, RealizeResult)
    # The query-echoing paraphrase has the highest cosine to the target.
    assert best.text in (_NEAR, "flibe melting point data")
    results = realizer.realize_many(target, seed_text=_TARGET_TEXT, budget=4)
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
    assert best.cosine > min(r.cosine for r in results)


def test_norm_penalty_demotes_out_of_shell_candidates():
    enc = HashingEncoder(dim=128)
    target = enc.encode([_TARGET_TEXT])[0]
    # A manifold whose shell is a narrow band that the candidates' norms miss,
    # so the penalty is active and lowers score below raw cosine.
    manifold = BenignManifold(norm_quantiles=(0.0, 1.0), target_fp=0.5).fit(
        enc.encode(["a b c", "d e f", "g h i"])
    )
    realizer = ParaphraseRealizer(enc, _paraphraser, manifold=manifold, norm_penalty=5.0)
    results = realizer.realize_many(target, seed_text=_TARGET_TEXT, budget=4)
    # With a penalty, score <= cosine for every candidate (penalty is >= 0).
    assert all(r.score <= r.cosine + 1e-9 for r in results)


def test_vec2text_realizer_refuses_without_an_inverter():
    enc = HashingEncoder(dim=32)
    realizer = Vec2TextRealizer(enc, inverter=None)
    with pytest.raises(NotImplementedError):
        realizer.realize_many(np.ones(32), seed_text="x", budget=2)


def test_vec2text_realizer_uses_a_provided_inverter():
    enc = HashingEncoder(dim=64)
    target = enc.encode([_TARGET_TEXT])[0]

    def inverter(_target_vec: np.ndarray, n: int) -> list[str]:
        return [_NEAR, _FAR][:n]

    realizer = Vec2TextRealizer(enc, inverter=inverter)
    results = realizer.realize_many(target, seed_text="x", budget=2)
    assert results and results[0].text == _NEAR  # closest-to-target first
