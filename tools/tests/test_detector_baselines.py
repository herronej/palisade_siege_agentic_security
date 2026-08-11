"""W4.1 detector-baseline comparison (real Prompt-Guard vs the regex denylist).

The offline pieces always run: the denylist's benign FPR over the W1 control, the
markdown rendering, and the denylist-only path when Prompt-Guard's weights are
absent (the R1-M5 fallback). The real-model path needs the gated weights and is
exercised only in the live run (`tools.detector_baselines`), not CI.
"""

from __future__ import annotations

from siege.redteam.baselines import DenylistDetector, PromptGuardAdapter
from tools.baseline_hardwin import BaselineHardWinResult, FamilyRow
from tools.detector_baselines import (
    DetectorBaseline,
    DetectorBaselinesResult,
    detector_benign_fpr,
    run_detector_baselines,
)


def _synthetic_hardwin(name: str) -> BaselineHardWinResult:
    row = FamilyRow(
        family="xc", n=22, undef_hwr=0.64, baseline_hwr=0.55,
        vg_hwr=0.18, detector_catch_on_hardwins=0.14,
    )
    return BaselineHardWinResult(
        detector=name, rows=(row,),
        sec_n=150, sec_undef_hwr=0.21, sec_baseline_hwr=0.16, sec_vg_hwr=0.03,
        all_n=205, all_undef_hwr=0.18, all_baseline_hwr=0.14, all_vg_hwr=0.02,
        surface_catch=0.20, catch_on_hardwins=0.22,
    )


def test_detector_benign_fpr_denylist_is_low():
    """The 9-pattern denylist flags no benign task in the W1 expanded control."""
    k, n, ids = detector_benign_fpr(DenylistDetector())
    assert n >= 150  # the W1 expanded control (~181 tasks)
    assert k == 0 and ids == []


def test_markdown_renders_headline_and_fallback():
    b = DetectorBaseline(
        detector="denylist-detector", kind="regex reimplementation",
        hardwin=_synthetic_hardwin("denylist-detector"),
        fpr_k=0, fpr_n=181, fpr_flagged_ids=(),
    )
    result = DetectorBaselinesResult(
        baselines=(b,), prompt_guard_available=False,
        prompt_guard_model="meta-llama/Prompt-Guard-86M",
    )
    md = result.to_markdown()
    assert "PALISADE (full)" in md
    assert "Prompt-Guard unavailable" in md  # R1-M5 fallback note when no weights
    assert "14%" in md  # the denylist hard-win headline
    assert "4.4%" in md  # PALISADE's benign FPR paired in the same table


def test_run_uses_denylist_only_when_prompt_guard_absent(monkeypatch):
    """With no weights, the tool reports the denylist alone and reproduces 14%."""
    monkeypatch.setattr(PromptGuardAdapter, "available", lambda self: False)
    result = run_detector_baselines()
    assert result.prompt_guard_available is False
    assert [b.detector for b in result.baselines] == ["denylist-detector"]
    dl = result.baselines[0].hardwin
    assert round(dl.all_baseline_hwr, 2) == 0.14
    assert round(dl.all_vg_hwr, 2) == 0.02
