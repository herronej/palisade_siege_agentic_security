"""
External-defense baselines for the competitive comparison (WI20).

A detection-only comparator on the same corpus. Self-contained (no
``palisade.gates`` import) so it stays inside the ``redteam`` read-only
guarantee and is an honest *external* defense, not the gate stack in disguise.
"""

from __future__ import annotations

from siege.redteam.baselines.detectors import (
    DenylistDetector,
    Detector,
    DetectorVerdict,
    PromptGuardAdapter,
    PromptGuard2Adapter,
    LlmJudgeDetector,
    ScreenResult,
    screen_texts,
)

__all__ = [
    "Detector",
    "DetectorVerdict",
    "DenylistDetector",
    "PromptGuardAdapter",
    "PromptGuard2Adapter",
    "LlmJudgeDetector",
    "ScreenResult",
    "screen_texts",
]
