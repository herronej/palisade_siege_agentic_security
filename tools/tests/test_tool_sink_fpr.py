"""
W16 -- benign tool->sink control: the friction contract. Offline/deterministic
(the fast-tier taint bound is deterministic; the live recovery clear-rate needs
the served Q-LLM, stubbed here).

Pins R2-3's structural point: the §8.2 taint bound denies *every* untrusted
tool-derived value at a high-privilege sink (100% friction by construction), and
the slow-tier recovery is a content classification that cannot separate these from
content-benign injections.
"""

from __future__ import annotations

import asyncio

from palisade.quarantine import QuarantineDecision
from tools.tool_sink_fpr import _FLOWS, _fast_tier_blocks, measure


class _StubResult:
    def __init__(self, output: QuarantineDecision) -> None:
        self.output = output


class _AlwaysAgent:
    def __init__(self, contains: bool) -> None:
        self._c = contains

    async def run(self, _prompt: str):
        return _StubResult(
            QuarantineDecision(
                contains_instructions=self._c,
                suspicious_score=1.0 if self._c else 0.0,
            )
        )


def test_taint_bound_blocks_every_tool_sink_flow():
    # The structural friction R2-3 names: an untrusted-tainted value at a
    # high-privilege sink is denied by construction, for every flow.
    assert all(_fast_tier_blocks(value, sink) for _l, _c, value, sink, _a in _FLOWS)
    assert len(_FLOWS) >= 6
    assert {sink for _l, _c, _v, sink, _a in _FLOWS} == {"submit_hpc_job", "create_file"}


def test_recovery_is_the_sanitize_clear_rate():
    # A sanitize that clears everything recovers every flow (production admits);
    # one that flags everything recovers none (stays at 100% friction).
    all_clear = asyncio.run(measure(samples=1, agent=_AlwaysAgent(False)))
    assert all(cleared for _l, _t, _b, cleared, _a in all_clear)
    assert all(blocked for _l, _t, blocked, _c, _a in all_clear)  # fast tier still blocks

    all_flag = asyncio.run(measure(samples=1, agent=_AlwaysAgent(True)))
    assert not any(cleared for _l, _t, _b, cleared, _a in all_flag)
