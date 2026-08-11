"""
ASR-at-budget logging (WI13).

The adaptive evaluation reports a *curve*, not a point: attack success
rate as a function of query budget. A static benchmark can only report the
endpoint; the whole reason to wrap the harness in an RL loop is to see how
fast a policy climbs. ``AsrAtBudget`` accumulates per-episode win kinds in
order and emits the cumulative soft/hard ASR at increasing budgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from siege.redteam.reward import WinKind


@dataclass(frozen=True)
class AsrPoint:
    """One point on the ASR-at-budget curve."""

    queries: int
    soft_asr: float
    hard_asr: float


@dataclass
class AsrAtBudget:
    """Accumulates episode outcomes and produces the ASR-at-budget curve."""

    outcomes: list[WinKind] = field(default_factory=list)

    def record(self, win: WinKind) -> None:
        self.outcomes.append(win)

    def __len__(self) -> int:
        return len(self.outcomes)

    def _asr_through(self, n: int) -> tuple[float, float]:
        window = self.outcomes[:n]
        if not window:
            return (0.0, 0.0)
        soft = sum(1 for w in window if w in (WinKind.SOFT, WinKind.HARD)) / len(window)
        hard = sum(1 for w in window if w is WinKind.HARD) / len(window)
        return (soft, hard)

    def soft_asr(self) -> float:
        return self._asr_through(len(self.outcomes))[0]

    def hard_asr(self) -> float:
        return self._asr_through(len(self.outcomes))[1]

    def curve(self, n_points: int = 10) -> list[AsrPoint]:
        """Cumulative ASR at ``n_points`` evenly spaced budgets.

        Always returns at least two points (start and end) when there is
        more than one episode, so the output is a curve, not a point.
        """
        total = len(self.outcomes)
        if total == 0:
            return []
        n_points = max(2, min(n_points, total))
        budgets = sorted({max(1, round(total * k / n_points)) for k in range(1, n_points + 1)})
        points: list[AsrPoint] = []
        for b in budgets:
            soft, hard = self._asr_through(b)
            points.append(AsrPoint(queries=b, soft_asr=soft, hard_asr=hard))
        return points

    def to_markdown(self) -> str:
        lines = ["| queries | soft ASR | hard ASR |", "|---|---|---|"]
        for p in self.curve():
            lines.append(f"| {p.queries} | {p.soft_asr:.0%} | {p.hard_asr:.0%} |")
        return "\n".join(lines)
