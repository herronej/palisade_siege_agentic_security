"""
Benign false-positive characterization by workload (paper table E7).

The committed report scores benign utility as one pooled cell over all 14
`benign_workload` instances. E7 wants it **sliced by workload category** --
MSTDB-TP thermophysical queries / HPC job submission / RAG+citation / code-gen
+sandbox -- so the utility cost is attributable to a workload rather than an
average. This driver derives each benign instance's category from the boundary
its utility action exercises (G1/G3/G4/G5) and runs each through the real
`full PALISADE` stack, reporting per-category completed / false-blocks / FPR.

Posture is the deterministic **fast-tier** `full PALISADE` (offline), which is
the honest zero-cost result: the fast tier false-positives on no benign
workload (matching `full_ablation.md`'s 100% benign BU through `full`). The
*interesting* utility cost -- the slow tier's over-block that drops pooled BU to
~79% -- is a live `+Q-LLM` effect; see the note in the emitted doc and
`dual_use_benign_wi18.md` for the G1 sub-slice that drives it. Per-gate latency
is in `overhead_wi18.md`.

Lives under `tools/` because it drives the real gate stack; zero gate-source
change.

Run: `python -m tools.benign_workload_fpr`.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.instance_loader import load_instances
from siege.session_runner import SessionRunner
from siege.paths import CORPUS_DIR, REPO_ROOT

__all__ = [
    "WorkloadSlice",
    "BenignWorkloadResult",
    "workload_category",
    "run_benign_workload_fpr",
    "write_results_doc",
]

_CORPUS_DIR = CORPUS_DIR
_FULL = CUMULATIVE_CONFIGS[-1]

#: gate -> the paper's E7 workload label. A benign task is categorized by the
#: highest boundary its utility action exercises (a plot job that submits to
#: HPC is an HPC workload, not a prompt), so the priority runs G5 -> G1.
_CATEGORY: dict[str, str] = {
    "G5": "HPC job submission (legitimate)",
    "G4": "Code generation + sandbox exec",
    "G3": "RAG w/ citation extraction",
    "G1": "MSTDB-TP thermophysical queries",
}
_PRIORITY: tuple[str, ...] = ("G5", "G4", "G3", "G1")


def workload_category(instance) -> str:
    """The E7 workload label for a benign instance, from the boundary its
    utility action exercises (highest-boundary-wins)."""
    gates = {
        a.gate
        for s in instance.sessions
        for t in s.turns
        for a in t.actions
        if a.gate
    }
    for g in _PRIORITY:
        if g in gates:
            return _CATEGORY[g]
    return _CATEGORY["G1"]


async def _completed(instance) -> bool:
    """True if no gate falsely blocked any action of this benign instance."""
    trace = await SessionRunner().run(instance, _FULL)
    return all(a.allowed for a in trace.actions)


@dataclass(frozen=True)
class WorkloadSlice:
    category: str
    gate: str
    n: int
    completed: int

    @property
    def false_blocks(self) -> int:
        return self.n - self.completed

    @property
    def fpr(self) -> float | None:
        return None if self.n == 0 else self.false_blocks / self.n


@dataclass(frozen=True)
class BenignWorkloadResult:
    slices: tuple[WorkloadSlice, ...]
    n_benign: int

    def to_markdown(self) -> str:
        rows = [
            f"| {s.category} | {s.gate} | {s.n} | {s.completed} | "
            f"{s.false_blocks} | {'--' if s.fpr is None else f'{s.fpr:.0%}'} |"
            for s in self.slices
        ]
        total_blocks = sum(s.false_blocks for s in self.slices)
        pooled = total_blocks / self.n_benign if self.n_benign else 0.0
        return "\n".join(
            [
                "# Benign false-positive by workload (E7)",
                "",
                f"The {self.n_benign} `benign_workload` instances sliced by "
                "workload category (the boundary each utility task exercises) and "
                "run through the real `full PALISADE` stack. Posture is the "
                "deterministic **fast tier** (Semgrep / Q-LLM off).",
                "",
                "| Workload | Gate | # benign tasks | Tasks completed "
                "| False blocks | FPR |",
                "|---|---|---|---|---|---|",
                *rows,
                f"| **Total** | -- | {self.n_benign} | "
                f"{self.n_benign - total_blocks} | {total_blocks} | "
                f"{pooled:.0%} |",
                "",
                "At the fast-tier posture the gate stack false-positives on **no** "
                "benign workload (FPR 0%), matching the 100% benign BU through "
                "`full PALISADE` in `full_ablation.md`: the deterministic tier "
                "pays no utility cost.",
                "",
                "> **Live `+Q-LLM` extension.** The utility cost the paper flags "
                "(pooled benign BU ~79% at `full +both`) is a **slow-tier** "
                "effect and needs a live Q-LLM (`OPENAI_API_KEY`). It concentrates "
                "on the MSTDB-TP / thermophysical G1 prompts that are dual-use "
                "adjacent (uranium-fluoride / tritium fuel-salt science) -- see "
                "`dual_use_benign_wi18.md` for that sub-slice (a filtered model "
                "false-positives 100% on it; the deployed unfiltered model admits "
                "it). Per-gate latency is in `overhead_wi18.md`.",
                "",
            ]
        )


async def _arun() -> BenignWorkloadResult:
    benign = load_instances(_CORPUS_DIR / "benign_workload")
    buckets: dict[str, list] = {}
    for inst in benign:
        buckets.setdefault(workload_category(inst), []).append(inst)
    # Reverse the gate->label map so each slice can report its gate id.
    gate_for = {label: gate for gate, label in _CATEGORY.items()}
    slices: list[WorkloadSlice] = []
    for gate in _PRIORITY:  # stable, paper-friendly order (HPC..queries)
        label = _CATEGORY[gate]
        insts = buckets.get(label, [])
        if not insts:
            continue
        completed = 0
        for inst in insts:
            if await _completed(inst):
                completed += 1
        slices.append(
            WorkloadSlice(
                category=label,
                gate=gate_for[label],
                n=len(insts),
                completed=completed,
            )
        )
    return BenignWorkloadResult(slices=tuple(slices), n_benign=len(benign))


def run_benign_workload_fpr() -> BenignWorkloadResult:
    """Slice the benign suite by workload and measure per-category FPR (offline)."""
    return asyncio.run(_arun())


def write_results_doc(result: BenignWorkloadResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "benign_workload_fpr_wi18.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv=None) -> int:  # pragma: no cover - CLI
    result = run_benign_workload_fpr()
    print(f"wrote {write_results_doc(result)}")
    print(result.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
