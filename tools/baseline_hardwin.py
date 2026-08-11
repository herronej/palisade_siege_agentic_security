"""Detection-only baseline hard-win rate over the SIEGE corpus (tab:e1 YY).

The competitive number the internal ablation does not give: a *detection-only*
defense (a LlamaFirewall / PromptGuard-style surface detector, here
``redteam.baselines.DenylistDetector`` with its own public
injection/jailbreak/mining/IOC patterns) run over the full 200-instance corpus,
scored on the **same** hard-win metric as PALISADE.

A detection-only defense blocks an attack iff it flags the attack's content; it
carries no capability model, so where it misses, the value reaches the sink
exactly as undefended. The faithful hard-win rate is therefore::

    baseline_hard_win(i)  =  undefended_hard_win(i)  AND  NOT detector_flags(i)

i.e. the undefended hard-wins the detector fails to catch. The structural point
(Section~sec:principle) falls straight out: the hard-wins are tag-dropped values
whose *text is benign*, so no content detector flags them --- the baseline leaves
essentially the whole undefended hard-win rate standing, while PALISADE's
capability bound drives it to the reported residual.

Fully offline/deterministic (the ``SessionRunner`` replay + a regex detector); no
endpoint needed. Run:
``python -m tools.baseline_hardwin``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from siege.redteam.baselines import DenylistDetector, Detector
from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.instance_loader import load_instances
from siege.scorer import score_trace
from siege.session_runner import SessionRunner
from palisade.paths import CORPUS_DIR, REPO_ROOT

__all__ = ["FamilyRow", "BaselineHardWinResult", "run_baseline_hardwin", "write_results_doc"]

_CORPUS = CORPUS_DIR
_OFF = CUMULATIVE_CONFIGS[0]        # baseline, all gates off (undefended)
_FULL = CUMULATIVE_CONFIGS[-1]      # full PALISADE

#: Single-trusted-principal misuse classes -- excluded from the security subset
#: (they carry no cross-principal hard-win), matching the tab:e2 headline basis.
_MISUSE = {"b1_1", "b1_2", "b1_3", "b1_4", "b1_5", "b1_7", "b1_8", "b1_10"}
#: Contract-adjudicated science classes -- likewise outside the security subset.
_SCIENCE = {"b3_3", "b3_4", "b4_5"}
_FAMS = ("b1", "b3", "b4", "b5", "xc")


def _class_key(template: str) -> str:
    return "_".join(template.split("_")[:2])  # b1_1_direct_injection -> b1_1


def _leaf_strings(v: Any) -> Iterable[str]:
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _leaf_strings(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _leaf_strings(x)


def _attack_texts(inst: Any) -> list[str]:
    """Every untrusted string in an instance's attack actions (what a detector sees)."""
    out: list[str] = []
    for session in inst.sessions:
        for turn in session.turns:
            for action in turn.actions:
                if getattr(action, "is_attack", False):
                    out.extend(s for s in _leaf_strings(action.payload) if s)
    return out


def _flagged_by(detector: Detector, inst: Any) -> bool:
    """A detection-only defense blocks the instance iff it flags any attack text."""
    return any(detector.flag(t).flagged for t in _attack_texts(inst))


@dataclass(frozen=True)
class FamilyRow:
    family: str
    n: int
    undef_hwr: float
    baseline_hwr: float
    vg_hwr: float
    detector_catch_on_hardwins: float | None  # flag rate over that family's undefended hard-wins


@dataclass(frozen=True)
class BaselineHardWinResult:
    detector: str
    rows: tuple[FamilyRow, ...]
    # security subset (excl. misuse + science) -- matches the tab:e2 headline basis
    sec_n: int
    sec_undef_hwr: float
    sec_baseline_hwr: float
    sec_vg_hwr: float
    # all-200 basis
    all_n: int
    all_undef_hwr: float
    all_baseline_hwr: float
    all_vg_hwr: float
    surface_catch: float          # detector flag rate over all attacks
    catch_on_hardwins: float | None  # detector flag rate over undefended hard-wins

    def to_markdown(self) -> str:
        def pct(x: float | None) -> str:
            return "--" if x is None else f"{x:.0%}"

        lines = [
            "# Detection-only baseline hard-win rate (tab:e1 YY)",
            "",
            f"A detection-only defense (`{self.detector}`, its own public "
            "injection/jailbreak/mining/IOC patterns -- a LlamaFirewall / "
            "PromptGuard-style surface detector) over the full 200-instance "
            "SIEGE corpus, scored on the same hard-win metric as "
            "PALISADE. It blocks an attack iff it flags the content; where it "
            "misses, the value reaches the sink as undefended (no capability "
            "backstop), so `baseline_hard_win = undefended_hard_win AND not flagged`.",
            "",
            "## Headline (security subset, matches the tab:e2 basis)",
            "",
            "| Defense | Hard-win rate |",
            "|---|---|",
            f"| Undefended (no defense) | **{pct(self.sec_undef_hwr)}** |",
            f"| Detection-only baseline (`{self.detector}`) | **{pct(self.sec_baseline_hwr)}** |",
            f"| PALISADE (full) | **{pct(self.sec_vg_hwr)}** |",
            "",
            f"_Security subset: {self.sec_n} instances (misuse + science excluded). "
            f"The detector's surface-catch rate over all attacks is "
            f"{pct(self.surface_catch)}, but its catch rate over the undefended "
            f"**hard-win** instances is only {pct(self.catch_on_hardwins)} -- the "
            "hard-wins are tag-dropped values whose text is benign, so a content "
            "detector cannot see them. That is the structural gap: the baseline "
            "leaves nearly the whole undefended hard-win rate standing; the "
            "capability bound is what closes it._",
            "",
            "## All-200 basis",
            "",
            "| Defense | Hard-win rate |",
            "|---|---|",
            f"| Undefended | {pct(self.all_undef_hwr)} |",
            f"| Detection-only baseline | {pct(self.all_baseline_hwr)} |",
            f"| PALISADE (full) | {pct(self.all_vg_hwr)} |",
            "",
            "## Per family",
            "",
            "| Family | n | Undef HWR | Baseline HWR | PALISADE HWR | Detector catch on hard-wins |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.rows:
            lines.append(
                f"| {r.family} | {r.n} | {pct(r.undef_hwr)} | {pct(r.baseline_hwr)} | "
                f"{pct(r.vg_hwr)} | {pct(r.detector_catch_on_hardwins)} |"
            )
        lines.append("")
        return "\n".join(lines)


async def _score(config: Any) -> dict[str, tuple[Any, Any]]:
    runner = SessionRunner()
    out: dict[str, tuple[Any, Any]] = {}
    for inst in load_instances(_CORPUS):
        if not inst.is_attack:
            continue
        trace = await runner.run(inst, config)
        out[inst.instance_id] = (inst, score_trace(trace, inst))
    return out


async def _arun(detector: Detector) -> BaselineHardWinResult:
    off = await _score(_OFF)
    full = await _score(_FULL)

    def fam_of(inst: Any) -> str:
        return inst.template.split("_")[0]

    def is_security(inst: Any) -> bool:
        k = _class_key(inst.template)
        return k not in _MISUSE and k not in _SCIENCE

    ids = list(off)
    flagged = {i: _flagged_by(detector, off[i][0]) for i in ids}
    undef_hw = {i: bool(off[i][1].hard_win) for i in ids}
    vg_hw = {i: bool(full[i][1].hard_win) for i in ids}
    base_hw = {i: undef_hw[i] and not flagged[i] for i in ids}

    def rate(pred, over) -> float | None:
        xs = [i for i in ids if over(i)]
        return (sum(1 for i in xs if pred(i)) / len(xs)) if xs else None

    rows: list[FamilyRow] = []
    for fam in _FAMS:
        sel = [i for i in ids if fam_of(off[i][0]) == fam]
        if not sel:
            continue
        hw_ids = [i for i in sel if undef_hw[i]]
        rows.append(FamilyRow(
            family=fam, n=len(sel),
            undef_hwr=sum(undef_hw[i] for i in sel) / len(sel),
            baseline_hwr=sum(base_hw[i] for i in sel) / len(sel),
            vg_hwr=sum(vg_hw[i] for i in sel) / len(sel),
            detector_catch_on_hardwins=(sum(flagged[i] for i in hw_ids) / len(hw_ids)) if hw_ids else None,
        ))

    sec = [i for i in ids if is_security(off[i][0])]
    hw_all = [i for i in ids if undef_hw[i]]
    return BaselineHardWinResult(
        detector=detector.name,
        rows=tuple(rows),
        sec_n=len(sec),
        sec_undef_hwr=sum(undef_hw[i] for i in sec) / len(sec),
        sec_baseline_hwr=sum(base_hw[i] for i in sec) / len(sec),
        sec_vg_hwr=sum(vg_hw[i] for i in sec) / len(sec),
        all_n=len(ids),
        all_undef_hwr=sum(undef_hw[i] for i in ids) / len(ids),
        all_baseline_hwr=sum(base_hw[i] for i in ids) / len(ids),
        all_vg_hwr=sum(vg_hw[i] for i in ids) / len(ids),
        surface_catch=sum(flagged[i] for i in ids) / len(ids),
        catch_on_hardwins=(sum(flagged[i] for i in hw_all) / len(hw_all)) if hw_all else None,
    )


def run_baseline_hardwin(detector: Detector | None = None) -> BaselineHardWinResult:
    return asyncio.run(_arun(detector or DenylistDetector()))


def write_results_doc(result: BaselineHardWinResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "baseline_hardwin.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Detection-only baseline hard-win (tab:e1 YY)")
    parser.add_argument("--report-out", default=None, metavar="PATH")
    args = parser.parse_args(argv)
    result = run_baseline_hardwin()
    print(f"wrote {write_results_doc(result, args.report_out)}")
    print(result.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
