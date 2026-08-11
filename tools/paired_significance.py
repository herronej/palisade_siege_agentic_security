"""Paired significance tests: PALISADE vs each content detector (R4-3.4).

The Pareto claim in the abstract ("dominates a regex denylist and a deployable
Prompt-Guard screen on both axes") was defended with unpaired exact-binomial
intervals, which overlap. That is the wrong instrument. Every system is scored
over the *same* 205 attack instances and the *same* 181 benign tasks, so the
per-instance outcomes are paired and McNemar's exact test applies. A paired test
conditions on the discordant pairs and is strictly more powerful than comparing
two marginal intervals.

Two axes, both paired:

* **hard win** over the 205-instance attack corpus. PALISADE's outcome is the
  deployed (``production``) predicate at ``full``. A detector's outcome is
  ``undefended_hard_win AND NOT flagged`` -- where it misses, the value reaches
  the sink exactly as undefended, which is the same convention
  ``tools.baseline_hardwin`` uses.
* **benign false block** over the 181-task control. PALISADE's outcome is the
  deterministic tier's block; a detector's is flagging any screened string.

Reports the discordant counts (b, c) and a two-sided exact McNemar p-value.
``b`` is "PALISADE better on this instance", ``c`` is "detector better", so
b > c favours PALISADE. Fully offline apart from the Prompt-Guard forward passes,
which run from the local HF cache.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

from siege import load_instances
from siege.ablation_matrix import CUMULATIVE_CONFIGS
from siege.corpus_builder import CORPUS_DIR
from siege.scorer import score_trace
from siege.session_runner import SessionRunner
from tools.baseline_hardwin import _OFF, _flagged_by
from tools.detector_baselines import _build_detectors, detector_benign_fpr

_FULL = next(c for c in CUMULATIVE_CONFIGS if c.name.lower().startswith("full"))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs.

    Under H0 each discordant pair is a fair coin, so the smaller count is
    Binomial(b+c, 1/2). Exact rather than chi-square because the discordant
    counts here are small (single digits on the benign axis).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True)
class PairedRow:
    axis: str
    comparator: str
    n: int
    palisade_k: int
    other_k: int
    both: int
    b: int  # PALISADE better (other fails, PALISADE does not)
    c: int  # comparator better
    p: float

    @property
    def favours(self) -> str:
        if self.b == self.c:
            return "tie"
        return "PALISADE" if self.b > self.c else self.comparator


def _paired(axis: str, comparator: str, cit: dict[str, bool], oth: dict[str, bool]) -> PairedRow:
    ids = sorted(set(cit) & set(oth))
    b = sum(1 for i in ids if oth[i] and not cit[i])
    c = sum(1 for i in ids if cit[i] and not oth[i])
    return PairedRow(
        axis=axis,
        comparator=comparator,
        n=len(ids),
        palisade_k=sum(cit[i] for i in ids),
        other_k=sum(oth[i] for i in ids),
        both=sum(1 for i in ids if cit[i] and oth[i]),
        b=b,
        c=c,
        p=mcnemar_exact(b, c),
    )


async def _outcomes(bound: str) -> tuple[dict[str, bool], dict[str, bool], dict[str, Any]]:
    """Per-instance (undefended hard win, PALISADE hard win, instance) over the attacks."""
    off_runner = SessionRunner()
    full_runner = SessionRunner(bound=bound)
    undef: dict[str, bool] = {}
    cit: dict[str, bool] = {}
    insts: dict[str, Any] = {}
    for inst in load_instances(CORPUS_DIR):
        if not inst.is_attack:
            continue
        insts[inst.instance_id] = inst
        undef[inst.instance_id] = bool(
            score_trace(await off_runner.run(inst, _OFF), inst).hard_win
        )
        cit[inst.instance_id] = bool(
            score_trace(await full_runner.run(inst, _FULL), inst).hard_win
        )
    return undef, cit, insts


async def _palisade_benign_blocks(bound: str) -> dict[str, bool]:
    """Per-instance benign false block under the deterministic tier."""
    from tools.benign_fpr import _utility_record

    runner = SessionRunner(bound=bound)
    out: dict[str, bool] = {}
    for inst in load_instances(CORPUS_DIR):
        if inst.is_attack:
            continue
        rec = _utility_record(await runner.run(inst, _FULL), None)
        out[inst.instance_id] = bool(rec is not None and not rec.allowed)
    return out


async def _arun(bound: str, model_name: str | None) -> list[PairedRow]:
    undef, cit_hw, insts = await _outcomes(bound)
    cit_benign = await _palisade_benign_blocks(bound)
    rows: list[PairedRow] = []
    detectors, available, resolved = _build_detectors(
        model_name=model_name or "meta-llama/Prompt-Guard-86M"
    )
    if not available:
        print(
            f"[paired] Prompt-Guard weights unavailable for {resolved!r}; "
            "reporting the regex denylist only."
        )
    for det, label, _kind in detectors:
        det_hw = {
            i: undef[i] and not _flagged_by(det, insts[i]) for i in undef
        }
        rows.append(_paired("hard win", label, cit_hw, det_hw))
        _k, _n, flagged_ids = detector_benign_fpr(det)
        flagged = set(flagged_ids)
        det_benign = {i: i in flagged for i in cit_benign}
        rows.append(_paired("benign block", label, cit_benign, det_benign))
    return rows


def to_markdown(rows: list[PairedRow], bound: str) -> str:
    out = [
        "# Paired significance tests: PALISADE vs content detectors (R4-3.4)",
        "",
        f"PALISADE scored at `full`, taint bound **{bound}**. Every system runs over "
        "the same 205 attack instances and the same 181 benign tasks, so outcomes "
        "are paired and McNemar's exact test applies. `b` counts instances where "
        "the comparator fails and PALISADE does not; `c` the reverse. Only the "
        "discordant pairs carry information.",
        "",
        "| Axis | Comparator | n | PALISADE | Comparator | both | b | c | exact p | favours |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sig = "**" if r.p < 0.05 else ""
        out.append(
            f"| {r.axis} | {r.comparator} | {r.n} | {r.palisade_k} | {r.other_k} | "
            f"{r.both} | {r.b} | {r.c} | {sig}{r.p:.4g}{sig} | {r.favours} |"
        )
    out += [
        "",
        "_Lower is better on both axes (fewer hard wins, fewer false blocks). "
        "A bolded p is significant at 0.05. Where p is not significant the "
        "honest wording is 'improves on' rather than 'dominates'._",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Paired (McNemar) PALISADE vs detector tests.")
    ap.add_argument("--bound", choices=("declarative", "production"), default="production")
    ap.add_argument("--model", default=None, help="Prompt-Guard model spec.")
    ap.add_argument("--report-out", default=None, metavar="PATH")
    args = ap.parse_args(argv)

    rows = asyncio.run(_arun(args.bound, args.model))
    md = to_markdown(rows, args.bound)
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"wrote {args.report_out}")
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
