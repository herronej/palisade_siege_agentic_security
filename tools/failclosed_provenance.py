"""
What the unlabeled-sink fail-closed rule costs on legitimate provenance work.

``full +failclosed`` was scored only against the 205/181 corpus, and that
corpus *cannot* price it: not one benign instance chains a retrieval into a
privileged sink, so the rule's measured FPR there is 0/181 by absence of the
workload, not by absence of cost (see ``tools.taint_interlock``).

The 132-session benign provenance control is the instrument that can. Every
instance is legitimate molten-salt work that retrieves and then writes or
submits, in two shapes:

``carry``  the sink reuses a retrieved value verbatim (a quoted constant).
``fresh``  the sink is freshly authored, sharing no distinctive text.

The deployed content-keyed guard refuses 0 of 132. The session interlock
refuses 132 of 132. This asks where the *narrow* rule -- deny only an
*unresolvable* argument at a high-stakes sink under live untrusted taint --
falls between those two poles.

Usage::

    python3 -m tools.failclosed_provenance [--bound production]
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from siege import load_instances
from siege.ablation_matrix import CUMULATIVE_CONFIGS, FAIL_CLOSED_CONFIG
from siege.corpus_builder import CORPUS_DIR
from siege.scorer import score_trace
from siege.session_runner import SessionRunner


def _full_config():
    for cfg in CUMULATIVE_CONFIGS:
        if cfg.name == "full PALISADE":
            return cfg
    raise LookupError("full PALISADE config not found")


@dataclass
class ConfigResult:
    name: str
    hard_wins: int = 0
    n_attack: int = 0
    benign_blocks: int = 0
    n_benign: int = 0
    hard_ids: list[str] = field(default_factory=list)
    benign_ids: list[str] = field(default_factory=list)
    n_prov: int = 0
    prov_blocks: int = 0
    prov_by_shape: Counter = field(default_factory=Counter)
    prov_total_by_shape: Counter = field(default_factory=Counter)
    prov_ids: list[str] = field(default_factory=list)
    prov_reasons: Counter = field(default_factory=Counter)

    @property
    def prov_fpr(self) -> float:
        return 100.0 * self.prov_blocks / self.n_prov if self.n_prov else 0.0

    @property
    def fpr(self) -> float:
        return 100.0 * self.benign_blocks / self.n_benign if self.n_benign else 0.0

    @property
    def hw_rate(self) -> float:
        return 100.0 * self.hard_wins / self.n_attack if self.n_attack else 0.0


async def _measure(cfg, bound: str) -> ConfigResult:
    from tools.benign_fpr import _utility_record
    from tools.benign_provenance_control import CONTROL_DIR

    runner = SessionRunner(bound=bound, interlock="off")
    res = ConfigResult(name=cfg.name)

    for inst in load_instances(CORPUS_DIR):
        trace = await runner.run(inst, cfg)
        if inst.is_attack:
            res.n_attack += 1
            if score_trace(trace, inst).hard_win:
                res.hard_wins += 1
                res.hard_ids.append(inst.instance_id)
        else:
            res.n_benign += 1
            rec = _utility_record(trace, None)
            if rec is not None and not rec.allowed:
                res.benign_blocks += 1
                res.benign_ids.append(inst.instance_id)

    for inst in load_instances(CONTROL_DIR):
        trace = await runner.run(inst, cfg)
        res.n_prov += 1
        shape = (getattr(inst, "variation_axis", "") or "").split("_", 1)[0]
        res.prov_total_by_shape[shape] += 1
        rec = _utility_record(trace, None)
        if rec is not None and not rec.allowed:
            res.prov_blocks += 1
            res.prov_by_shape[shape] += 1
            res.prov_ids.append(inst.instance_id)
            reason = (getattr(rec, "reason", "") or "").split(":", 1)[0].strip()
            res.prov_reasons[reason or "(no reason)"] += 1
    return res


def to_markdown(base: ConfigResult, fc: ConfigResult, bound: str) -> str:
    L: list[str] = []
    L.append("# `full +failclosed` against the 132-session provenance control\n")
    L.append(
        "`full +failclosed` denies a high-stakes sink whose argument resolves "
        "to *no* registered label while untrusted taint is live in the session "
        "(`g2_fail_closed_unlabeled`). It was scored only against the 205/181 "
        "corpus, where its benign cost reads 0/181 -- a number produced by the "
        "absence of the workload, since no benign corpus instance chains a "
        "retrieval into a sink. The 132-session provenance control is the "
        "instrument that can price it. Taint bound "
        f"`{bound}`, fast tier, interlock `off`.\n"
    )
    L.append(
        f"| config | hard wins / {base.n_attack} | corpus benign blocks / "
        f"{base.n_benign} | **provenance control blocked / {base.n_prov}** |"
    )
    L.append("|---|---|---|---|")
    for r in (base, fc):
        L.append(
            f"| `{r.name}` | {r.hard_wins} ({r.hw_rate:.1f}%) | "
            f"{r.benign_blocks} ({r.fpr:.1f}%) | "
            f"**{r.prov_blocks} ({r.prov_fpr:.1f}%)** |"
        )

    L.append("\n## By shape\n")
    L.append("| shape | n | blocked under `full` | blocked under `+failclosed` |")
    L.append("|---|---|---|---|")
    for shape in sorted(base.prov_total_by_shape):
        n = base.prov_total_by_shape[shape]
        L.append(
            f"| `{shape}` | {n} | {base.prov_by_shape.get(shape, 0)} | "
            f"**{fc.prov_by_shape.get(shape, 0)}** |"
        )

    if fc.prov_reasons:
        L.append("\n## Why the provenance sessions were refused\n")
        L.append("| signature | blocks |")
        L.append("|---|---|")
        for sig, n in fc.prov_reasons.most_common():
            L.append(f"| `{sig}` | {n} |")

    closed = base.hard_wins - fc.hard_wins
    added = fc.prov_blocks - base.prov_blocks
    L.append("\n## Reading\n")
    L.append(
        f"`+failclosed` closes **{closed}** of the {base.hard_wins} residual "
        f"hard wins and refuses **{fc.prov_blocks} of {fc.n_prov}** legitimate "
        f"retrieve-then-act sessions ({added:+d} against `full`'s "
        f"{base.prov_blocks}). On the 181-task corpus control the same rule "
        f"costs {fc.benign_blocks - base.benign_blocks:+d} blocks, which is "
        "the number that cannot be reported on its own.\n"
    )
    if fc.prov_ids:
        L.append(
            "Refused provenance sessions: "
            + ", ".join(f"`{i}`" for i in sorted(fc.prov_ids)[:20])
            + (" ..." if len(fc.prov_ids) > 20 else "")
            + "\n"
        )
    L.append("_Generated by `tools.failclosed_provenance`._")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bound", default="production", choices=["declarative", "production"])
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args(argv)

    base = asyncio.run(_measure(_full_config(), args.bound))
    fc = asyncio.run(_measure(FAIL_CLOSED_CONFIG, args.bound))
    md = to_markdown(base, fc, args.bound)
    if args.report_out:
        args.report_out.write_text(md, encoding="utf-8")
        print(f"wrote {args.report_out}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
