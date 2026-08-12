"""
Per-gate leave-one-out ablation (paper table E3).

The committed sweep (`full_ablation.md`) is the **cumulative** add-order matrix
(baseline -> +G4 -> ... -> full): each gate's *marginal* value as it comes
online. This driver produces the **dual** view -- each gate's contribution
*removed from the otherwise-complete* stack (`full` minus one gate) -- rolled up
per boundary family (B1/B3/B4/B5/XC) into the compact grid the paper's E3 table
wants. A gate whose removal lets a family's attacks back in is a gate the full
system needs.

Posture is the deterministic **fast-tier** `full PALISADE` (Semgrep / Q-LLM
off), so this runs fully offline (no keys) and every `full -Gx` column is
directly comparable to the `full PALISADE` column of `full_ablation.md`. Two
caveats the doc restates: `-G2` removes the enforcement-only wrapper (a floor,
bundled inside `full`), and **`-G6`** is the egress grounding sink (citation-
provenance binding, the second sink of the two-sink mechanism): removing it lets
forged citations reach the answer, a grounding hard win. The paper's "Fast-tier
only (all gates)" row IS the `full PALISADE` reference row here.

Lives under `tools/` because it drives the real gate stack; it only *calls* the
gates (zero gate-source change).

Run: `python -m tools.leave_one_out_ablation`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

from siege.eval.siege_runner import run_siege_evaluation
from siege.ablation_matrix import LEAVE_ONE_OUT_CONFIGS
from siege.scorer import wilson_interval
from siege.paths import REPO_ROOT

__all__ = [
    "FamilyRollup",
    "ConfigRollup",
    "LeaveOneOutResult",
    "run_leave_one_out",
    "write_results_doc",
]

#: (label, template-prefix) for the boundary families, in paper order. B2 has
#: no attack classes (enforcement-only), so it is absent by construction.
_FAMILIES: tuple[tuple[str, str], ...] = (
    ("B1", "b1"),
    ("B3", "b3"),
    ("B4", "b4"),
    ("B5", "b5"),
    ("XC", "xc"),
)


def _pool(pairs: list[tuple[float, int]]) -> tuple[int, int]:
    """Pool per-cell (rate, n) into (successes, N). ``round(rate*n)`` recovers
    the integer success count exactly (rate is k/n over an integer cell)."""
    k = sum(round(r * n) for r, n in pairs)
    n = sum(n for _, n in pairs)
    return k, n


@dataclass(frozen=True)
class FamilyRollup:
    """Pooled soft-ASR and hard-win for one boundary family under one config."""

    family: str
    n: int
    soft_k: int
    hard_k: int

    @property
    def soft_asr(self) -> float | None:
        return self.soft_k / self.n if self.n else None

    @property
    def hard_win(self) -> float | None:
        return self.hard_k / self.n if self.n else None


@dataclass(frozen=True)
class ConfigRollup:
    """One ablation config: per-family rollups + the pooled overall + benign FPR."""

    config_name: str
    families: tuple[FamilyRollup, ...]
    overall_soft_k: int
    overall_hard_k: int
    overall_n: int
    #: False-blocked benign instances, pooled over **every** benign template.
    benign_blocked_k: int
    n_benign: int

    def family(self, label: str) -> FamilyRollup | None:
        return next((f for f in self.families if f.family == label), None)

    @property
    def benign_fpr(self) -> float | None:
        return self.benign_blocked_k / self.n_benign if self.n_benign else None

    @property
    def overall_soft_asr(self) -> float | None:
        return self.overall_soft_k / self.overall_n if self.overall_n else None

    @property
    def overall_hard_win(self) -> float | None:
        return self.overall_hard_k / self.overall_n if self.overall_n else None


def _rollup_config(cells, config_name: str) -> ConfigRollup:
    """Roll one config's per-(boundary,template) cells up to per-family + overall."""
    attack = [c for c in cells if c.config_name == config_name and c.asr is not None]
    families: list[FamilyRollup] = []
    for label, prefix in _FAMILIES:
        fam = [c for c in attack if c.template.startswith(prefix)]
        soft_k, n = _pool([(c.asr, c.n_attack) for c in fam])
        hard_k, _ = _pool(
            [(c.hard_win_rate or 0.0, c.n_attack) for c in fam]
        )
        families.append(FamilyRollup(family=label, n=n, soft_k=soft_k, hard_k=hard_k))
    o_soft, o_n = _pool([(c.asr, c.n_attack) for c in attack])
    o_hard, _ = _pool([(c.hard_win_rate or 0.0, c.n_attack) for c in attack])
    # Pool over EVERY benign template. The benign control ships as two cells
    # (`benign_workload` n=24, `benign_diverse` n=157); taking only the first
    # reported the FPR over 157 rather than the 181 the paper's table and
    # `benign_fpr.md` use, which is where the 5%-vs-4.4% disagreement came from.
    benign_cells = [
        c
        for c in cells
        if c.config_name == config_name and c.bu is not None and c.n_benign
    ]
    blocked_k, benign_n = _pool(
        [(max(0.0, 1.0 - c.bu), c.n_benign) for c in benign_cells]
    )
    return ConfigRollup(
        config_name=config_name,
        families=tuple(families),
        overall_soft_k=o_soft,
        overall_hard_k=o_hard,
        overall_n=o_n,
        benign_blocked_k=blocked_k,
        n_benign=benign_n,
    )


@dataclass(frozen=True)
class LeaveOneOutResult:
    configs: tuple[ConfigRollup, ...]
    n_attack: int
    n_benign: int
    seed: int

    def _grid(self, *, metric: str) -> list[str]:
        """A ``metric in {'soft','hard'}`` grid: config rows x family columns."""
        head = "Overall ASR" if metric == "soft" else "Overall hard-win"
        lines = [
            "| Configuration | B1 | B3 | B4 | B5 | XC | " + head + " | Benign FPR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for cfg in self.configs:
            cells = []
            for label, _ in _FAMILIES:
                fr = cfg.family(label)
                rate = None if fr is None else (
                    fr.soft_asr if metric == "soft" else fr.hard_win
                )
                cells.append(_fmt(rate))
            overall = (
                cfg.overall_soft_asr if metric == "soft" else cfg.overall_hard_win
            )
            lines.append(
                f"| {cfg.config_name} | " + " | ".join(cells) + " | "
                f"{_fmt(overall)} | {_fmt_fpr(cfg)} |"
            )
        return lines

    def to_markdown(self) -> str:
        ref = self.configs[0].config_name if self.configs else "full"
        o_soft = self.configs[0].overall_soft_asr if self.configs else None
        o_n = self.configs[0].overall_n if self.configs else 0
        lo, hi = wilson_interval(
            self.configs[0].overall_hard_k if self.configs else 0, o_n or 1
        )
        return "\n".join(
            [
                "# Per-gate leave-one-out ablation (E3)",
                "",
                "Each gate's contribution **removed from the otherwise-complete "
                "stack** -- the dual of the cumulative sweep in `full_ablation.md`. "
                f"Rolled up per boundary family over {self.n_attack} attack + "
                f"{self.n_benign} benign instances (seed {self.seed}). Posture is "
                "the deterministic fast-tier `full PALISADE` (Semgrep / Q-LLM "
                "off), so every `full -Gx` column is comparable to the "
                "`full PALISADE` column of `full_ablation.md`.",
                "",
                "> The reference row `full PALISADE` is also the paper's "
                "\"Fast-tier only (all gates)\" row. `-G6` is the egress "
                "grounding sink (citation-provenance binding): removing it lets "
                "forged citations reach the answer, a grounding hard win. `-G2` "
                "removes the enforcement-only wrapper, which is bundled inside "
                "`full` with the trust scorer -- read it as a floor, not a "
                "boundary defense.",
                "",
                "## Hard-win rate (the §8.2 capability bound -- headline)",
                "",
                "A `taint:T` value reaching a high-privilege sink. Lower is better; "
                "a `full -Gx` cell **above** the `full PALISADE` reference is the "
                "hard win that gate was holding shut.",
                "",
                *self._grid(metric="hard"),
                "",
                "## Soft-win ASR (inspection evasion)",
                "",
                "Attack action admitted by a live gate (capability tag may remain "
                "intact). This is where most single-gate contributions show up "
                "at the fast tier.",
                "",
                *self._grid(metric="soft"),
                "",
                f"_Reference (`{ref}`): overall soft-win ASR "
                f"{_fmt(o_soft)}, overall hard-win "
                f"{round(lo * 100)}-{round(hi * 100)}% (95% CI, N={o_n}). "
                "Cells pool the family's per-class binomials; most families are "
                "n=20-50, so read the columns, not single cells._",
                "",
            ]
        )


def _fmt(rate: float | None) -> str:
    return "--" if rate is None else f"{rate:.0%}"


def _fmt_fpr(cfg: ConfigRollup) -> str:
    """Benign FPR as ``k/n = x.x%``. One decimal and an explicit denominator, so
    the column is directly checkable against `benign_fpr.md` and the paper's
    ablation table rather than rounding to a different-looking integer."""
    if not cfg.n_benign:
        return "--"
    return f"{cfg.benign_blocked_k}/{cfg.n_benign} = {cfg.benign_fpr:.1%}"


async def _arun(
    max_instances: int | None, bound: str = "declarative"
) -> LeaveOneOutResult:
    result = await run_siege_evaluation(
        configs=LEAVE_ONE_OUT_CONFIGS, max_instances=max_instances, bound=bound
    )
    configs = tuple(
        _rollup_config(result.cells, cfg.name) for cfg in LEAVE_ONE_OUT_CONFIGS
    )
    return LeaveOneOutResult(
        configs=configs,
        n_attack=result.n_attack,
        n_benign=result.n_benign,
        seed=result.seed,
    )


def run_leave_one_out(
    *, max_instances: int | None = None, bound: str = "declarative"
) -> LeaveOneOutResult:
    """Run the leave-one-out sweep (offline, deterministic) and roll it up.

    ``bound`` selects which label the capability check reads: ``declarative``
    (the authored L*, holding propagation fixed so a row's movement is
    attributable to the gate) or ``production`` (the deployed content-keyed
    predicate). See ``siege.session_runner.TaintBound``.
    """
    return asyncio.run(_arun(max_instances, bound))


def write_results_doc(result: LeaveOneOutResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "leave_one_out_ablation_wi18.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description=(
            "Per-gate leave-one-out ablation (E3): full PALISADE then "
            "full -Gx per gate, rolled up by boundary family. Offline / "
            "deterministic (fast-tier posture)."
        )
    )
    parser.add_argument(
        "--max", type=int, default=None, help="Instances to run (id-sorted subset)."
    )
    parser.add_argument(
        "--bound",
        choices=("declarative", "production"),
        default="declarative",
        help=(
            "Which label the capability bound reads (default: declarative, the "
            "authored L*). 'production' uses the deployed content-keyed "
            "predicate."
        ),
    )
    parser.add_argument(
        "--report-out",
        default=None,
        metavar="PATH",
        help="Write the report to PATH (default: docs/palisade/...).",
    )
    args = parser.parse_args(argv)
    result = run_leave_one_out(max_instances=args.max, bound=args.bound)
    print(f"wrote {write_results_doc(result, args.report_out)}")
    print(result.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
