"""
Soft-win vs hard-win decomposition by gate -- paper table E5.

The "guarantee where it resides" result: for each gate, how far an attack gets
through the tiers -- past the deterministic **fast** tier (soft), past the
**slow** (Q-LLM) tier too (soft), and finally **label-stripped at a sink**
(hard, the §8.2 capability breach). The columns are nested by construction --
fast-evaded >= slow-evaded >= label-stripped -- so the table shows the defense
handing off inward: most attacks clear the fast tier, fewer clear the slow tier,
and only a tag-dropped value reaches a sink with its label gone.

It is a **re-view of the same ablation** `full_ablation.md` measures, grouped by
each attack's routing/sink gate rather than by threat suite, so it needs no gate
or scorer change (the frozen stack stays frozen):

- fast-tier evaded = soft ASR at the fast-tier `full PALISADE` column;
- slow-tier evaded = soft ASR at the `full +all` column (Q-LLM slow tier on);
- label stripped at sink = the hard-win count at `full +all`.

Each attack template is assigned to the gate that inspects its sink action
(`b1_9` attached-content routes to G4, etc.); the cross-boundary `xc_*` chains
are grouped as **G6 Egress**, the capability/provenance bound that is the last
line for a tag-dropped value.

Reuses `full_ablation.run_full_ablation`, so `--stub` is offline/deterministic
(the fast-tier + hard columns are real; the slow-tier column is a stub and needs
a live Q-LLM for real numbers -- exactly like `full_ablation.md`'s live run).

``--qllm-samples k`` pools ``k`` independent live Q-LLM draws into the `+both`
slow-tier column (like ``full_ablation --qllm-samples``), smoothing the
single-draw variance; the reported `(count)` is renormalized onto the
per-instance attempt base so rates stay well-defined for ``k > 1``.

Run (offline): `python -m tools.soft_hard_decomposition --stub`
Run (live):    `OPENAI_API_KEY=... python -m ...soft_hard_decomposition --qllm-samples 5 --report-out ...`
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from siege.full_ablation import run_full_ablation
from siege.instance_loader import load_instances
from siege.smoke_static_qllm import DEFAULT_OPENAI_BASE_URL, DEFAULT_QLLM_MODEL
from palisade.paths import CORPUS_DIR, REPO_ROOT

__all__ = [
    "E5Row",
    "E5Result",
    "template_gate_map",
    "decompose",
    "write_results_doc",
]

_CORPUS = CORPUS_DIR
_FAST_CONFIG = "full PALISADE"
_SLOW_CONFIG = "full +all"
_SINK_ORDER = {"G1": 1, "G3": 2, "G4": 3, "G5": 4}
_GATE_LABELS = {
    "G1": "G1 Prompt",
    "G3": "G3 RAG",
    "G4": "G4 Code",
    "G5": "G5 HPC",
    "G6": "G6 Egress (cross-boundary)",
}
_GATE_ROWS = ("G1", "G3", "G4", "G5", "G6")


def template_gate_map() -> dict[str, str]:
    """Map each attack template to the gate that owns its E5 row.

    Single-gate attacks go to their sink gate (the highest-privilege gate their
    attack action reaches); the cross-boundary `xc_*` chains go to **G6**, the
    egress/capability bound that is the last line for a tag-dropped value.
    """
    gates: dict[str, set[str]] = defaultdict(set)
    for inst in load_instances(_CORPUS):
        if inst.kind != "attack":
            continue
        for session in inst.sessions:
            for turn in session.turns:
                for action in turn.actions:
                    if getattr(action, "is_attack", False) and action.gate:
                        gates[inst.template].add(action.gate)
    mapping: dict[str, str] = {}
    for template, gs in gates.items():
        if template.startswith("xc"):
            mapping[template] = "G6"
        else:
            mapping[template] = max(gs, key=lambda g: _SINK_ORDER.get(g, 0))
    return mapping


@dataclass(frozen=True)
class E5Row:
    gate: str
    attempted: int
    fast_evaded: int  # soft-win count at the fast-tier `full` column
    slow_evaded: int  # soft-win count at the `full +all` column
    label_stripped: int  # hard-win count at `full +all`

    def _rate(self, k: int) -> float | None:
        return None if self.attempted == 0 else k / self.attempted

    @property
    def fast_rate(self) -> float | None:
        return self._rate(self.fast_evaded)

    @property
    def slow_rate(self) -> float | None:
        return self._rate(self.slow_evaded)

    @property
    def hard_rate(self) -> float | None:
        return self._rate(self.label_stripped)


@dataclass(frozen=True)
class E5Result:
    rows: tuple[E5Row, ...]
    seed: int
    stub: bool
    n_attack: int
    qllm_samples: int = 1

    def _total(self) -> E5Row:
        return E5Row(
            gate="Total",
            attempted=sum(r.attempted for r in self.rows),
            fast_evaded=sum(r.fast_evaded for r in self.rows),
            slow_evaded=sum(r.slow_evaded for r in self.rows),
            label_stripped=sum(r.label_stripped for r in self.rows),
        )

    def to_markdown(self) -> str:
        # Computed, not literal: this prose used to hardcode the residual counts
        # and went stale the moment the bound or a gate rule changed.
        _tot = self._total()
        total_hard, total_n = _tot.label_stripped, _tot.attempted

        def cell(k: int, rate: float | None) -> str:
            return "--" if rate is None else f"{rate:.0%} ({k})"

        def row(r: E5Row) -> str:
            label = _GATE_LABELS.get(r.gate, r.gate)
            return (
                f"| {label} | {r.attempted} | {cell(r.fast_evaded, r.fast_rate)} | "
                f"{cell(r.slow_evaded, r.slow_rate)} | "
                f"{cell(r.label_stripped, r.hard_rate)} | "
                f"{'--' if r.hard_rate is None else f'{r.hard_rate:.0%}'} |"
            )

        stub_note = (
            "\n> **Two columns are fictitious in this run** (`--stub`). Only "
            "the fast-tier column is real. The slow-tier evasion number needs "
            "a live Q-LLM, and **so does the hard-win column** -- it is scored "
            "at `full +all`, not at the fast tier, so a stub Q-LLM produces a "
            "hard-win count that is neither the deployed figure nor the "
            "`+both` one. Do not compare it against the paper's `prod.` "
            "column: that is the fast-tier number and is larger. Rerun without "
            "`--stub`, or read the live `full_ablation.md`.\n"
            if self.stub
            else ""
        )
        samples_note = (
            f"\n> The slow-tier (`+both`) column pools **{self.qllm_samples}** "
            "independent live Q-LLM samples per instance, so its rate is a "
            "multi-seed average; the `(count)` is renormalized onto the "
            "per-instance attempt base.\n"
            if self.qllm_samples > 1
            else ""
        )
        body = [row(r) for r in self.rows] + [row(self._total())]
        return "\n".join(
            [
                "# Soft-win vs hard-win decomposition by gate (E5)",
                "",
                "How far an attack gets through each gate's tiers: past the "
                "deterministic **fast** tier (soft), past the **slow** Q-LLM tier "
                "too (soft), and finally **label-stripped at a sink** (hard -- the "
                "§8.2 capability breach). Columns are nested: fast-evaded >= "
                "slow-evaded >= label-stripped.",
                stub_note,
                samples_note,
                f"Seed {self.seed}; {self.n_attack} attack instances, grouped by "
                "each attack's sink gate (`xc_*` cross-boundary chains -> G6).",
                "",
                "| Gate | Attempts | Fast-tier evaded (soft) | Slow-tier evaded "
                "(soft) | Label stripped at sink (hard, `+both`) | Hard-win "
                "rate (`+both`) |",
                "|---|---|---|---|---|---|",
                *body,
                "",
                "_Cells show `rate (count)`. Fast-tier evaded = admitted by the "
                "deterministic tier (`full PALISADE`); slow-tier evaded = "
                "admitted with the Q-LLM tier on too (`full +all`); label "
                "stripped = a value reached a high-privilege sink without the "
                "taint its provenance implies (a hard win). **The hard column "
                "is scored at `full +all`, like the slow-tier column beside "
                "it, not at the fast tier** -- it is the `+all` figure, not "
                f"the fast-tier one; the Total row above carries the count "
                f"({total_hard} of {total_n}). The hard column "
                "concentrates in **G6**, the cross-boundary tag-drop chains, "
                "because a dropped tag is invisible to a single gate. Under "
                "the declarative bound the single-gate rows strip no label at "
                "all. The G5 row is zero because the closed-vocabulary "
                "scheduler-field rule attributes a short submission field to "
                "its source positionally, without the distinctiveness floor "
                "the containment guard needs._",
                "",
            ]
        )


def decompose(result: Any) -> tuple[E5Row, ...]:
    """Roll a `SIEGEResult`'s cells into the per-gate E5 rows."""
    tmap = template_gate_map()
    cells = result.cells

    def pool(gate: str, config: str) -> tuple[int, int, int]:
        fam = [
            c
            for c in cells
            if c.config_name == config
            and c.asr is not None
            and tmap.get(c.template) == gate
        ]
        n = sum(c.n_attack for c in fam)
        soft = sum(round(c.asr * c.n_attack) for c in fam)
        hard = sum(round((c.hard_win_rate or 0.0) * c.n_attack) for c in fam)
        return n, soft, hard

    def _rebase(count: int, trials: int, base: int) -> int:
        """Express a `+both` trial count on the per-instance attempt base.

        The `+both` column pools ``k`` live Q-LLM samples per instance, so its
        cells count ``k x instances`` trials while the fast-tier column counts
        ``instances``. Renormalize the slow-tier counts onto the shared
        per-instance base so the rate is right for ``k > 1`` (not ``k x`` too
        large) and every column reads against one denominator.
        """
        if not trials or not base:
            return count
        return round(count / trials * base)

    rows: list[E5Row] = []
    for gate in _GATE_ROWS:
        n_fast, soft_fast, _ = pool(gate, _FAST_CONFIG)
        n_slow, soft_slow, hard_slow = pool(gate, _SLOW_CONFIG)
        attempted = n_fast or n_slow
        rows.append(
            E5Row(
                gate=gate,
                attempted=attempted,
                fast_evaded=soft_fast,
                slow_evaded=_rebase(soft_slow, n_slow, attempted),
                label_stripped=_rebase(hard_slow, n_slow, attempted),
            )
        )
    return tuple(rows)


async def _arun(
    model: Any,
    *,
    max_instances: int | None,
    stub: bool,
    qllm_samples: int = 1,
    bound: str = "declarative",
) -> E5Result:
    result, _stats = await run_full_ablation(
        model,
        max_instances=max_instances,
        qllm_samples=qllm_samples,
        bound=bound,
    )
    return E5Result(
        rows=decompose(result),
        seed=result.seed,
        stub=stub,
        n_attack=result.n_attack,
        qllm_samples=max(1, int(qllm_samples)),
    )


def write_results_doc(result: E5Result, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "soft_hard_decomposition_e5.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="E5 soft/hard decomposition by gate (re-views the ablation)."
    )
    parser.add_argument("--stub", action="store_true", help="Offline TestModel Q-LLM.")
    parser.add_argument("--max", type=int, default=None, help="Instance subset.")
    parser.add_argument(
        "--qllm-samples",
        type=int,
        default=1,
        help="Independent live Q-LLM samples pooled into the +both slow-tier "
        "column (matches full_ablation --qllm-samples; smooths single-draw variance).",
    )
    parser.add_argument("--model", default=None, help=f"Q-LLM (default {DEFAULT_QLLM_MODEL}).")
    parser.add_argument(
        "--openai-base-url",
        default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL,
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
    parser.add_argument("--report-out", default=None, metavar="PATH")
    args = parser.parse_args(argv)

    if args.stub:
        from pydantic_ai.models.test import TestModel

        model: Any = TestModel()
    else:
        model = args.model or DEFAULT_QLLM_MODEL
        if isinstance(model, str) and model.startswith("openai:"):
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url

    result = asyncio.run(
        _arun(
            model,
            max_instances=args.max,
            stub=args.stub,
            qllm_samples=args.qllm_samples,
            bound=args.bound,
        )
    )
    print(f"wrote {write_results_doc(result, args.report_out)}")
    print(result.to_markdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
