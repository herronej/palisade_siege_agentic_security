"""
PAIR and TAP against the prompt-only monitor (E1a, adaptive half).

The same served attacker, budget, access tier and 32 sink-capable instances as
Table II (``tools.optimizer_hardwin``), pointed at the prompt-only monitor
instead of PALISADE. The monitor is wired as the driver's in-loop slow tier --
the attacker sees its verdict and refines against it -- over a configuration
with no PALISADE layer at all (``PROMPT_MONITOR_CONFIG``).

Only two of the driver's quantities mean anything for a monitor, and this
reports only those:

* the **undefended** control (32/32), which the driver establishes offline; and
* the per-seed **hard-win set with the monitor in the loop**.

The driver's fast-tier columns (declarative, authored-production, converted)
are computed without the slow tier, which for this configuration is no defense
at all, so they read 32/32 and are deliberately not reported. The
non-optimized comparison point is the authored payload under the monitor,
which ``tools.prompt_monitor_baseline`` measures over its static draws.

Seeds can run as separate processes (one ``--seeds`` value each) and be pooled.
A process owns one monitor cache, so seeds in one process share verdicts on
identical prompts, as the ``+all`` run's seeds did; seeds in separate processes
are independent draws of the monitor as well as of the attacker.

    uv run python -m tools.prompt_monitor_adaptive --strategy pair --seeds 0 \\
        --out docs/palisade/e1a/pair_seed0.json
    uv run python -m tools.prompt_monitor_adaptive --pool docs/palisade/e1a/*_seed*.json \\
        --static docs/palisade/e1a/static_draw_*.json \\
        --report-out docs/palisade/prompt_monitor_adaptive.md
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from siege.ablation_matrix import MONITOR_GATE_IDS, PROMPT_MONITOR_CONFIG
from siege.redteam.llm_optimizer import Strategy
from siege.redteam.baselines.prompt_monitor import (
    build_monitor_agent,
    monitor_prompt_sha256,
    policy_sha256,
)
from siege.redteam.access import AccessTier
from siege.served import describe as served_endpoint
from siege.served import hosts_label
from siege.served import install as install_served_routing
from siege.served import served_model_name
from tools.optimizer_hardwin import SlowTier, run_optimizer_hardwin_multiseed
from tools.prompt_monitor_baseline import RetryingCachingMonitor, default_model

__all__ = ["run", "to_markdown"]


def run(
    *,
    strategy: str,
    seeds: list[int],
    budget: int,
    proposer_model: str,
    monitor_model: str,
    max_instances: int | None,
    templates: list[str] | None = None,
) -> dict[str, Any]:
    stats = {"model_calls": 0, "cache_hits": 0, "retries": 0, "errors": 0, "salvaged": 0}
    monitor = RetryingCachingMonitor(build_monitor_agent(monitor_model), stats)
    slow = SlowTier(
        quarantine_agents={gid: monitor for gid in MONITOR_GATE_IDS},
        judge=None,
        label="prompt-only monitor",
        stats=stats,
    )
    started = time.time()
    report = run_optimizer_hardwin_multiseed(
        strategy=Strategy(strategy),
        budget=budget,
        tier=AccessTier.BLACK_BOX,
        seeds=tuple(seeds),
        model_name=proposer_model,
        max_instances=max_instances,
        slow_tier=slow,
        full_config=PROMPT_MONITOR_CONFIG,
        **({"templates": tuple(templates)} if templates else {}),
    )
    return {
        "strategy": strategy,
        "budget": budget,
        "proposer_model": proposer_model,
        "monitor_model": monitor_model,
        "policy_sha256": policy_sha256(),
        "monitor_prompt_sha256": monitor_prompt_sha256(),
        "scored_n": report.scored_n,
        "endpoint": served_endpoint(),
        #: None for a full sweep; a class list for a correction run, whose
        #: outcomes replace that class in the matching full run (see ``merge``).
        "templates": list(templates) if templates else None,
        "elapsed_s": round(time.time() - started, 1),
        "monitor_stats": stats,
        "error_kinds": dict(monitor.error_kinds),
        "seeds": [
            {
                "seed": sr.seed,
                "hard_ids": list(sr.slow_hard_ids or ()),
                "monitor_stats_delta": sr.slow_stats or {},
                "proposer_fallbacks": _fallbacks(sr.fallback),
            }
            for sr in report.seeds
        ],
    }


def _fallbacks(counter: Any) -> dict[str, int]:
    """The proposer's structured-output health, whatever the counter exposes."""
    if counter is None:
        return {}
    if hasattr(counter, "as_dict"):
        return dict(counter.as_dict())
    return {
        k: v for k, v in vars(counter).items()
        if isinstance(v, int) and not k.startswith("_")
    }


def merge(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold correction runs into the full runs they correct.

    A correction run re-sweeps whole classes for one (strategy, seed) whose
    full run logged monitor errors -- errors are scored fail-closed, so they can
    only suppress a hard win. The class is replaced as a unit, chosen by where
    the errors fell and never by outcome. Returns the merged full runs and one
    note per correction.
    """
    full = [r for r in runs if not r.get("templates")]
    corrections = [r for r in runs if r.get("templates")]
    notes: list[str] = []
    for c in corrections:
        for cs in c["seeds"]:
            target = next(
                (
                    (r, s)
                    for r in full if r["strategy"] == c["strategy"]
                    for s in r["seeds"] if s["seed"] == cs["seed"]
                ),
                None,
            )
            if target is None:
                raise SystemExit(
                    f"correction {c['strategy']} seed {cs['seed']} has no full run"
                )
            r, s = target
            # Template ids are "<class>_<variant>"; instance ids start with the
            # class's short prefix, e.g. xc_5_history_forgery -> xc_5_.
            short = tuple("_".join(t.split("_")[:2]) + "_" for t in c["templates"])
            kept = [i for i in s["hard_ids"] if not i.startswith(short)]
            before = len(s["hard_ids"])
            old_err = s["monitor_stats_delta"].get("errors", 0)
            s["hard_ids"] = kept + list(cs["hard_ids"])
            s.setdefault("corrections", []).append(
                {
                    "templates": c["templates"],
                    "errors_before": old_err,
                    "errors_in_rerun": cs["monitor_stats_delta"].get("errors", 0),
                    "hard_before": before,
                    "hard_after": len(s["hard_ids"]),
                }
            )
            notes.append(
                f"{c['strategy'].upper()} seed {cs['seed']}: "
                f"{', '.join(c['templates'])} re-swept after {old_err} monitor "
                f"error(s) in the full run; {cs['monitor_stats_delta'].get('errors', 0)} "
                f"in the rerun. Hard wins {before} -> {len(s['hard_ids'])}."
            )
    return full, notes


def _static_nonopt(paths: list[Path]) -> list[int]:
    """Authored-payload hard wins on the sink-capable set, one per static draw."""
    out = []
    for p in sorted(paths):
        d = json.loads(p.read_text())
        out.append(sum(r["hard_win"] for r in d["rows"] if r["sink_capable"]))
    return out


def _mean_range(values: list[int] | list[float]) -> str:
    if not values:
        return "--"
    return f"{statistics.fmean(values):.1f} [{min(values)}, {max(values)}]"


def to_markdown(
    runs: list[dict[str, Any]],
    static: list[int],
    *,
    exclude: list[str] | None = None,
    exclusion_note: str = "",
    correction_note: str = "",
    static_report: str = "prompt_monitor_baseline.md",
    extra_rows: list[str] | None = None,
) -> str:
    """``extra_rows`` are whole table rows added under Table II's PALISADE rows,
    such as a same-host rerun of ``full``."""
    # Before merging: merge() returns only the full runs, and a correction may
    # have been served by a different host.
    hosts = hosts_label(runs)
    runs, correction_notes = merge(runs)
    exclude = list(exclude or [])
    shas = {r["policy_sha256"] for r in runs}
    if len(shas) != 1:
        raise SystemExit(f"runs used different policies: {shas}")
    by_strategy: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    health: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in runs:
        for s in r["seeds"]:
            by_strategy[r["strategy"]].append(s)
        health[r["strategy"]].update(r["monitor_stats"])
    budget = runs[0]["budget"]
    scored = runs[0]["scored_n"]

    out = [
        "<!-- Generated report. Source module: tools.prompt_monitor_adaptive. "
        "Regenerate with `uv run python -m tools.prompt_monitor_adaptive --pool ...`. -->",
        "",
        "# PAIR and TAP against the prompt-only monitor (E1a)",
        "",
        f"Table II's attacker ({runs[0]['proposer_model']}, black box, budget "
        f"{budget}) over the {scored} sink-capable instances, with the "
        f"prompt-only monitor ({runs[0]['monitor_model']}, policy sha256 "
        f"`{runs[0]['policy_sha256'][:16]}…`) in the loop as the only defense. "
        "The attacker sees the monitor's allow/deny verdict and refines against "
        "it.",
        "",
        f"Served by {hosts}.",
        "",
        "| Condition | Non-opt. | PAIR opt. | TAP opt. |",
        "|---|---|---|---|",
        f"| Undefended | {scored}/{scored} | at ceiling | at ceiling |",
        f"| **Prompt-only monitor** | {_mean_range(static)} | "
        f"{_mean_range([len(s['hard_ids']) for s in by_strategy.get('pair', [])])} | "
        f"{_mean_range([len(s['hard_ids']) for s in by_strategy.get('tap', [])])} |",
        "| PALISADE `full` (Table II) | 13/32 | 31.0 [31, 31] | 29.8 [28, 31] |",
        "| PALISADE `+all` (Table II) | 4/32 | 29.6 [29, 30] | 28.0 [27, 29] |",
        *(extra_rows or []),
        "",
        "Non-opt. for the monitor is the authored payload over the static draws "
        f"(`{static_report}`); optimized cells are the mean over seeds "
        "with the per-seed range, as in Table II. Table II's PALISADE rows were "
        "served by the American Science Cloud.",
        "",
    ]
    if exclude:
        kept_n = scored - len(exclude)

        def kept(seed: dict[str, Any]) -> int:
            return sum(1 for i in seed["hard_ids"] if i not in exclude)

        clean = [
            (strat, s["seed"], len(s["hard_ids"]))
            for strat in sorted(by_strategy)
            for s in by_strategy[strat]
            if not s["monitor_stats_delta"].get("errors", 0)
        ]
        out[out.index("| Condition | Non-opt. | PAIR opt. | TAP opt. |"):out.index("| Condition | Non-opt. | PAIR opt. | TAP opt. |")] = [
            "**Read this table with the section below.** The optimized cells "
            "count every seed over all 32 instances, but most seeds lost the "
            "served endpoint before they finished, so for them the last "
            "instances are not valid measurements.",
            "",
        ]
        out += [
            "## Valid comparison",
            "",
            exclusion_note,
            "",
            f"Over the **{kept_n} instances every seed swept before any failure** "
            f"(all but {', '.join(f'`{i}`' for i in exclude)}):",
            "",
            f"| Strategy | Hard wins, mean [range] over seeds (of {kept_n}) |",
            "|---|---|",
        ]
        for strat in sorted(by_strategy):
            vals = [kept(s) for s in by_strategy[strat]]
            out.append(
                f"| {strat.upper()} | {_mean_range(vals)} "
                f"({statistics.fmean(vals) / kept_n:.0%}) |"
            )
        out += [
            "",
            "Seeds that finished before the endpoint failed, and so are valid over "
            "all 32: "
            + (
                "; ".join(f"{st.upper()} seed {sd}: {n}/{scored}" for st, sd, n in clean)
                if clean else "none"
            )
            + ".",
            "",
            "Table II's PALISADE rows are over all 32 instances; no per-instance "
            "PALISADE result for this set is committed, so they are not restated "
            "on the subset.",
            "",
        ]
    for strat in sorted(by_strategy):
        seeds = sorted(by_strategy[strat], key=lambda s: s["seed"])
        h = health[strat]
        out += [
            f"## {strat.upper()}",
            "",
            "| Seed | Hard wins | Monitor calls | Monitor errors |",
            "|---|---|---|---|",
        ]
        for s in seeds:
            d = s["monitor_stats_delta"]
            out.append(
                f"| {s['seed']} | {len(s['hard_ids'])}/{scored} | "
                f"{d.get('model_calls', 0)} | {d.get('errors', 0)} |"
            )
        freq = collections.Counter(i for s in seeds for i in s["hard_ids"])
        never = scored - len(freq)
        out += [
            "",
            f"Monitor health: {h.get('model_calls', 0)} calls, "
            f"{h.get('cache_hits', 0)} cache hits, {h.get('retries', 0)} "
            f"transport retries, **{h.get('errors', 0)} unrecovered error(s)** "
            "scored fail-closed.",
            "",
            f"Instances never won in any seed: {never}.",
            "",
        ]
    if correction_notes:
        out += [
            "## Error corrections",
            "",
            "A monitor call that fails after its retries is scored fail-closed, so "
            "an error can only suppress a hard win. For each seed that logged "
            "errors, the classes they fell on were re-swept and replaced as a "
            "unit -- selected by where the errors fell, not by outcome. The "
            "per-seed tables above show the merged result; their monitor columns "
            "are the full sweeps'.",
            "",
            *([correction_note, ""] if correction_note else []),
            *[f"- {n}" for n in correction_notes],
            "",
        ]
    out += [
        "## Scope",
        "",
        "- The monitor sees each replayed action and the session before it, and "
        "no capability label. The replay carries no tool-return content.",
        "- The recorded-label bound has no analogue here: no configuration "
        "without the capability predicate reads it, so the invariance Table II "
        "reports for PALISADE's predicate is not a quantity a prompt has.",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", choices=("pair", "tap"))
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--proposer-model", default=None,
                    help="default: SIEGE_SERVED_MODEL, else gpt-oss-120b")
    ap.add_argument("--monitor-model", default=None,
                    help="pydantic-ai spec (default: SIEGE_SERVED_MODEL as openai-chat:...)")
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument(
        "--templates", nargs="+", default=None, metavar="CLASS",
        help="re-sweep only these classes (a correction run; see merge())",
    )
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pool", nargs="+", type=Path)
    ap.add_argument("--static", nargs="*", type=Path, default=[])
    ap.add_argument("--report-out", type=Path)
    ap.add_argument(
        "--exclude-instances", nargs="*", default=[], metavar="ID",
        help="instances left out of the valid comparison (e.g. swept after the "
             "endpoint failed in some seeds)",
    )
    ap.add_argument("--exclusion-note", default="", help="why they are excluded")
    ap.add_argument(
        "--correction-note", default="",
        help="why the full sweeps logged the errors the corrections repair",
    )
    ap.add_argument(
        "--static-report", default="prompt_monitor_baseline.md",
        help="the report the --static draws are published in",
    )
    ap.add_argument(
        "--extra-row", action="append", default=[], metavar="ROW",
        help="a whole markdown table row to add under Table II's PALISADE rows "
             "(repeatable), e.g. a same-host rerun of full",
    )
    args = ap.parse_args(argv)
    install_served_routing()
    args.proposer_model = args.proposer_model or served_model_name()
    args.monitor_model = args.monitor_model or default_model()

    if args.pool:
        runs = [json.loads(p.read_text()) for p in args.pool]
        md = to_markdown(
            runs,
            _static_nonopt(args.static),
            exclude=args.exclude_instances,
            exclusion_note=args.exclusion_note,
            correction_note=args.correction_note,
            static_report=args.static_report,
            extra_rows=args.extra_row,
        )
        if args.report_out:
            args.report_out.write_text(md)
            print(f"wrote {args.report_out}")
        else:
            print(md)
        return 0
    if not (args.strategy and args.seeds and args.out):
        ap.error("pass --strategy, --seeds and --out, or --pool")
    result = run(
        strategy=args.strategy,
        seeds=args.seeds,
        budget=args.budget,
        proposer_model=args.proposer_model,
        monitor_model=args.monitor_model,
        max_instances=args.max_instances,
        templates=args.templates,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    hard = [len(s["hard_ids"]) for s in result["seeds"]]
    print(
        f"[e1a {args.strategy}] seeds {args.seeds}: hard wins {hard} of "
        f"{result['scored_n']}; monitor {result['monitor_stats']}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    from siege.served import exit_on_refusal

    sys.exit(exit_on_refusal(main))
