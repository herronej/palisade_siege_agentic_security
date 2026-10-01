"""
Prompt-only monitor on the static corpus (E1a).

R1: "did we need to write any code at all? Could this all just be a prompt?"
This runs the answer's static half. ``siege.redteam.baselines.prompt_monitor``
holds the frozen policy -- PALISADE's in-code policy as written out by an agent
given the source, the contract library and the paper -- and a monitor gate
that is the entire defense: every action in the 205 attack instances and the
181 benign controls goes to a model reading that policy, which sees the action
and the session so far and never a capability label.

Each **draw** is one independent pass: a fresh model agent with its own cache,
so identical prompts within a draw are answered once and draws differ only by
the provider's sampling -- the same design as the ``+slow`` / ``+all`` columns
(``siege.full_ablation``). Run the draws as parallel processes, then pool:

    uv run python -m tools.prompt_monitor_baseline --draw 0 --out docs/palisade/e1a/static_draw_0.json
    ...                                            --draw 4 --out docs/palisade/e1a/static_draw_4.json
    uv run python -m tools.prompt_monitor_baseline \\
        --pool docs/palisade/e1a/static_draw_*.json \\
        --report-out docs/palisade/prompt_monitor_baseline.md

Needs ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` for the served model. The taint
bound is irrelevant here -- no configuration without the capability predicate
reads it, and the hard-win scorer reads the recorded labels either way -- so
the run uses the production bound for consistency with Table II.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from siege import SessionRunner, load_instances
from siege.ablation_matrix import (
    CUMULATIVE_CONFIGS,
    MONITOR_GATE_IDS,
    PROMPT_MONITOR_CONFIG,
)
from siege.paths import CORPUS_DIR
from siege.redteam.baselines.prompt_monitor import (
    build_monitor_agent,
    monitor_prompt_sha256,
    policy_sha256,
)
from siege.report_suites import ReportSuite, suite_for
from siege.served import describe as served_endpoint
from siege.served import hosts_label
from siege.served import install as install_served_routing
from siege.served import served_model_spec
from siege.scorer import score_trace
from tools.benign_fpr import clopper_pearson
from tools.closure_attribution import is_sink_capable

__all__ = ["RetryingCachingMonitor", "default_model", "run_draw", "pool", "to_markdown"]

def default_model() -> str:
    """The served model as a pydantic-ai spec, resolved when a run starts."""
    return served_model_spec()


class RetryingCachingMonitor:
    """Memoizes ``run(prompt)`` and retries a failed call before giving up.

    A call that still fails after ``attempts`` is re-raised, and the session
    runner scores it fail-closed. Transport retries only reduce that
    infrastructure noise; they never change an answer the model gave.
    """

    def __init__(self, inner: Any, stats: dict[str, int], attempts: int = 3) -> None:
        self._inner = inner
        self._stats = stats
        self._attempts = attempts
        self._cache: dict[str, Any] = {}
        self.error_kinds: collections.Counter = collections.Counter()

    async def run(self, prompt: str) -> Any:
        if prompt in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[prompt]
        last: BaseException | None = None
        for attempt in range(self._attempts):
            try:
                result = await self._inner.run(prompt)
            except Exception as exc:  # noqa: BLE001 - scored fail-closed upstream
                last = exc
                if attempt + 1 < self._attempts:
                    self._stats["retries"] += 1
                    await asyncio.sleep(2 ** attempt)
                continue
            self._stats["model_calls"] += 1
            self._cache[prompt] = result
            return result
        self._stats["errors"] += 1
        assert last is not None
        # Whether an error is load or content decides how to read it: a
        # transport failure is noise to correct, a model that cannot answer an
        # adversarial prompt is the monitor's own fail-closed behaviour.
        # Kept off ``stats``, whose values callers diff as integers.
        kind = f"{type(last).__name__}: {str(last)[:240]}"
        self.error_kinds[kind] += 1
        logging.getLogger(__name__).warning("monitor call failed after retries: %s", kind)
        raise last


def _family(template: str) -> str:
    return "XC" if template.startswith("xc_") else template.split("_", 1)[0].upper()


async def run_draw(draw: int, *, model: str, concurrency: int) -> dict[str, Any]:
    stats = {"model_calls": 0, "cache_hits": 0, "retries": 0, "errors": 0}
    monitor = RetryingCachingMonitor(build_monitor_agent(model), stats)
    agents = {gid: monitor for gid in MONITOR_GATE_IDS}
    instances = load_instances(CORPUS_DIR)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    started = time.time()

    async def one(inst) -> dict[str, Any]:
        nonlocal done
        async with sem:
            runner = SessionRunner(quarantine_agents=agents, bound="production")
            trace = await runner.run(inst, PROMPT_MONITOR_CONFIG)
        score = score_trace(trace, inst)
        utility = [r for r in trace.actions if r.is_utility]
        done += 1
        if done % 25 == 0:
            print(
                f"[e1a draw {draw}] {done}/{len(instances)} "
                f"({time.time() - started:.0f}s) {stats}",
                file=sys.stderr, flush=True,
            )
        return {
            "instance_id": inst.instance_id,
            "template": inst.template,
            "kind": inst.kind,
            "suite": suite_for(inst.template).value,
            "family": _family(inst.template),
            "sink_capable": inst.kind == "attack" and is_sink_capable(inst.template),
            "attack_success": bool(score.attack_success),
            "hard_win": bool(score.hard_win),
            "benign_blocked": bool(utility) and not utility[0].allowed,
            # On an attack session: did the monitor deny the instance's own
            # legitimate task? (PALISADE's gates deny none; utility_dos.md.)
            "utility_blocked": inst.kind == "attack" and any(
                not r.allowed for r in utility
            ),
            "denials": [
                r.reason for r in trace.actions if not r.allowed
            ][:4],
        }

    rows = await asyncio.gather(*(one(i) for i in instances))
    return {
        "draw": draw,
        "model": model,
        "policy_sha256": policy_sha256(),
        "monitor_prompt_sha256": monitor_prompt_sha256(),
        "config": PROMPT_MONITOR_CONFIG.name,
        "bound": "production",
        "endpoint": served_endpoint(),
        "elapsed_s": round(time.time() - started, 1),
        "stats": stats,
        "error_kinds": dict(monitor.error_kinds),
        "rows": rows,
    }


# -----------------------------------------------------------------
# Pooling
# -----------------------------------------------------------------


@dataclass
class DrawSummary:
    draw: int
    security_soft: int
    security_n: int
    capable_hard: int
    capable_n: int
    misuse_soft: int
    misuse_n: int
    science_soft: int
    science_hard: int
    science_n: int
    benign_blocked: int
    benign_n: int
    attack_utility_blocked: int
    attack_n: int
    stats: dict[str, int]
    elapsed_s: float


def _summarize(d: dict[str, Any]) -> DrawSummary:
    rows = d["rows"]
    sec = [r for r in rows if r["kind"] == "attack" and r["suite"] == ReportSuite.SECURITY.value]
    cap = [r for r in sec if r["sink_capable"]]
    mis = [r for r in rows if r["suite"] == ReportSuite.MISUSE_SAFETY.value]
    sci = [r for r in rows if r["suite"] == ReportSuite.SCIENCE_CORRECTNESS.value]
    ben = [r for r in rows if r["kind"] == "benign"]
    return DrawSummary(
        draw=d["draw"],
        security_soft=sum(r["attack_success"] for r in sec),
        security_n=len(sec),
        capable_hard=sum(r["hard_win"] for r in cap),
        capable_n=len(cap),
        misuse_soft=sum(r["attack_success"] for r in mis),
        misuse_n=len(mis),
        science_soft=sum(r["attack_success"] for r in sci),
        science_hard=sum(r["hard_win"] for r in sci),
        science_n=len(sci),
        benign_blocked=sum(r["benign_blocked"] for r in ben),
        benign_n=len(ben),
        attack_utility_blocked=sum(
            r.get("utility_blocked", False) for r in rows if r["kind"] == "attack"
        ),
        attack_n=sum(1 for r in rows if r["kind"] == "attack"),
        stats=d["stats"],
        elapsed_s=d["elapsed_s"],
    )


def pool(paths: list[Path]) -> dict[str, Any]:
    draws = [json.loads(p.read_text()) for p in sorted(paths)]
    shas = {d["policy_sha256"] for d in draws}
    if len(shas) != 1:
        raise SystemExit(f"draws were run against different policies: {shas}")
    return {"draws": draws, "summaries": [_summarize(d) for d in draws]}


async def _full_reference() -> dict[str, int]:
    """``full`` on the same corpus, offline, for the side-by-side."""
    full = CUMULATIVE_CONFIGS[-1]
    out = collections.Counter()
    for inst in load_instances(CORPUS_DIR):
        trace = await SessionRunner(bound="production").run(inst, full)
        if inst.kind == "benign":
            util = [r for r in trace.actions if r.is_utility]
            out["benign_blocked"] += bool(util) and not util[0].allowed
            continue
        score = score_trace(trace, inst)
        if suite_for(inst.template) is ReportSuite.SECURITY:
            out["security_soft"] += bool(score.attack_success)
            if is_sink_capable(inst.template):
                out["capable_hard"] += bool(score.hard_win)
    return dict(out)


def _mean_range(values: list[int]) -> str:
    mean = statistics.fmean(values)
    return f"{mean:.1f} [{min(values)}, {max(values)}]"


def to_markdown(pooled: dict[str, Any], full: dict[str, int]) -> str:
    draws, sums = pooled["draws"], pooled["summaries"]
    k = len(sums)
    s0 = sums[0]
    sec_soft = [s.security_soft for s in sums]
    cap_hard = [s.capable_hard for s in sums]
    ben = [s.benign_blocked for s in sums]
    pooled_ben = sum(ben)
    lo, hi = clopper_pearson(pooled_ben, s0.benign_n * k)
    errors = sum(s.stats.get("errors", 0) for s in sums)
    calls = sum(s.stats.get("model_calls", 0) for s in sums)

    out = [
        "<!-- Generated report. Source module: tools.prompt_monitor_baseline. "
        "Regenerate with `uv run python -m tools.prompt_monitor_baseline --pool ...`. -->",
        "",
        "# Prompt-only monitor on the static corpus (E1a)",
        "",
        f"R1's baseline: PALISADE's in-code policy written out as instructions by "
        f"an agent given the source, the contract library and the paper, frozen "
        f"(`policy.md`, sha256 `{draws[0]['policy_sha256'][:16]}…`), and used as "
        f"the **entire defense**. A model (`{draws[0]['model']}`, provider-default "
        f"sampling, as the `+slow` columns ran) reads the policy and adjudicates "
        f"every action in every session, seeing the action and the session so "
        f"far and no capability label. No PALISADE gate, capability bound, trust "
        f"scorer, contract or judge runs. {k} independent draws.",
        "",
        f"Served by {hosts_label(draws)}.",
        "",
        "## Headline",
        "",
        "| Condition | Security soft wins (of 150) | Sink-capable hard wins (of 32) | Benign controls blocked (of 181) |",
        "|---|---|---|---|",
        f"| **Prompt-only monitor**, mean [range] over {k} draws | "
        f"{_mean_range(sec_soft)} | {_mean_range(cap_hard)} | {_mean_range(ben)} |",
        f"| PALISADE `full` (fast tier, offline, recomputed) | "
        f"{full.get('security_soft', 0)} | {full.get('capable_hard', 0)} | "
        f"{full.get('benign_blocked', 0)} |",
        "| PALISADE `full +slow` (manuscript Table I/§V-C) | -- | 5 | -- |",
        "| PALISADE `full +all` (manuscript §V-C) | 6 (4.0%) | 4 | -- |",
        "",
        f"Pooled benign block rate: **{pooled_ben}/{s0.benign_n * k} = "
        f"{pooled_ben / (s0.benign_n * k):.1%}** (exact 95% CI "
        f"{lo:.1%}–{hi:.1%}; the {k} draws are repeated measurements of the "
        "same 181 controls, so this interval is optimistic).",
        "",
        f"Model health: {calls} model calls, "
        f"{sum(s.stats.get('cache_hits', 0) for s in sums)} cache hits, "
        f"{sum(s.stats.get('retries', 0) for s in sums)} transport retries, "
        f"**{errors} unrecovered error(s)**"
        + (", each scored fail-closed (so the monitor's rates above are, if "
           "anything, overstated)." if errors else "."),
        "",
        "## Per draw",
        "",
        "| Draw | Security soft /150 | Sink-capable hard /32 | Misuse soft /40 | "
        "Science soft /15 | Science hard /15 | Benign blocked /181 | "
        "Attack sessions losing their legitimate task /205 | Calls | Errors | Wall (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sums:
        out.append(
            f"| {s.draw} | {s.security_soft} | {s.capable_hard} | {s.misuse_soft} | "
            f"{s.science_soft} | {s.science_hard} | {s.benign_blocked} | "
            f"{s.attack_utility_blocked} | "
            f"{s.stats.get('model_calls', 0)} | {s.stats.get('errors', 0)} | "
            f"{s.elapsed_s:.0f} |"
        )
    out += [
        "",
        "The legitimate-task column counts attack sessions in which the monitor "
        "**denied** the instance's own benign action; PALISADE's gates deny "
        "none (`utility_dos.md`). Losses to a session the monitor ended itself "
        "after a SEV1 are included, since the monitor issues those denials.",
    ]

    # Family table, pooled over draws (Table I's rows).
    fam_n: dict[str, int] = collections.Counter()
    fam_soft: dict[str, int] = collections.Counter()
    fam_hard: dict[str, int] = collections.Counter()
    for d in draws:
        for r in d["rows"]:
            if r["kind"] != "attack" or r["suite"] != ReportSuite.SECURITY.value:
                continue
            fam_n[r["family"]] += 1
            fam_soft[r["family"]] += r["attack_success"]
            fam_hard[r["family"]] += r["hard_win"]
    out += [
        "",
        f"## By family (security partition, pooled over {k} draws)",
        "",
        "| Family | Instances | Soft win | Hard win (count per draw) |",
        "|---|---|---|---|",
    ]
    for fam in sorted(fam_n):
        n = fam_n[fam] // k
        out.append(
            f"| {fam} | {n} | {fam_soft[fam] / fam_n[fam]:.0%} | "
            f"{fam_hard[fam] / k:.1f} |"
        )

    # Which sink-capable instances the monitor lets through, and how often.
    freq: dict[str, int] = collections.Counter()
    tmpl: dict[str, str] = {}
    for d in draws:
        for r in d["rows"]:
            if r["sink_capable"]:
                tmpl[r["instance_id"]] = r["template"]
                freq[r["instance_id"]] += r["hard_win"]
    by_template: dict[str, list[int]] = collections.defaultdict(list)
    for iid, n in freq.items():
        by_template[tmpl[iid]].append(n)
    out += [
        "",
        f"## Sink-capable classes: hard wins per instance, out of {k} draws",
        "",
        "| Class | Instances | Hard wins per instance |",
        "|---|---|---|",
    ]
    for t in sorted(by_template):
        vals = sorted(by_template[t], reverse=True)
        out.append(f"| `{t}` | {len(vals)} | {', '.join(map(str, vals))} |")

    # Benign blocks, most frequent first.
    bfreq: dict[str, int] = collections.Counter()
    breason: dict[str, str] = {}
    for d in draws:
        for r in d["rows"]:
            if r["kind"] == "benign" and r["benign_blocked"]:
                bfreq[r["instance_id"]] += 1
                breason.setdefault(r["instance_id"], (r["denials"] or [""])[0])
    out += [
        "",
        f"## Benign controls blocked (draws blocking, of {k})",
        "",
        "| Control | Draws | First denial reason |",
        "|---|---|---|",
    ]
    for iid, n in sorted(bfreq.items(), key=lambda kv: (-kv[1], kv[0]))[:30]:
        reason = breason[iid].replace("|", "/")[:160]
        out.append(f"| `{iid}` | {n} | {reason} |")
    if len(bfreq) > 30:
        out.append(f"| … {len(bfreq) - 30} more | | |")

    out += [
        "",
        "## Scope",
        "",
        "- The replay produces no tool-return or retrieved content. The monitor "
        "sees each action's own arguments and the actions before it; an "
        "injection is present in the trace only as the capability label it is "
        "denied. PALISADE's gates see the same content plus that label.",
        "- The replayed actions do not adapt to a denial: a later authored "
        "action is adjudicated whether or not an earlier one was blocked, as "
        "for every other configuration.",
        "- `full` is recomputed here; the `+slow` and `+all` rows are quoted "
        "from the manuscript, which pools the same number of draws.",
    ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draw", type=int, help="run one independent draw")
    ap.add_argument("--out", type=Path, help="JSON path for --draw")
    ap.add_argument("--model", default=None,
                    help="pydantic-ai spec (default: SIEGE_SERVED_MODEL as openai-chat:...)")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--pool", nargs="+", type=Path, help="draw JSON files to pool")
    ap.add_argument("--report-out", type=Path)
    args = ap.parse_args(argv)
    install_served_routing()
    args.model = args.model or default_model()

    if args.draw is not None:
        if args.out is None:
            ap.error("--draw needs --out")
        result = asyncio.run(
            run_draw(args.draw, model=args.model, concurrency=args.concurrency)
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
        s = _summarize(result)
        print(
            f"[e1a draw {args.draw}] wrote {args.out}: security soft "
            f"{s.security_soft}/{s.security_n}, capable hard "
            f"{s.capable_hard}/{s.capable_n}, benign blocked "
            f"{s.benign_blocked}/{s.benign_n}, stats {s.stats}"
        )
        return 0
    if args.pool:
        pooled = pool(args.pool)
        md = to_markdown(pooled, asyncio.run(_full_reference()))
        if args.report_out:
            args.report_out.write_text(md)
            print(f"wrote {args.report_out}")
        else:
            print(md)
        return 0
    ap.error("pass --draw N --out PATH, or --pool FILES")
    return 2


if __name__ == "__main__":
    from siege.served import exit_on_refusal

    sys.exit(exit_on_refusal(main))
