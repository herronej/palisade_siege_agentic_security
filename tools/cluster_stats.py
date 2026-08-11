"""
Cluster-robust uncertainty for the SIEGE headline rates (R4 / R2-12).

**The problem.** Every interval in the paper is an exact-binomial (or
McNemar) computed as if the 205 instances were independent draws. They
are not. They are clustered within 42 authored classes, usually five
near-identical variants apiece, generated from one template with one
variation axis. Two instances of ``b5_11`` share a template, a payload
shape, and a gate path; they are closer to one observation than to two.
Treating them as independent makes every interval too narrow and every
p-value too small, which is precisely the objection a reviewer raised.

**What this computes.** Three cluster-aware analyses, all taking the
*class* as the independent unit:

``cluster bootstrap``
    Resample the 42 classes with replacement, carry every instance of a
    drawn class, recompute the statistic; report percentile intervals.
    This is the standard nonparametric fix for clustered data and needs
    no distributional assumption.

``leave-one-class-out``
    Drop each class in turn and recompute the deployed rate. Reports the
    range and the single class that moves the headline most, which
    answers "does one authored family carry the result?" directly.

``class-level paired permutation``
    For each detector comparison, take the per-class difference in hard
    wins (PALISADE minus comparator) and permute its sign at the class
    level. Exchangeability holds at the class level under the null, which
    is the assumption McNemar makes at the instance level and should not.

**The two comparators the detector panel does not cover (R13-14).** The
fixed-classifier panel above tests the denylist and both Prompt-Guard
postures, and misses the only two comparisons that decide an argument the
paper makes:

``value-lineage peer``
    The comparison the architectural-price claim rests on --- the peer
    closes 23 of the 27 shared-sink hard wins where our deployment closes
    14. Its denominator is **not** the 205: the peer's published design
    models only four sink types, so the test runs on the 27 instances and
    the 6 classes inside them. Offline and deterministic.

``LLM judge``
    The only comparator that beats us on the static hard-win axis (9/205
    against our 16/205). Needs the served endpoint, so it is gated behind
    ``--live-judge`` and the per-instance sets it produces are written into
    the report for audit.

Run with the same ``--bound`` as the headline table (``production``).

Usage::

    python3 -m tools.cluster_stats \\
        [--bound production] [--reps 10000] [--seed 42] \\
        [--live-judge] [--judge-model gpt-oss-120b] [--report-out PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from siege import load_instances
from siege.corpus_builder import CORPUS_DIR
from tools.baseline_hardwin import _flagged_by
from tools.detector_baselines import _build_detectors, detector_benign_fpr
from tools.paired_significance import (
    _palisade_benign_blocks,
    _outcomes,
    mcnemar_exact,
)


# -----------------------------------------------------------------
# Cluster bootstrap
# -----------------------------------------------------------------


def _cluster_bootstrap(
    clusters: dict[str, list[str]],
    stat: Callable[[list[str]], float | None],
    reps: int,
    seed: int,
) -> tuple[float | None, tuple[float, float] | None]:
    """Percentile CI for ``stat`` under resampling of whole clusters.

    ``stat`` receives the flat list of instance ids in the resample and
    returns the statistic, or ``None`` when it is undefined (an empty
    conditional denominator, say); undefined replicates are dropped and
    counted against the effective replicate total.
    """
    keys = sorted(clusters)
    point = stat([i for k in keys for i in clusters[k]])
    if not keys:
        return point, None
    rng = random.Random(seed)
    vals: list[float] = []
    for _ in range(reps):
        drawn = [keys[rng.randrange(len(keys))] for _ in keys]
        ids = [i for k in drawn for i in clusters[k]]
        v = stat(ids)
        if v is not None:
            vals.append(v)
    if len(vals) < reps * 0.5:
        return point, None
    vals.sort()
    lo = vals[int(0.025 * (len(vals) - 1))]
    hi = vals[int(0.975 * (len(vals) - 1))]
    return point, (lo, hi)


def _rate(flags: dict[str, bool]) -> Callable[[list[str]], float | None]:
    def f(ids: list[str]) -> float | None:
        if not ids:
            return None
        return 100.0 * sum(1 for i in ids if flags[i]) / len(ids)

    return f


def _conditional(
    undef: dict[str, bool], cit: dict[str, bool]
) -> Callable[[list[str]], float | None]:
    """Residual among instances that produce an undefended hard win."""

    def f(ids: list[str]) -> float | None:
        den = [i for i in ids if undef[i]]
        if not den:
            return None
        return 100.0 * sum(1 for i in den if cit[i]) / len(den)

    return f


# -----------------------------------------------------------------
# Class-level permutation
# -----------------------------------------------------------------


def _perm_test(
    diffs: list[int], reps: int, seed: int
) -> tuple[float, float, int, float]:
    """Two-sided sign-flip permutation for per-class differences.

    Returns ``(observed mean difference, p, k, p_floor)`` where ``k`` is
    the number of classes with a nonzero difference and ``p_floor`` is
    the smallest two-sided p attainable with that many sign-flippable
    units, ``2 / 2**k``. Classes whose difference is zero contribute
    nothing under sign flipping, so ``k`` -- not the class count and
    certainly not the instance count -- is what powers the test. Where
    ``p_floor`` approaches the significance threshold, the design cannot
    demonstrate significance no matter how large the effect.
    """
    if not diffs:
        return 0.0, 1.0, 0, 1.0
    obs = mean(diffs)
    k = sum(1 for d in diffs if d != 0)
    p_floor = min(1.0, 2.0 / (2**k)) if k else 1.0
    rng = random.Random(seed)
    hits = 0
    for _ in range(reps):
        v = mean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(v) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (reps + 1), k, p_floor


def _perm_row(
    axis: str,
    label: str,
    denom: str,
    ours: dict[str, bool],
    theirs: dict[str, bool],
    clusters: dict[str, list[str]],
    reps: int,
    seed: int,
) -> tuple[str, str, str, float, float, float, int, float]:
    """One class-level paired row plus its instance-level McNemar cross-check.

    ``ours`` and ``theirs`` map instance id -> *bad outcome* (a surviving hard
    win, or a benign false block), so a positive mean difference is against us.
    Only ids present in ``clusters`` are scored, which is what lets the peer row
    run on its own 27-instance denominator rather than the corpus-wide one.
    """
    diffs = [
        sum(1 for i in ids if ours[i]) - sum(1 for i in ids if theirs[i])
        for ids in clusters.values()
    ]
    obs, p_perm, k, floor = _perm_test(diffs, reps, seed)
    ids = [i for v in clusters.values() for i in v]
    b = sum(1 for i in ids if theirs[i] and not ours[i])
    c = sum(1 for i in ids if ours[i] and not theirs[i])
    return (axis, label, denom, obs, p_perm, mcnemar_exact(b, c), k, floor)


async def _peer_rows(
    cit: dict[str, bool], reps: int, seed: int
) -> tuple[list[tuple[str, str, str, float, float, float, int, float]], dict[str, int]]:
    """Class-level paired tests against the runnable value-lineage peer.

    Scored on the 27 shared-sink hard wins the peer's design models, clustered
    by the 6 classes inside them. Two comparisons: our *deployment* (every
    mechanism it brings, the 14) and our *label predicate alone* (the 7). Both
    against the peer's lineage closure (the 23).
    """
    from tools.dataflow_peer import _run as _peer_run

    res = await _peer_run()
    clusters: dict[str, list[str]] = defaultdict(list)
    for v in res.verdicts:
        clusters[v.cls].append(v.instance_id)

    # Bad outcome = a hard win the mechanism leaves open. Every instance here is
    # an undefended hard win, so "not closed" and "hard win remains" coincide.
    peer = {v.instance_id: not v.lineage_closes for v in res.verdicts}
    label_only = {v.instance_id: not v.content_closes for v in res.verdicts}
    deployed = {v.instance_id: cit[v.instance_id] for v in res.verdicts}
    n = len(res.verdicts)
    denom = f"{n} shared sinks"

    rows = [
        _perm_row(
            "hard win", "value-lineage peer (vs deployed)", denom,
            deployed, peer, clusters, reps, seed,
        ),
        _perm_row(
            "hard win", "value-lineage peer (vs label predicate)", denom,
            label_only, peer, clusters, reps, seed,
        ),
    ]
    counts = {
        "n": n,
        "classes": len(clusters),
        "peer_closes": sum(1 for v in peer.values() if not v),
        "deployed_closes": sum(1 for v in deployed.values() if not v),
        "label_closes": sum(1 for v in label_only.values() if not v),
    }
    return rows, counts


async def _judge_rows(
    cit: dict[str, bool],
    cit_benign: dict[str, bool],
    atk_cls: dict[str, list[str]],
    ben_cls: dict[str, list[str]],
    reps: int,
    seed: int,
    model: str,
) -> tuple[
    list[tuple[str, str, str, float, float, float, int, float]], dict[str, Any]
]:
    """Class-level paired tests against the live LLM judge, on the full corpus.

    The judge reads text at every boundary, so unlike the peer it has no sink
    restriction and shares the detector panel's denominators.
    """
    from tools.judge_union import run_judge_union

    res = await run_judge_union(model=model)
    if not res.judge_reliable:
        raise RuntimeError(
            f"judge call-error rate {res.judge_error_rate:.1%} exceeds 2%; "
            "refusing to credit it with instances it never answered"
        )
    judge_hw = set(res.judge_hw)
    judge_ben = set(res.judge_benign_ids)
    j_hw = {i: i in judge_hw for i in cit}
    j_ben = {i: i in judge_ben for i in cit_benign}

    rows = [
        _perm_row(
            "hard win", "LLM judge", "205", cit, j_hw, atk_cls, reps, seed
        ),
        _perm_row(
            "benign block", "LLM judge", "181",
            cit_benign, j_ben, ben_cls, reps, seed,
        ),
    ]
    meta = {
        "model": model,
        "error_rate": res.judge_error_rate,
        "judge_hw": res.judge_hw,
        "judge_benign": res.judge_benign_ids,
        "judge_hw_n": len(res.judge_hw),
        "judge_benign_n": len(res.judge_benign_ids),
    }
    return rows, meta


# -----------------------------------------------------------------
# Report
# -----------------------------------------------------------------


@dataclass
class Row:
    name: str
    point: float | None
    ci: tuple[float, float] | None
    naive: str = ""


def _fmt(r: Row) -> str:
    p = "--" if r.point is None else f"{r.point:.1f}%"
    c = "--" if r.ci is None else f"[{r.ci[0]:.1f}, {r.ci[1]:.1f}]"
    return f"| {r.name} | {p} | {c} | {r.naive} |"


def to_markdown(
    rates: list[Row],
    loco: list[tuple[str, float]],
    loco_base: float,
    perms: list[tuple[str, str, float, float, float, int, float]],
    n_cls: int,
    n_ben_cls: int,
    reps: int,
    extra: list[tuple[str, str, str, float, float, float, int, float]] | None = None,
    peer_counts: dict[str, int] | None = None,
    judge_meta: dict[str, Any] | None = None,
) -> str:
    L: list[str] = []
    L.append("# Cluster-robust intervals over attack classes (R4 / R2-12)\n")
    L.append(
        f"The 205 attack instances are clustered in **{n_cls} authored "
        "classes**, 38 of them five near-identical variants of one template "
        "along one variation axis. Exact-binomial intervals treat them as "
        "independent draws, which understates uncertainty. Below, the class is "
        f"the resampling unit: {reps:,} nonparametric cluster-bootstrap "
        "replicates, percentile interval. The naive column is the "
        "instance-level exact-binomial interval the paper currently reports.\n"
    )
    L.append(
        f"**The benign control is not clustered the same way and we do not "
        f"pretend otherwise.** Its 157 diverse tasks carry 157 *distinct* "
        f"variation axes -- they are different tasks, not variants of one -- so "
        f"the instance-level interval is appropriate there. The {n_ben_cls}-way "
        "row below resamples the four authored categories plus the original "
        "workload as a conservative sensitivity check only; at five clusters "
        "the percentile interval is unstable and should not be read as the "
        "headline uncertainty.\n"
    )
    L.append("| statistic | point | cluster 95% CI | naive 95% CI |")
    L.append("|---|---|---|---|")
    for r in rates:
        L.append(_fmt(r))
    L.append(
        "\nThe cluster intervals are the honest ones. Where they are much wider "
        "than the naive interval, the instance-level figure was borrowing "
        "confidence from within-class repetition."
    )

    L.append("\n## Leave-one-class-out sensitivity\n")
    L.append(
        f"Deployed hard-win rate with each class removed in turn "
        f"(base {loco_base:.1f}%). Answers whether one authored family carries "
        "the headline.\n"
    )
    lo = min(loco, key=lambda t: t[1])
    hi = max(loco, key=lambda t: t[1])
    L.append(
        f"**Range {lo[1]:.1f}%--{hi[1]:.1f}%.** Dropping `{hi[0]}` raises it "
        f"most ({hi[1]:.1f}%); dropping `{lo[0]}` lowers it most ({lo[1]:.1f}%)."
    )
    L.append("\n| class dropped | deployed HW | delta |")
    L.append("|---|---|---|")
    for name, v in sorted(loco, key=lambda t: -abs(t[1] - loco_base))[:8]:
        L.append(f"| `{name}` | {v:.1f}% | {v - loco_base:+.1f} |")
    L.append("\n_Eight classes with the largest effect; the rest move it less._")

    L.append("\n## Class-level paired permutation vs. McNemar\n")
    L.append(
        "Per-class difference in hard wins or false blocks (PALISADE minus "
        f"comparator), sign-flipped at the class level over {reps:,} "
        "permutations. McNemar's instance-level p is shown beside it: it "
        "assumes exchangeable instances, which clustering violates, so it is "
        "the optimistic one.\n"
    )
    L.append(
        "| axis | comparator | mean class diff | k | perm p | min p | McNemar p |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for axis, label, obs, p_perm, p_mc, k, floor in perms:
        L.append(
            f"| {axis} | {label} | {obs:+.2f} | {k} | {p_perm:.4f} | "
            f"{floor:.3f} | {p_mc:.2e} |"
        )
    L.append(
        "\n`k` is the number of classes with a nonzero difference -- the units "
        "the test can actually flip -- and `min p` the smallest two-sided p "
        "attainable at that `k`. Where `min p` is near or above 0.05 the design "
        "cannot show significance whatever the effect size, so a large p there "
        "is an absence of power, not evidence of no difference. The benign rows "
        "are in that regime by construction: five clusters admit no p below "
        "0.0625."
    )
    L.append(
        "\n**The hard-win rows are the substantive result, and they do not all "
        "survive.** McNemar reads the corpus as 205 independent instances; once "
        "the class is the unit, only a handful of classes carry each "
        "comparison, and the evidence is correspondingly weaker. Any comparison "
        "whose permutation p exceeds 0.05 should be reported as directional "
        "rather than significant."
    )

    if extra:
        L.append(
            "\n## The two comparators the panel above does not cover (R13-14)\n"
        )
        L.append(
            "The panel tests fixed content classifiers. It misses the peer "
            "comparison the architectural-price argument rests on, and the judge "
            "-- the one comparator that beats us on the static hard-win axis. "
            "**Denominators differ by row and are stated, not pooled:** the "
            "value-lineage peer's published design models four sink types, so it "
            "is scored on the shared-sink hard wins and the classes inside them, "
            "not on the 205.\n"
        )
        L.append(
            "| axis | comparator | denominator | mean class diff | k | perm p | "
            "min p | McNemar p |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for axis, label, den, obs, p_perm, p_mc, k, floor in extra:
            L.append(
                f"| {axis} | {label} | {den} | {obs:+.2f} | {k} | {p_perm:.4f} "
                f"| {floor:.3f} | {p_mc:.2e} |"
            )
        L.append(
            "\nA **positive** mean class difference is against us: it counts hard "
            "wins (or benign false blocks) our mechanism leaves and the "
            "comparator closes."
        )

    if peer_counts:
        L.append(
            f"\n**Peer comparison.** On the {peer_counts['n']} shared-sink hard "
            f"wins across {peer_counts['classes']} classes, the lineage peer "
            f"closes **{peer_counts['peer_closes']}**, our deployment "
            f"**{peer_counts['deployed_closes']}**, our label predicate alone "
            f"**{peer_counts['label_closes']}**. With only "
            f"{peer_counts['classes']} classes the sign-flip design bottoms out "
            "at the `min p` shown, so read a large p as absent power rather than "
            "as evidence of no difference."
        )

    if judge_meta:
        L.append(
            f"\n**Judge comparison.** Live `{judge_meta['model']}`, call-error "
            f"rate {judge_meta['error_rate']:.1%}. Judge hard wins: "
            f"**{judge_meta['judge_hw_n']}/205**; judge benign false blocks: "
            f"**{judge_meta['judge_benign_n']}/181**. The per-instance sets are "
            "recorded below so the row is auditable without re-running the "
            "endpoint.\n"
        )
        L.append(
            "- judge hard wins: "
            + ", ".join(f"`{i}`" for i in judge_meta["judge_hw"])
        )
        L.append(
            "- judge benign false blocks: "
            + (
                ", ".join(f"`{i}`" for i in judge_meta["judge_benign"])
                or "_none_"
            )
        )

    L.append("\n_Generated by `tools.cluster_stats`._")
    return "\n".join(L) + "\n"


async def _arun(
    bound: str,
    reps: int,
    seed: int,
    model_name: str | None,
    live_judge: bool = False,
    judge_model: str = "gpt-oss-120b",
) -> str:
    undef, cit, insts = await _outcomes(bound)
    cit_benign = await _palisade_benign_blocks(bound)

    # ---- cluster maps -------------------------------------------------
    atk_cls: dict[str, list[str]] = defaultdict(list)
    for iid, inst in insts.items():
        atk_cls[inst.template].append(iid)
    ben_cls: dict[str, list[str]] = defaultdict(list)
    for inst in load_instances(CORPUS_DIR):
        if inst.is_attack or inst.instance_id not in cit_benign:
            continue
        axis = getattr(inst, "variation_axis", "") or ""
        # benign_diverse axes are unique per instance, so the coarsest
        # honest grouping is the authored category prefix (a/b/c/d);
        # the original workload is its own stratum.
        key = (
            f"benign_diverse:{axis.split('_', 1)[0]}"
            if inst.template == "benign_diverse" and axis
            else inst.template
        )
        ben_cls[key].append(inst.instance_id)

    rates: list[Row] = []
    p, ci = _cluster_bootstrap(atk_cls, _rate(undef), reps, seed)
    rates.append(Row("undefended hard win (205)", p, ci, "[13.0, 24.0]"))
    p, ci = _cluster_bootstrap(atk_cls, _rate(cit), reps, seed)
    rates.append(Row("deployed hard win (205)", p, ci, "[4.5, 12.4]"))
    p, ci = _cluster_bootstrap(atk_cls, _conditional(undef, cit), reps, seed)
    rates.append(
        Row("residual | undefended HW (37)", p, ci, "[27.1, 60.5]")
    )
    p, ci = _cluster_bootstrap(ben_cls, _rate(cit_benign), reps, seed)
    rates.append(Row("benign false-positive (181)", p, ci, "[1.9, 8.5]"))

    # ---- leave-one-class-out -----------------------------------------
    all_ids = [i for ids in atk_cls.values() for i in ids]
    base = 100.0 * sum(1 for i in all_ids if cit[i]) / len(all_ids)
    loco: list[tuple[str, float]] = []
    for cls in sorted(atk_cls):
        keep = [i for c, ids in atk_cls.items() if c != cls for i in ids]
        loco.append((cls, 100.0 * sum(1 for i in keep if cit[i]) / len(keep)))

    # ---- class-level permutation vs detectors -------------------------
    perms: list[tuple[str, str, float, float, float, int, float]] = []
    detectors, available, resolved = _build_detectors(
        model_name=model_name or "meta-llama/Prompt-Guard-86M"
    )
    if not available:
        print(
            f"[cluster] Prompt-Guard weights unavailable for {resolved!r}; "
            "regex denylist only."
        )
    for det, label, _kind in detectors:
        det_hw = {i: undef[i] and not _flagged_by(det, insts[i]) for i in undef}
        diffs = [
            sum(1 for i in ids if cit[i]) - sum(1 for i in ids if det_hw[i])
            for ids in atk_cls.values()
        ]
        obs, p_perm, k, floor = _perm_test(diffs, reps, seed)
        b = sum(1 for i in undef if det_hw[i] and not cit[i])
        c = sum(1 for i in undef if cit[i] and not det_hw[i])
        perms.append(
            ("hard win", label, obs, p_perm, mcnemar_exact(b, c), k, floor)
        )

        _k, _n, flagged_ids = detector_benign_fpr(det)
        flagged = set(flagged_ids)
        det_ben = {i: i in flagged for i in cit_benign}
        bdiffs = [
            sum(1 for i in ids if cit_benign[i]) - sum(1 for i in ids if det_ben[i])
            for ids in ben_cls.values()
        ]
        obs, p_perm, k, floor = _perm_test(bdiffs, reps, seed)
        b = sum(1 for i in cit_benign if det_ben[i] and not cit_benign[i])
        c = sum(1 for i in cit_benign if cit_benign[i] and not det_ben[i])
        perms.append(
            ("benign block", label, obs, p_perm, mcnemar_exact(b, c), k, floor)
        )

    # ---- R13-14: the two comparators the panel does not cover -----------
    extra, peer_counts = await _peer_rows(cit, reps, seed)
    judge_meta: dict[str, Any] | None = None
    if live_judge:
        jrows, judge_meta = await _judge_rows(
            cit, cit_benign, atk_cls, ben_cls, reps, seed, judge_model
        )
        extra = extra + jrows
    else:
        print(
            "[cluster] judge rows skipped (pass --live-judge to score the "
            "served comparator)."
        )

    return to_markdown(
        rates, loco, base, perms, len(atk_cls), len(ben_cls), reps,
        extra=extra, peer_counts=peer_counts, judge_meta=judge_meta,
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="Cluster-robust SIEGE intervals.")
    ap.add_argument("--bound", default="production", choices=["production", "declarative"])
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None)
    ap.add_argument(
        "--live-judge",
        action="store_true",
        help="Score the LLM-judge comparator (needs OPENAI_BASE_URL + key).",
    )
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args(argv)

    text = asyncio.run(
        _arun(
            args.bound,
            args.reps,
            args.seed,
            args.model,
            live_judge=args.live_judge,
            judge_model=args.judge_model,
        )
    )
    if args.report_out:
        args.report_out.write_text(text)
        print(f"wrote {args.report_out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
