"""
SIEGE evaluation runner (the shared ``--gate siege`` CLI
backend).

Threads every loaded instance through the 6-configuration B4-first
cumulative ablation matrix using the ``SessionRunner`` + scorer,
and returns a ``SIEGEResult`` the report formatter turns into the
per-(boundary, template) cell tables.

Unlike the per-gate runners, this one is *corpus-driven*: it loads
instance documents off disk (the bundled fixtures by default, or
a directory of instances) rather than generating scenarios in
code. That gives one harness shape every template plugs instances into.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from siege import (
    CUMULATIVE_CONFIGS,
    AblationConfig,
    Cell,
    InProcessTurnBuffer,
    InstanceScore,
    SessionRunner,
    aggregate_cells,
    load_instances,
    score_trace,
)
from siege.corpus_builder import CORPUS_DIR
from siege.scorer import METRIC_DEFINITIONS


@dataclass(frozen=True)
class SIEGEResult:
    """Top-level result for the report formatter.

    Fields:
        configs: the ablation configurations measured (the 6 cumulative
            B4-first ones by default).
        cells: per-(boundary, template, config) aggregated cells.
        scores: every per-instance score (kept so a caller can compute
            extra breakdowns, e.g. hard-win audits).
        n_instances / n_attack / n_benign: corpus bookkeeping.
        seed: forwarded for report provenance (the harness itself is
            deterministic; the seed is recorded for parity with the
            other eval reports).
        metric_definitions: the pinned BU / UA / ASR definitions.
    """

    configs: tuple[AblationConfig, ...]
    cells: tuple[Cell, ...]
    scores: tuple[InstanceScore, ...]
    n_instances: int
    n_attack: int
    n_benign: int
    seed: int
    metric_definitions: dict[str, str]
    #: WI11 contract-coverage over the corpus's declared scientific claims.
    contract_coverage: float | None = None
    n_claims: int = 0
    n_claims_covered: int = 0
    n_claims_violated: int = 0
    #: How many independent live Q-LLM samples the ``+Q-LLM`` / ``+both``
    #: columns pool per instance (1 = single draw). >1 means those columns'
    #: cell n (and Wilson CI) reflect ``qllm_samples × instances`` -- the
    #: multi-seed average that smooths the single-live-pass variance.
    qllm_samples: int = 1
    #: Per-template ``(caught, total)`` contract verdicts over the
    #: science-correctness cluster's attack instances. ``caught`` = the
    #: contract layer flagged the poisoned claim (covered + violation); the
    #: residual ``total - caught`` is the contract-evaded ASR (the C-evade
    #: mission risk). Config-independent -- contracts evaluate the claim, not
    #: the gate stack -- so it is the cluster's load-bearing metric in place of
    #: the structurally-100% gate ASR.
    science_contract: dict[str, tuple[int, int]] = field(default_factory=dict)


async def run_siege_evaluation(
    *,
    instances_dir: str | Path | None = None,
    seed: int = 42,
    configs: tuple[AblationConfig, ...] = CUMULATIVE_CONFIGS,
    mode: str = "authored",
    agent_driver: Any | None = None,
    quarantine_agents: dict[str, Any] | None = None,
    judge: Any | None = None,
    slow_tier_judge: Any | None = None,
    max_instances: int | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
    bound: str = "declarative",
) -> SIEGEResult:
    """Run the full SIEGE ablation over the loaded instances.

    Args:
        instances_dir: directory of instance YAML/JSON files. None loads
            the committed active attack corpus.
        seed: recorded for report provenance.
        configs: the ablation matrix. Defaults to the 6 cumulative
            B4-first configs.
        mode: ``"authored"`` (default, offline replay) or ``"live"``
            (model-in-the-loop via ``LiveSessionRunner`` + ``agent_driver``).
        agent_driver: the live-mode ``AgentDriver`` (required when
            ``mode == "live"``).
        quarantine_agents: per-gate Q-LLM slow-tier agents (the live path
            wires these; works in authored mode too).
        judge: optional ``LlmJudge`` for indeterminate success criteria.
            This is a *measurement* instrument consumed by the scorer, not a
            defense; do not confuse it with ``slow_tier_judge``.
        slow_tier_judge: optional detection-only ``Detector`` wired as an
            extra slow-tier gate stage (``Gate._apply_judge_check``). A
            *defense*, consulted only by configs with ``judge_active``.
        max_instances: cost guard -- run only the first N instances
            (deterministic prefix of the id-sorted corpus).
        progress_cb: optional ``callable(done, total)`` invoked after each
            scored instance -- for a progress display. ``total`` is
            ``len(configs) * len(instances)``.

    Returns:
        A ``SIEGEResult``.
    """
    instances = load_instances(instances_dir if instances_dir is not None else CORPUS_DIR)
    if max_instances is not None:
        instances = instances[: max(0, int(max_instances))]

    live = mode == "live"
    if live and agent_driver is None:
        raise ValueError(
            "mode='live' requires an agent_driver (a LiveSessionRunner AgentDriver)"
        )
    score_kwargs = {"judge": judge} if judge is not None else {}

    scores: list[InstanceScore] = []
    total = len(configs) * len(instances)
    done = 0
    # Emit an initial tick so a progress display renders at 0/total right
    # away, instead of staying blank until the first (often slow,
    # Q-LLM-bound) cell completes.
    if progress_cb is not None:
        progress_cb(0, total)
    # Outer loop over configs, inner over instances, so cells group
    # cleanly per config in the cumulative order. Each instance gets a
    # fresh in-process turn buffer (no persistent cross-session store).
    _log = logging.getLogger(__name__)
    errors = 0
    for config in configs:
        for instance in instances:
            if live:
                from siege.live_session_runner import LiveSessionRunner

                runner = LiveSessionRunner(
                    agent_driver=agent_driver,
                    quarantine_agents=quarantine_agents,
                    memory_store=InProcessTurnBuffer(),
                )
            else:
                runner = SessionRunner(
                    memory_store=InProcessTurnBuffer(),
                    quarantine_agents=quarantine_agents,
                    judge=slow_tier_judge,
                    bound=bound,
                )
            try:
                trace = await runner.run(instance, config)
                scores.append(score_trace(trace, instance, **score_kwargs))
            except Exception:
                _log.warning(
                    "instance %s failed under %s; skipping",
                    instance.instance_id, config.name, exc_info=True,
                )
                errors += 1
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)
    if errors:
        _log.warning(
            "%d/%d instance evaluations failed and were skipped", errors, total
        )

    # Re-group so cells iterate (boundary, template) with all configs
    # adjacent -- the report wants one row block per (boundary, template).
    scores_by_cell_order = _reorder_for_report(scores, configs)
    cells = aggregate_cells(scores_by_cell_order)

    cov = _contract_coverage(instances)

    n_attack = sum(1 for i in instances if i.is_attack)
    return SIEGEResult(
        configs=tuple(configs),
        cells=tuple(cells),
        scores=tuple(scores),
        n_instances=len(instances),
        n_attack=n_attack,
        n_benign=len(instances) - n_attack,
        seed=seed,
        metric_definitions=dict(METRIC_DEFINITIONS),
        contract_coverage=cov[0],
        n_claims=cov[1],
        n_claims_covered=cov[2],
        n_claims_violated=cov[3],
        science_contract=cov[4],
    )


def _contract_coverage(
    instances,
) -> tuple[float | None, int, int, int, dict[str, tuple[int, int]]]:
    """Coverage + violation counts over the corpus's declared claims (WI11),
    plus the per-template science-cluster contract verdicts (Track E).

    Returns ``(coverage, n_claims, n_covered, n_violated, science_contract)``.
    Coverage is the fraction of declared scientific claims bounded by >=1
    active contract; ``None`` (with zero counts) when the contract library
    can't be loaded so the eval still runs without it. ``science_contract``
    maps each science-correctness template to ``(caught, total)`` over its
    attack instances -- ``caught`` = the contract layer flagged the poisoned
    claim (covered + violation) -- the cluster's real metric in place of the
    structurally-100% gate ASR.
    """
    try:
        from siege.oracles import (
            CorrectnessOracle,
            extract_instance_claims,
        )

        from siege.report_suites import ReportSuite, suite_for
    except Exception:  # pragma: no cover - oracle import optional
        return (None, 0, 0, 0, {})
    n_claims = sum(len(extract_instance_claims(i)) for i in instances)
    if n_claims == 0:
        return (None, 0, 0, 0, {})
    try:
        oracle = CorrectnessOracle.from_default()
    except Exception:  # pragma: no cover - contracts_dir optional
        return (None, n_claims, 0, 0, {})
    n_covered = n_violated = 0
    science: dict[str, list[int]] = {}
    for inst in instances:
        claims = extract_instance_claims(inst)
        if not claims:
            continue
        verdicts = [oracle.evaluate(c) for c in claims]
        n_covered += sum(1 for v in verdicts if v.covered)
        n_violated += sum(1 for v in verdicts if not v.ok)
        if inst.is_attack and suite_for(inst.template) is ReportSuite.SCIENCE_CORRECTNESS:
            caught = any(v.covered and not v.ok for v in verdicts)
            tally = science.setdefault(inst.template, [0, 0])
            tally[0] += int(caught)
            tally[1] += 1
    coverage = n_covered / n_claims
    science_contract = {t: (c, n) for t, (c, n) in science.items()}
    return (coverage, n_claims, n_covered, n_violated, science_contract)


def _reorder_for_report(
    scores: list[InstanceScore], configs: tuple[AblationConfig, ...]
) -> list[InstanceScore]:
    """Order scores so all configs of one (boundary, template) are adjacent.

    ``aggregate_cells`` preserves first-seen order; emitting
    (boundary, template) blocks with configs in cumulative order makes
    the report read as "watch ASR fall as gates are added."
    """
    config_rank = {c.name: i for i, c in enumerate(configs)}

    def key(s: InstanceScore) -> tuple[str, str, int]:
        return (s.boundary, s.template, config_rank.get(s.config_name, 0))

    return sorted(scores, key=key)


# -----------------------------------------------------------------
# Authored-vs-live parity (WI17)
# -----------------------------------------------------------------


@dataclass(frozen=True)
class ParityCell:
    template: str
    config_name: str
    authored_asr: float | None
    live_asr: float | None
    authored_bu: float | None
    live_bu: float | None

    @property
    def asr_delta(self) -> float | None:
        if self.authored_asr is None or self.live_asr is None:
            return None
        return self.live_asr - self.authored_asr


@dataclass(frozen=True)
class ParityReport:
    cells: tuple[ParityCell, ...]
    n_instances: int

    def divergent(self) -> list[ParityCell]:
        return [c for c in self.cells if c.asr_delta not in (None, 0.0)]

    def to_markdown(self) -> str:
        lines = [
            "# SIEGE authored-vs-live parity (WI17)",
            "",
            f"_{self.n_instances} instance(s); live = model-in-the-loop + Q-LLM slow tier._",
            "",
            "| template | config | authored ASR | live ASR | Δ ASR | authored BU | live BU |",
            "|---|---|---|---|---|---|---|",
        ]

        def pct(x: float | None) -> str:
            return "--" if x is None else f"{x:.0%}"

        for c in self.cells:
            d = c.asr_delta
            lines.append(
                f"| {c.template} | {c.config_name} | {pct(c.authored_asr)} | "
                f"{pct(c.live_asr)} | {('--' if d is None else f'{d:+.0%}')} | "
                f"{pct(c.authored_bu)} | {pct(c.live_bu)} |"
            )
        return "\n".join(lines)


async def run_parity(
    *,
    instances_dir: str | Path | None = None,
    configs: tuple[AblationConfig, ...] = CUMULATIVE_CONFIGS,
    agent_driver: Any,
    quarantine_agents: dict[str, Any] | None = None,
    judge: Any | None = None,
    max_instances: int | None = None,
) -> ParityReport:
    """Run the corpus in both modes; report per-cell ASR/BU divergence."""
    authored = await run_siege_evaluation(
        instances_dir=instances_dir, configs=configs, mode="authored",
        judge=judge, max_instances=max_instances,
    )
    live = await run_siege_evaluation(
        instances_dir=instances_dir, configs=configs, mode="live",
        agent_driver=agent_driver, quarantine_agents=quarantine_agents,
        judge=judge, max_instances=max_instances,
    )
    a_by = {(c.template, c.config_name): c for c in authored.cells}
    l_by = {(c.template, c.config_name): c for c in live.cells}
    cells = [
        ParityCell(
            template=t, config_name=cfg,
            authored_asr=a_by[(t, cfg)].asr, live_asr=l_by[(t, cfg)].asr,
            authored_bu=a_by[(t, cfg)].bu, live_bu=l_by[(t, cfg)].bu,
        )
        for (t, cfg) in a_by
        if (t, cfg) in l_by
    ]
    return ParityReport(cells=tuple(cells), n_instances=authored.n_instances)
