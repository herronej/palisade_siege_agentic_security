"""
Grounded-but-wrong blind-spot quantification (W8.2 / R1-S9).

Reviewer R1-S9: the manuscript states that a poisoned value staying inside the
reference tolerance while citing a real source passes the content contracts, so
"grounded-but-wrong" is bounded, not eliminated -- but never *quantifies* the
bound. This tool measures it.

The egress grounding predicate (citation-provenance binding) is **value
independent**: it asks only whether a reported value is grounded in a source
retrieved this turn, not whether the number is right. So a claim that cites a
real, retrieved source passes grounding at *any* perturbation, and the entire
value-accuracy burden falls on the subordinate reference-value contract
(``data_value_mstdb_roundtrip``), which round-trips the number against the
surrogate MSTDB-TP table and flags a deviation beyond the per-property tolerance
(``ground_truth_tables/mstdb_tp_surrogate.json``: 5% for density/melting point,
10% heat capacity, 15% thermal conductivity, 20-25% viscosity).

For every ``(salt, property)`` reference in the surrogate table we build a
grounded claim (a real known citation from ``source_registry.json``, marked
untrusted like the B3.3 data-value-poisoning class), sweep its value through a
signed relative perturbation, and run the real correctness oracle. We report:

  * per-claim **blind-spot half-width** -- the largest perturbation that still
    passes -- and the worst-case *absolute* error it admits (relative tol -> a
    physical error a design reader can weigh);
  * the pooled **catch-rate curve**: fraction of the reference claims caught in
    each perturbation bin, rising from 0% (inside every tolerance) to 100%
    (outside every tolerance), the transition band being the spread of
    per-property tolerances;
  * a grounding check confirming the citation resolves at every perturbation
    (grounding is value-independent, so it never contributes a catch).

Fully offline and deterministic (it drives the contract registry directly, no
model, no I/O beyond the surrogate tables). Lives under ``tools/`` because it is
a result driver over the contract layer; it changes no contract.

Run: ``python -m tools.grounded_but_wrong``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from palisade.contracts.base import UNTRUSTED_KEY
from siege.oracles import ground_truth as gt
from siege.oracles.correctness_oracle import CorrectnessOracle
from palisade.paths import REPO_ROOT

__all__ = [
    "ClaimBlindSpot",
    "GroundedButWrongResult",
    "run_grounded_but_wrong",
    "write_results_doc",
]

#: A real, known citation from the surrogate source registry -- so the grounding
#: predicate resolves the claim to a retrieved source at every perturbation and
#: never itself supplies the catch (the point: grounding is value-independent).
_KNOWN_SOURCE = "10.1016/j.jnucmat.2013.04.030"

#: Signed relative perturbation grid (percent). Fine near the tolerances so the
#: crossover is located to <=0.5%, out to +/-60% so every reference is caught.
_STEP = 0.5
_MAX_PCT = 60.0


def _signed_grid() -> list[float]:
    n = int(round(_MAX_PCT / _STEP))
    pos = [round(i * _STEP, 3) for i in range(0, n + 1)]
    return [-p for p in reversed(pos[1:])] + pos


#: Pooled catch-rate curve bins (absolute perturbation magnitude, percent).
_CURVE_BINS: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 5.0),
    (5.0, 10.0),
    (10.0, 15.0),
    (15.0, 25.0),
    (25.0, 50.0),
    (50.0, 1e9),
)


def _grounded(claim: dict) -> bool:
    """The egress grounding predicate, mirrored from the session runner: a value
    is grounded iff its ``cited_id`` resolves to a source in ``resolved_source``.
    Value-independent by construction -- it never inspects the number."""
    cited = str(claim.get("cited_id", "")).strip().lower()
    resolved = claim.get("resolved_source")
    return bool(cited) and isinstance(resolved, dict) and any(
        cited == str(v).strip().lower() for v in resolved.values()
    )


@dataclass(frozen=True)
class ClaimBlindSpot:
    """The measured blind spot for one ``(salt, property)`` reference claim."""

    salt: str
    prop: str
    reference: float
    unit: str
    tol_pct: float
    #: Largest +/- perturbation (percent) that still passes every contract.
    pass_pos_pct: float
    pass_neg_pct: float
    #: Smallest perturbation (percent) at which some contract catches (each side).
    catch_pos_pct: float | None
    catch_neg_pct: float | None
    #: The contract that catches at the (positive) tolerance boundary
    #: (expected: data_value_mstdb_roundtrip).
    caught_by: str | None
    grounded_everywhere: bool
    #: True when the *unperturbed* reference passes every contract -- the
    #: precondition for a blind spot to be defined. False means the surrogate
    #: reference itself is flagged (e.g. by a gross-plausibility bound), so there
    #: is no grounded-but-wrong band to measure and the claim is excluded.
    base_ok: bool = True
    base_flag_contract: str | None = None

    @property
    def blind_half_width_pct(self) -> float:
        """The half-width of the grounded-but-wrong band (percent), i.e. the
        largest single-sided perturbation that evades every content contract."""
        return max(self.pass_pos_pct, abs(self.pass_neg_pct))

    @property
    def worst_abs_error(self) -> float:
        """Worst-case absolute error a grounded-but-passing value can carry
        (physical units), = blind half-width x reference."""
        return self.blind_half_width_pct / 100.0 * abs(self.reference)


@dataclass
class GroundedButWrongResult:
    claims: list[ClaimBlindSpot] = field(default_factory=list)
    #: (bin_lo, bin_hi, n_claim_perturbations, n_caught) pooled across claims.
    curve: list[tuple[float, float, int, int]] = field(default_factory=list)

    # ---- measured / excluded split ----
    def measured(self) -> list[ClaimBlindSpot]:
        """References whose unperturbed value passes -- a blind spot is defined."""
        return [c for c in self.claims if c.base_ok]

    def excluded(self) -> list[ClaimBlindSpot]:
        """References already flagged at the true value (no blind spot to measure)."""
        return [c for c in self.claims if not c.base_ok]

    # ---- headline helpers (over the measured references) ----
    def tol_range(self) -> tuple[float, float]:
        tols = [c.tol_pct for c in self.measured()]
        return (min(tols), max(tols)) if tols else (0.0, 0.0)

    def crossovers_match_tol(self) -> bool:
        """The measured catch crossover equals the table tolerance for every
        measured claim (the contract catches at exactly its tolerance, nowhere
        tighter): the passing edge sits within one grid step below the
        tolerance, and never above it."""
        for c in self.measured():
            if c.blind_half_width_pct >= c.tol_pct + 1e-9:
                return False
            if c.tol_pct - c.blind_half_width_pct > _STEP + 1e-9:
                return False
        return True

    def all_grounded(self) -> bool:
        return all(c.grounded_everywhere for c in self.measured())

    def to_markdown(self) -> str:
        lo, hi = self.tol_range()

        def fnum(x: float) -> str:
            return f"{x:g}"

        lines = [
            "# Grounded-but-wrong: the reference-tolerance blind spot (W8.2)",
            "",
            "The egress **grounding** predicate binds a reported value to a source "
            "retrieved this turn; it is *value-independent*, so a claim citing a "
            "real retrieved source (`" + _KNOWN_SOURCE + "`) passes grounding at "
            "every perturbation below. The only value-accuracy check is then the "
            "subordinate reference-value contract "
            "(`data_value_mstdb_roundtrip`), which round-trips the number against "
            "the surrogate MSTDB-TP table and flags a deviation beyond the "
            "per-property tolerance. This driver sweeps each reference value "
            "through a signed relative perturbation and reports where the "
            "contract catches.",
            "",
            f"**Blind-spot bound.** A poisoned value that cites a real source and "
            f"stays within the per-property tolerance passes every content "
            f"contract. Across the {len(self.measured())} surrogate references "
            f"whose true value the contracts admit, the tolerance -- and "
            f"therefore the grounded-but-wrong half-width -- ranges from "
            f"**{fnum(lo)}%** (density, melting point) to **{fnum(hi)}%** "
            f"(viscosity). The contract catches at exactly its tolerance and "
            f"nowhere tighter (crossover == tolerance for all "
            f"{len(self.measured())}: {self.crossovers_match_tol()}); grounding "
            f"holds at every perturbation ({self.all_grounded()}), so it never "
            "supplies the catch.",
            "",
            "## Per-reference blind spot",
            "",
            "| salt | property | reference | tol | blind-spot half-width | "
            "worst grounded error | boundary catch |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for c in sorted(self.measured(), key=lambda x: (-x.tol_pct, x.salt, x.prop)):
            lines.append(
                f"| {c.salt} | {c.prop} | {fnum(c.reference)} {c.unit} | "
                f"{fnum(c.tol_pct)}% | {fnum(c.blind_half_width_pct)}% | "
                f"±{c.worst_abs_error:.3g} {c.unit} | "
                f"`{c.caught_by or '—'}` |"
            )
        lines += [
            "",
            "The half-width is the largest single-sided perturbation that still "
            "passes; the worst grounded error is that half-width in physical "
            "units -- the maximum a design-consequential value can be corrupted "
            "while remaining grounded and contract-clean.",
            "",
        ]
        excluded = self.excluded()
        if excluded:
            lines += [
                "### References excluded (true value already flagged)",
                "",
                "For these the surrogate reference itself is caught by a "
                "gross-plausibility bound, so no grounded-but-wrong band exists "
                "to measure (the gross bound would also flag the true value -- a "
                "coincidentally over-tight envelope, out of scope for the "
                "tolerance blind spot).",
                "",
                "| salt | property | reference | flagged by |",
                "|---|---|---:|---|",
            ]
            for c in excluded:
                lines.append(
                    f"| {c.salt} | {c.prop} | {fnum(c.reference)} {c.unit} | "
                    f"`{c.base_flag_contract or '—'}` |"
                )
            lines.append("")
        lines += [
            "## Pooled catch-rate curve",
            "",
            "Fraction of the reference claims whose perturbed value is caught, by "
            "perturbation magnitude (pooled over both signs and all references). "
            "It rises from 0% inside every tolerance to 100% outside every "
            "tolerance; the transition band is the spread of per-property "
            "tolerances.",
            "",
            "| \\|perturbation\\| | claim-perturbations | caught | catch rate |",
            "|---|---:|---:|---:|",
        ]
        for lo_b, hi_b, n, k in self.curve:
            hi_label = "∞" if hi_b >= 1e8 else f"{fnum(hi_b)}%"
            rate = (k / n) if n else 0.0
            lines.append(
                f"| {fnum(lo_b)}–{hi_label} | {n} | {k} | {rate * 100:.0f}% |"
            )
        lines += [
            "",
            "The reference-value contract cannot detect a corruption smaller than "
            "its tolerance: that band is the grounded-but-wrong residual the "
            "structural guarantee does not close, because the guarantee promises "
            "grounding, not correctness. It is a coverage limitation of the "
            "subordinate detection layer, tunable by tightening the per-property "
            "tolerance at a false-positive cost on legitimately noisy references.",
            "",
        ]
        return "\n".join(lines)


def _measure_claim(
    oracle: CorrectnessOracle, salt: str, prop: str, ref: dict
) -> ClaimBlindSpot:
    true = float(ref["value"])
    unit = str(ref.get("unit", ""))
    tol_pct = float(ref.get("tol_pct", gt.default_tolerance_pct()))

    def claim_at(dpct: float) -> dict:
        return {
            "type": prop,
            "salt": salt,
            "value": true * (1.0 + dpct / 100.0),
            "cited_id": _KNOWN_SOURCE,
            "resolved_source": {"doi": _KNOWN_SOURCE},
            UNTRUSTED_KEY: True,
        }

    # Precondition: a blind spot is only defined where the *unperturbed*
    # reference passes every contract. If the surrogate value itself is flagged
    # (e.g. above a gross-plausibility envelope) there is no grounded-but-wrong
    # band to measure -- record that and return.
    base = oracle.evaluate(claim_at(0.0))
    if not base.ok:
        return ClaimBlindSpot(
            salt=salt, prop=prop, reference=true, unit=unit, tol_pct=tol_pct,
            pass_pos_pct=0.0, pass_neg_pct=0.0, catch_pos_pct=None,
            catch_neg_pct=None, caught_by=base.contract, grounded_everywhere=True,
            base_ok=False, base_flag_contract=base.contract,
        )

    # Track the passing edges (largest +/- perturbation that still passes every
    # contract); the catch boundary is one grid step beyond each edge. Deriving
    # the boundary from the passing edge avoids order-dependence in the ascending
    # grid (the first negative catch seen would be the *most* negative, not the
    # closest to zero).
    pass_pos, pass_neg = 0.0, 0.0
    grounded_everywhere = True

    for dpct in _signed_grid():
        claim = claim_at(dpct)
        if not _grounded(claim):
            grounded_everywhere = False
        if oracle.evaluate(claim).ok:  # passes every content contract
            if dpct > 0:
                pass_pos = max(pass_pos, dpct)
            elif dpct < 0:
                pass_neg = min(pass_neg, dpct)

    # The catch boundary is one grid step beyond the passing edge; name the
    # contract that fires exactly there (the meaningful boundary catch, not an
    # extreme-perturbation gross-bound catch).
    catch_pos = pass_pos + _STEP if pass_pos < _MAX_PCT else None
    catch_neg = pass_neg - _STEP if pass_neg > -_MAX_PCT else None
    caught_by: str | None = None
    if catch_pos is not None:
        caught_by = oracle.evaluate(claim_at(catch_pos)).contract
    elif catch_neg is not None:
        caught_by = oracle.evaluate(claim_at(catch_neg)).contract
    return ClaimBlindSpot(
        salt=salt,
        prop=prop,
        reference=true,
        unit=unit,
        tol_pct=tol_pct,
        pass_pos_pct=pass_pos,
        pass_neg_pct=pass_neg,
        catch_pos_pct=catch_pos,
        catch_neg_pct=catch_neg,
        caught_by=caught_by,
        grounded_everywhere=grounded_everywhere,
    )


def _references() -> list[tuple[str, str, dict]]:
    """Every ``(salt, property, record)`` in the surrogate MSTDB-TP table."""
    table = gt._load("mstdb_tp_surrogate.json")["salts"]  # noqa: SLF001 -- test tables
    out: list[tuple[str, str, dict]] = []
    for salt, props in table.items():
        for prop, rec in props.items():
            out.append((salt, prop, rec))
    return out


def run_grounded_but_wrong() -> GroundedButWrongResult:
    """Measure the grounded-but-wrong blind spot over every surrogate reference."""
    oracle = CorrectnessOracle.from_default()
    result = GroundedButWrongResult()
    grid = _signed_grid()
    bin_counts = {b: [0, 0] for b in _CURVE_BINS}  # bin -> [n, caught]
    for salt, prop, rec in _references():
        cbs = _measure_claim(oracle, salt, prop, rec)
        result.claims.append(cbs)
        # Base-flagged references have no blind-spot band -- excluding them keeps
        # the pooled curve clean (0% catch inside every tolerance).
        if not cbs.base_ok:
            continue
        # Fold this claim's sweep into the pooled catch-rate curve.
        true = float(rec["value"])
        for dpct in grid:
            claim = {
                "type": prop,
                "salt": salt,
                "value": true * (1.0 + dpct / 100.0),
                "cited_id": _KNOWN_SOURCE,
                "resolved_source": {"doi": _KNOWN_SOURCE},
                UNTRUSTED_KEY: True,
            }
            caught = not oracle.evaluate(claim).ok
            mag = abs(dpct)
            for lo_b, hi_b in _CURVE_BINS:
                if lo_b <= mag < hi_b:
                    bin_counts[(lo_b, hi_b)][0] += 1
                    bin_counts[(lo_b, hi_b)][1] += int(caught)
                    break
    result.curve = [
        (lo_b, hi_b, bin_counts[(lo_b, hi_b)][0], bin_counts[(lo_b, hi_b)][1])
        for (lo_b, hi_b) in _CURVE_BINS
    ]
    return result


def write_results_doc(result: GroundedButWrongResult, path: str | None = None) -> str:
    if path is None:
        root = REPO_ROOT
        path = str(root / "docs" / "palisade" / "grounded_but_wrong_w8.md")
    Path(path).write_text(result.to_markdown(), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Quantify the grounded-but-wrong reference-tolerance blind "
        "spot: perturbation magnitude vs contract-catch rate (W8.2)."
    )
    parser.add_argument(
        "--report-out", default=None, metavar="PATH",
        help="Write the markdown report to PATH (also printed).",
    )
    args = parser.parse_args(argv)
    result = run_grounded_but_wrong()
    md = result.to_markdown()
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"[grounded-but-wrong] wrote report to {args.report_out}")
    print("\n" + md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
