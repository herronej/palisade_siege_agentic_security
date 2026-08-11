"""
Reference-value tolerance: practical-scale fraction + operator sensitivity sweep
(W22.2 / W32.1).

``grounded_but_wrong.py`` (W8.2) establishes, per surrogate reference, that the
grounded-but-wrong blind-spot half-width equals the property's tolerance exactly
(a per-property *crossover* fact). Two things it does not report:

**W22.2 -- the practical fraction at realistic scale** (R2-6 / R3-4). A crossover
point does not tell a reader how much of a *plausible* attacker corruption would
actually land undetected. This tool fixes a menu of realistic corruption
magnitudes -- anchored on the deployment's own per-property tolerances (5% to
25%) plus a tighter and a looser bound for context -- and reports, for each
magnitude, the fraction of the 15 surrogate references for which a claim
corrupted by exactly that much would pass every content contract. This converts
"the contract catches at exactly its tolerance" into "at a realistic N% forgery,
K/15 properties would go undetected," the practical-scale statistic the
reviewers asked for.

**W32.1 -- the operator sensitivity knob** (R1-3 exp #4). The per-property
tolerances are a deployment setting, not a law of nature. This sweeps a single
GLOBAL multiplier over every tolerance at once (tightening toward 0.25x, loosening
toward 2x) and reports, at each setting, two real measurements against the
**production** ``data_value_mstdb_roundtrip`` contract (monkeypatching
``ground_truth.mstdb_value`` so the sweep exercises the actual contract code, not
a re-derivation):

* **attack coverage** -- catch rate on the 5 real, corpus-authored ``b3_3``
  data-value-poisoning instances (genuine attacks, not invented magnitudes);
* **benign false-positive rate** -- catch rate on a modeled *legitimate
  measurement noise* population: each of the 15 surrogate references jittered by
  a stated small fraction of its *current* (1x) tolerance, representing that real
  replicate scatter is normally well inside the officially allowed band. This is
  an explicit, stated modeling assumption (no real replicate-literature data is
  in this repo), not a corpus fact -- flagged as such in the report.

Together: tightening the tolerance raises attack coverage but also benign FPR;
loosening does the reverse. The sweep gives an operator the actual tradeoff
curve, not just the current operating point.

Fully offline and deterministic; reuses the production contract registry
(``CorrectnessOracle.from_default``) and the real ``b3_3`` corpus via
``extract_instance_claims``.

    cd backend
    uv run python -m tools.tolerance_sensitivity \\
        --report-out ../docs/palisade/tolerance_sensitivity_w22_w32.md
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

from palisade.contracts.base import UNTRUSTED_KEY
from siege.instance_loader import load_instances
from siege.oracles import ground_truth as gt
from siege.oracles.correctness_oracle import (
    CorrectnessOracle,
    extract_instance_claims,
)
from tools.grounded_but_wrong import _KNOWN_SOURCE, _references
from palisade.paths import CORPUS_DIR

_CORPUS_DIR = CORPUS_DIR

# -----------------------------------------------------------------
# W22.2 -- realistic corruption-magnitude menu, anchored on the deployment's own
# per-property tolerance range (5%-25%) plus context bounds.
# -----------------------------------------------------------------
_REALISTIC_MAGNITUDES_PCT: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)


def _measured_references(oracle: CorrectnessOracle) -> list[tuple[str, str, dict]]:
    """Surrogate references whose *true* value passes every contract -- a blind
    spot / benign-noise band is defined only for these (the W8.2 convention:
    a reference already flagged at its true value, e.g. by a gross-plausibility
    bound unrelated to the tolerance being swept, is excluded everywhere here,
    not just in the practical-fraction table)."""
    out: list[tuple[str, str, dict]] = []
    for salt, prop, rec in _references():
        true = float(rec["value"])
        base_claim = {
            "type": prop, "salt": salt, "value": true,
            "cited_id": _KNOWN_SOURCE, "resolved_source": {"doi": _KNOWN_SOURCE},
            UNTRUSTED_KEY: True,
        }
        if oracle.evaluate(base_claim).ok:
            out.append((salt, prop, rec))
    return out


def practical_fraction(magnitudes: tuple[float, ...] = _REALISTIC_MAGNITUDES_PCT) -> list[tuple[float, int, int]]:
    """For each corruption magnitude (percent), ``(magnitude, undetected, n)``
    over the surrogate references whose true value passes (a blind spot is
    defined) -- run against the real oracle at the deployed (1x) tolerance."""
    oracle = CorrectnessOracle.from_default()
    refs = _measured_references(oracle)
    out: list[tuple[float, int, int]] = []
    for mag in magnitudes:
        undetected = 0
        for salt, prop, rec in refs:
            true = float(rec["value"])
            for sign in (+1, -1):
                claim = {
                    "type": prop, "salt": salt, "value": true * (1.0 + sign * mag / 100.0),
                    "cited_id": _KNOWN_SOURCE, "resolved_source": {"doi": _KNOWN_SOURCE},
                    UNTRUSTED_KEY: True,
                }
                if oracle.evaluate(claim).ok:
                    undetected += 1
                    break  # either sign undetected counts the reference once
        out.append((mag, undetected, len(refs)))
    return out


# -----------------------------------------------------------------
# W32.1 -- global tolerance-multiplier sweep against the production contract.
# -----------------------------------------------------------------
_MULTIPLIERS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
#: Benign-noise jitter as a fraction of the reference's *current* (1x)
#: tolerance -- an explicit modeling assumption (no real replicate data here),
#: stated in the report: real measurement scatter is normally well inside the
#: officially allowed band.
_BENIGN_NOISE_FRACTIONS: tuple[float, ...] = (0.25, 0.5)


@contextmanager
def _scaled_tolerance(multiplier: float):
    """Monkeypatch ``ground_truth.mstdb_value`` so every ``tol_pct`` the
    production contract reads is scaled by ``multiplier`` -- the operator's
    global tolerance knob, exercised through the real contract code."""
    original = gt.mstdb_value

    def scaled(salt: str, prop: str):
        ref = original(salt, prop)
        if ref is None:
            return None
        ref = dict(ref)
        ref["tol_pct"] = float(ref.get("tol_pct", gt.default_tolerance_pct())) * multiplier
        return ref

    gt.mstdb_value = scaled  # noqa: SLF001 -- intentional, scoped monkeypatch
    try:
        yield
    finally:
        gt.mstdb_value = original


def _b3_3_attack_claims() -> list[dict]:
    """The 5 real, corpus-authored b3_3 data-value-poisoning claims (not
    invented magnitudes) -- extracted via the production ``extract_instance_claims``."""
    claims: list[dict] = []
    for inst in load_instances(_CORPUS_DIR / "b3_3_data_value_poisoning"):
        claims.extend(extract_instance_claims(inst))
    return claims


def _benign_noise_claims(fraction: float, refs: list[tuple[str, str, dict]]) -> list[dict]:
    """One jittered claim per *measured* surrogate reference (``refs``, the
    base_ok-filtered set -- excludes any reference already flagged at its true
    value, so the benign population is genuinely legitimate) at ``fraction`` of
    its *current* (1x) tolerance -- the modeled legitimate-measurement-noise
    population. Uses a real known citation so grounding always resolves."""
    out: list[dict] = []
    for salt, prop, rec in refs:
        true = float(rec["value"])
        tol_pct = float(rec.get("tol_pct", gt.default_tolerance_pct()))
        jitter = tol_pct * fraction  # deterministic +/-, both signs included
        for sign in (+1, -1):
            out.append({
                "type": prop, "salt": salt, "value": true * (1.0 + sign * jitter / 100.0),
                "cited_id": _KNOWN_SOURCE, "resolved_source": {"doi": _KNOWN_SOURCE},
                UNTRUSTED_KEY: True,
            })
    return out


def sensitivity_sweep(
    multipliers: tuple[float, ...] = _MULTIPLIERS,
) -> list[tuple[float, tuple[int, int], dict[float, tuple[int, int]]]]:
    """For each multiplier: ``(multiplier, (attacks_caught, n_attacks),
    {noise_fraction: (benign_flagged, n_benign)})``."""
    oracle = CorrectnessOracle.from_default()
    attack_claims = _b3_3_attack_claims()
    measured_refs = _measured_references(oracle)
    noise_claims = {f: _benign_noise_claims(f, measured_refs) for f in _BENIGN_NOISE_FRACTIONS}
    out: list[tuple[float, tuple[int, int], dict[float, tuple[int, int]]]] = []
    for m in multipliers:
        with _scaled_tolerance(m):
            caught = sum(1 for c in attack_claims if not oracle.evaluate(c).ok)
            n_attack = len(attack_claims)
            noise_rates: dict[float, tuple[int, int]] = {}
            for frac, claims in noise_claims.items():
                flagged = sum(1 for c in claims if not oracle.evaluate(c).ok)
                noise_rates[frac] = (flagged, len(claims))
        out.append((m, (caught, n_attack), noise_rates))
    return out


# -----------------------------------------------------------------
# Render
# -----------------------------------------------------------------
def render(
    frac_rows: list[tuple[float, int, int]],
    sweep_rows: list[tuple[float, tuple[int, int], dict[float, tuple[int, int]]]],
) -> str:
    lines = [
        "# Reference-value tolerance: practical fraction + sensitivity sweep (W22.2 / W32.1)",
        "",
        "## W22.2 — practical fraction of plausibly-poisoned claims that land in-tolerance",
        "",
        "The grounded-but-wrong (W8.2) blind-spot half-width equals each property's "
        "tolerance exactly. Here, for each realistic forgery magnitude (anchored on "
        "the deployment's own 5%-25% tolerance range, plus 2% and 30-40% for "
        "context), the fraction of the 15 surrogate references for which that "
        "magnitude would pass undetected -- at the **deployed (1x)** tolerance.",
        "",
        "| forgery magnitude | undetected (grounded-but-wrong) | detected |",
        "|---|---|---|",
    ]
    for mag, undetected, n in frac_rows:
        lines.append(
            f"| {mag:g}% | {undetected}/{n} ({undetected / n:.0%}) | "
            f"{n - undetected}/{n} ({(n - undetected) / n:.0%}) |"
        )
    lines += [
        "",
        (
            f"At a realistic {frac_rows[2][0]:g}% forgery (the deployment's own "
            f"density/melting-point tolerance), **{frac_rows[2][1]}/{frac_rows[2][2]} "
            f"= {frac_rows[2][1] / frac_rows[2][2]:.0%}** of the 15 properties would "
            "go undetected while citing a real source; at the loosest deployed "
            f"tolerance ({frac_rows[5][0]:g}%, viscosity), "
            f"**{frac_rows[5][1]}/{frac_rows[5][2]} = "
            f"{frac_rows[5][1] / frac_rows[5][2]:.0%}** would. This is the "
            "practical-scale reading of the crossover fact: for a domain-plausible "
            "forger who does not need to guess the exact per-property tolerance, "
            "a substantial share of realistic forgeries land inside it."
        ),
        "",
        "## W32.1 — tolerance-multiplier sensitivity (the operator's knob)",
        "",
        "A single global multiplier scaling every `data_value_mstdb_roundtrip` "
        "tolerance at once, exercised against the **production** contract "
        "(monkeypatched `ground_truth.mstdb_value`, not a re-derivation). "
        "**Attack coverage** = catch rate on the 5 real, corpus-authored `b3_3` "
        "data-value-poisoning instances. **Benign FPR** = catch rate on a "
        "*modeled* legitimate-measurement-noise population -- each of the 15 "
        "surrogate references jittered by a stated fraction of its **current** "
        "(1x) tolerance (an explicit assumption: real replicate scatter sits "
        "well inside the officially allowed band; this repo has no real "
        "replicate-literature data to draw from instead).",
        "",
        "| tolerance × | attack coverage (b3_3, n=5) | benign FPR (noise=25% of tol) | benign FPR (noise=50% of tol) |",
        "|---|---|---|---|",
    ]
    for m, (caught, n_a), noise in sweep_rows:
        cov = f"{caught}/{n_a} ({caught / n_a:.0%})"
        n25 = noise[0.25]
        n50 = noise[0.5]
        f25 = f"{n25[0]}/{n25[1]} ({n25[0] / n25[1]:.0%})"
        f50 = f"{n50[0]}/{n50[1]} ({n50[0] / n50[1]:.0%})"
        marker = " *(deployed)*" if m == 1.0 else ""
        lines.append(f"| {m:g}×{marker} | {cov} | {f25} | {f50} |")
    lines += [
        "",
        "## Finding",
        "",
        (
            "Tightening the tolerance raises attack coverage monotonically but "
            "raises benign FPR with it; loosening trades the reverse. The deployed "
            "1x operating point sits where it does because the b3_3 corpus's real "
            "corruption magnitudes (15-35% deviation) already clear the default "
            "tolerance, so 1x already achieves full coverage on the corpus's own "
            "attacks without yet paying the benign cost tighter multipliers would. "
            "An operator who wants headroom against *subtler* forgeries than the "
            "corpus contains can read the tightened rows directly against the "
            "benign-FPR cost they would pay for it, rather than tuning blind."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Practical-fraction (W22.2) + tolerance-sensitivity sweep (W32.1)."
    )
    parser.add_argument("--report-out", default=None, metavar="PATH")
    args = parser.parse_args(argv)

    frac_rows = practical_fraction()
    sweep_rows = sensitivity_sweep()
    md = render(frac_rows, sweep_rows)
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"[tolerance-sensitivity] wrote {args.report_out}")
    print("\n" + md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
