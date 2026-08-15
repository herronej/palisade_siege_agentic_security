"""
Reference tables for the deployed contract library.

Loads the surrogate JSON tables under ``ground_truth_tables/`` and exposes
a small, dependency-free lookup API. The WI11 correctness contracts (in
the operator ``palisade_contracts/`` repo) call these to decide whether
a claimed value / citation / unit conversion is consistent with the
recorded ground truth.

This module imports **nothing** from ``palisade.contracts`` so the
external operator contracts can import it freely without an import cycle.

It lives in the sidecar rather than the benchmark because the *deployed*
contract library reads it: a host installing PALISADE alone still needs the
reference-value contract to resolve. SIEGE's correctness oracle reads it too,
via the same module.

> **Surrogate-stub.** The values here are plausibility-band central
> values, not certified MSTDB-TP data. A curated,
> domain-scientist-signed tolerance-bound table per claim type replaces
> them.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_TABLES_DIR = Path(__file__).resolve().parent / "ground_truth_tables"


@lru_cache(maxsize=None)
def _load(name: str) -> dict[str, Any]:
    path = _TABLES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


# -----------------------------------------------------------------
# Salt thermophysical values (MSTDB-TP surrogate)
# -----------------------------------------------------------------


def _norm_salt(salt: str) -> str:
    return str(salt).strip().replace("_", "-").upper()


def _norm_prop(prop: str) -> str:
    return str(prop).strip().lower().replace(" ", "_").replace("-", "_")


def mstdb_value(salt: str, prop: str) -> dict[str, Any] | None:
    """Surrogate ground-truth ``{value, unit, tol_pct}`` for ``(salt, prop)``.

    Returns None when the table has no entry (the contract then abstains).
    Salt/property matching is case- and separator-insensitive.
    """
    salts = _load("mstdb_tp_surrogate.json")["salts"]
    want_salt = _norm_salt(salt)
    want_prop = _norm_prop(prop)
    for s, props in salts.items():
        if _norm_salt(s) != want_salt:
            continue
        for p, rec in props.items():
            if _norm_prop(p) == want_prop:
                return dict(rec)
    return None


def default_tolerance_pct() -> float:
    return float(_load("mstdb_tp_surrogate.json")["_meta"]["default_tolerance_pct"])


# -----------------------------------------------------------------
# Physical constants, absolute bounds, unit conversions
# -----------------------------------------------------------------


def physical_constant(name: str) -> dict[str, Any] | None:
    """Look up a physical constant by canonical name or alias."""
    consts = _load("physical_constants.json")["constants"]
    key = str(name).strip().lower()
    for cname, rec in consts.items():
        aliases = {cname.lower(), *(a.lower() for a in rec.get("aliases", []))}
        if key in aliases:
            return dict(rec)
    return None


def absolute_bounds(prop: str) -> dict[str, Any] | None:
    """Gross-error plausibility envelope ``{lo, hi, unit}`` for a property."""
    bounds = _load("physical_constants.json")["absolute_bounds"]
    want = _norm_prop(prop)
    for p, rec in bounds.items():
        if _norm_prop(p) == want:
            return dict(rec)
    return None


def unit_conversions() -> dict[str, Any]:
    return dict(_load("physical_constants.json")["unit_conversions"])


def expected_temperature_trend(quantity: str) -> str | None:
    """e.g. viscosity -> 'decreases_with_T'."""
    trends = _load("physical_constants.json")["unit_conversions"]["temperature_trends"]
    return trends.get(_norm_prop(quantity))


# -----------------------------------------------------------------
# Source registry (citation integrity)
# -----------------------------------------------------------------


def _registry() -> dict[str, Any]:
    return _load("source_registry.json")


def source_status(identifier: str) -> str:
    """Classify a citation identifier against the surrogate registry.

    Returns one of ``"known"`` / ``"retracted"`` / ``"unknown"``. Matching
    is exact on the trimmed identifier across the DOI / arXiv / dataset /
    report allowlists.
    """
    ident = str(identifier).strip()
    reg = _registry()
    if ident in reg.get("retracted", []):
        return "retracted"
    for key in ("known_dois", "known_arxiv", "known_datasets", "known_reports"):
        if ident in reg.get(key, []):
            return "known"
    return "unknown"


def is_known_source(identifier: str) -> bool:
    return source_status(identifier) == "known"
