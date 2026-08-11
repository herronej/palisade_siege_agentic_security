"""
Unit tests for the initial molten-salt property contract set.

Covers in-bounds and out-of-bounds claims for the three contracts:

  * density rho(T) = A - B*T with per-family MSTDB-TP envelopes;
  * Andrade-style log-linear viscosity eta(T) = A * exp(B/T);
  * heat capacity within a factor of 2 of the database-reported value.

Coefficient envelopes are exercised for FLiNaK, FLiBe, and a chloride
family (NaCl-KCl), per the acceptance criteria.
"""

from __future__ import annotations

import math

from palisade.contracts import default_registry, load_contract_library
from palisade.contracts.salts import (
    REVIEW,
    SaltDensityContract,
    SaltHeatCapacityContract,
    SaltViscosityContract,
)


# -----------------------------------------------------------------
# Density: rho(T) = A - B*T
# -----------------------------------------------------------------


def _density(**kw) -> dict:
    return {"type": "density", **kw}


def test_density_in_bounds_flinak() -> None:
    c = SaltDensityContract()
    A, B, T = 2579.3, 0.624, 873.0
    result = c.check(_density(family="flinak", A=A, B=B, T=T, value=A - B * T))
    assert result.ok is True
    assert result.version == "1"


def test_density_in_bounds_flibe_and_chloride() -> None:
    c = SaltDensityContract()
    for family, A, B in (("flibe", 2413.0, 0.488), ("nacl-kcl", 2021.8, 0.5426)):
        T = 1000.0
        result = c.check(_density(family=family, A=A, B=B, T=T, value=A - B * T))
        assert result.ok is True, f"{family} should be in bounds"


def test_density_coefficient_A_out_of_bounds() -> None:
    c = SaltDensityContract()
    result = c.check(_density(family="flinak", A=3500.0, B=0.624, T=873.0))
    assert result.ok is False
    assert result.details.get("coefficient") == "A"
    assert "95% CI" in result.reason


def test_density_coefficient_B_out_of_bounds() -> None:
    c = SaltDensityContract()
    result = c.check(_density(family="flibe", A=2413.0, B=1.5, T=873.0))
    assert result.ok is False
    assert result.details.get("coefficient") == "B"


def test_density_value_inconsistent_with_correlation() -> None:
    c = SaltDensityContract()
    # In-bounds coefficients and an in-absolute-band value, but far from A - B*T
    # (so the round-trip check, not the gross band, is what catches it).
    result = c.check(_density(family="flinak", A=2579.3, B=0.624, T=873.0, value=3000.0))
    assert result.ok is False
    assert "inconsistent" in result.reason


def test_density_negative_value_violates() -> None:
    c = SaltDensityContract()
    result = c.check(_density(family="flinak", value=-1.0))
    assert result.ok is False
    assert "positive" in result.reason


def test_density_unknown_family_passes_with_note() -> None:
    c = SaltDensityContract()
    # A plausible (in gross-band) value for an unknown family: the absolute band
    # constrains it, but the family-specific envelope is unavailable.
    result = c.check(_density(family="unobtanium-salt", A=1.0, B=1.0, T=900.0, value=2000.0))
    assert result.ok is True
    assert "no MSTDB-TP envelope" in result.details.get("note", "")


def test_density_gross_value_caught_without_family_envelope() -> None:
    """finding 3: an order-of-magnitude / unit error is caught by the absolute
    band even when the family is unknown (no more free pass on a mislabel)."""
    c = SaltDensityContract()
    result = c.check(_density(family="unobtanium-salt", value=3.0))
    assert result.ok is False
    assert "outside plausible" in result.reason


# -----------------------------------------------------------------
# Viscosity: Andrade eta(T) = A * exp(B/T)
# -----------------------------------------------------------------


def _visc(**kw) -> dict:
    return {"type": "viscosity", **kw}


def test_viscosity_in_bounds_andrade_flinak() -> None:
    c = SaltViscosityContract()
    A, B, T = 0.040, 4170.0, 873.0
    predicted = A * math.exp(B / T)
    result = c.check(_visc(family="flinak", A=A, B=B, T=T, value=predicted))
    assert result.ok is True
    assert result.version == "1"


def test_viscosity_in_bounds_flibe() -> None:
    c = SaltViscosityContract()
    A, B, T = 0.116, 3760.0, 873.0
    predicted = A * math.exp(B / T)
    result = c.check(_visc(family="flibe", A=A, B=B, T=T, value=predicted))
    assert result.ok is True


def test_viscosity_coefficient_out_of_bounds() -> None:
    c = SaltViscosityContract()
    result = c.check(_visc(family="flinak", A=0.04, B=9000.0, T=873.0, value=5.0))
    assert result.ok is False
    assert result.details.get("coefficient") == "B"


def test_viscosity_value_inconsistent_with_andrade() -> None:
    c = SaltViscosityContract()
    # In-bounds coefficients (predicted ~4.7 mPa*s) but a value far off,
    # still inside the absolute plausibility band.
    result = c.check(_visc(family="flinak", A=0.040, B=4170.0, T=873.0, value=50.0))
    assert result.ok is False
    assert "inconsistent" in result.reason


def test_viscosity_negative_and_absurd_values_violate() -> None:
    c = SaltViscosityContract()
    assert c.check(_visc(family="flinak", value=-2.0)).ok is False
    huge = c.check(_visc(family="flinak", value=10_000.0))
    assert huge.ok is False and "outside plausible" in huge.reason


def test_viscosity_without_coefficients_uses_absolute_band() -> None:
    c = SaltViscosityContract()
    # No A/B: only positivity + absolute band are checked.
    assert c.check(_visc(family="flinak", value=5.0)).ok is True


# -----------------------------------------------------------------
# Heat capacity: within a factor of 2 of database value
# -----------------------------------------------------------------


def _cp(**kw) -> dict:
    return {"type": "heat_capacity", **kw}


def test_heat_capacity_in_bounds_per_family() -> None:
    c = SaltHeatCapacityContract()
    assert c.check(_cp(family="flinak", value=1880.0)).ok is True
    assert c.check(_cp(family="flibe", value=2386.0)).ok is True
    assert c.check(_cp(family="nacl-kcl", value=1080.0)).ok is True


def test_heat_capacity_just_inside_2x_band() -> None:
    c = SaltHeatCapacityContract()
    # FLiNaK reference 1880 -> band [940, 3760].
    assert c.check(_cp(family="flinak", value=3700.0)).ok is True
    assert c.check(_cp(family="flinak", value=950.0)).ok is True


def test_heat_capacity_above_2x_violates() -> None:
    c = SaltHeatCapacityContract()
    result = c.check(_cp(family="flinak", value=4000.0))
    assert result.ok is False
    assert "within 2x" in result.reason
    assert result.details.get("reference") == 1880.0


def test_heat_capacity_below_half_violates() -> None:
    c = SaltHeatCapacityContract()
    result = c.check(_cp(family="flibe", value=900.0))  # ref 2386 -> band [1193, 4772]
    assert result.ok is False


def test_heat_capacity_explicit_reference_overrides_table() -> None:
    c = SaltHeatCapacityContract()
    # Unknown family but an explicit reference -> 2x band around it.
    assert c.check(_cp(family="mystery", value=2000.0, reference=1100.0)).ok is True
    assert c.check(_cp(family="mystery", value=3000.0, reference=1100.0)).ok is False


def test_heat_capacity_negative_violates() -> None:
    c = SaltHeatCapacityContract()
    assert c.check(_cp(family="flinak", value=-5.0)).ok is False


# -----------------------------------------------------------------
# Registration / metadata
# -----------------------------------------------------------------


def test_salt_contracts_registered_in_default_registry() -> None:
    reg = load_contract_library()
    by_domain = {c.name for c in reg.by_domain("molten_salts")}
    assert {
        "salt_density_temperature_dependence",
        "salt_viscosity_andrade",
        "salt_heat_capacity_within_2x",
    } <= by_domain
    # Lookup by claim type resolves each contract.
    assert reg.by_claim_type("density")
    assert reg.by_claim_type("viscosity")
    assert reg.by_claim_type("heat_capacity")


def test_salt_contracts_cover_required_families() -> None:
    from palisade.contracts.salts import _DENSITY_BOUNDS, _VISCOSITY_BOUNDS

    for table in (_DENSITY_BOUNDS, _VISCOSITY_BOUNDS):
        assert "flinak" in table
        assert "flibe" in table
        assert any("cl" in fam for fam in table), "expected a chloride family"


def test_review_marker_present() -> None:
    # The in-repo sign-off record exists so review status is tracked.
    assert set(REVIEW) == {"status", "reviewed_by", "mstdb_tp_version"}
