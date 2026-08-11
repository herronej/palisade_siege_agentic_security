"""
Spectroscopy contracts: bandgap / FWHM / peak-center, plus
cathodoluminescence (CL) and photoluminescence (PL) operating bounds.

As with the salt contracts, the numeric envelopes here are wide
plausibility bounds meant to catch gross errors (negative widths, peaks
outside the measurement window, excitation above emission), not to certify
spectroscopic accuracy.

Claim shapes::

    {"type": "bandgap", "value": 1.12} # eV
    {"type": "fwhm", "value": 0.08, "units": "eV"} # width > 0
    {"type": "peak_center", "value": 521.0, "window": [400, 800]} # nm
    {"type": "cl_voltage", "value": 5.0} # kV
    {"type": "pl_excitation","value": 405.0, "emission": 520.0} # nm
    {"type": "pl_emission", "value": 520.0} # nm
"""
from __future__ import annotations

from palisade.contracts.base import Contract, ContractResult, register

_BANDGAP_RANGE = (0.0, 12.0) # eV, insulators top out well under this
_CL_VOLTAGE_RANGE = (0.1, 30.0) # kV, typical SEM/CL accelerating voltage
_PL_WAVELENGTH_RANGE = (200.0, 2500.0) # nm, UV through NIR


@register
class BandgapContract(Contract):
    """A semiconductor/insulator bandgap must be non-negative and below an
    upper plausibility bound."""

    name = "bandgap_bounds"
    domain = "spectroscopy"
    applies_to_claims = ("bandgap", "band_gap", "eg")

    def check(self, claim: dict) -> ContractResult:
        value = claim.get("value")
        if value is None:
            return self.abstained_result(claim, note="no value to check")
        lo, hi = _BANDGAP_RANGE
        if value < lo:
            return self.violated(claim, f"bandgap {value} eV is negative")
        if value > hi:
            return self.violated(claim, f"bandgap {value} eV exceeds plausible {hi} eV")
        return self.passed(claim)


@register
class FWHMContract(Contract):
    """A peak full-width-at-half-maximum must be strictly positive."""

    name = "fwhm_positive"
    domain = "spectroscopy"
    applies_to_claims = ("fwhm", "linewidth", "full_width_half_max")

    def check(self, claim: dict) -> ContractResult:
        value = claim.get("value")
        if value is None:
            return self.abstained_result(claim, note="no value to check")
        if value <= 0:
            return self.violated(claim, f"FWHM must be positive, got {value}")
        return self.passed(claim)


@register
class PeakCenterContract(Contract):
    """A reported peak center must lie inside the measurement window when one
    is provided with the claim."""

    name = "peak_center_in_window"
    domain = "spectroscopy"
    applies_to_claims = ("peak_center", "peak", "peak_position")

    def check(self, claim: dict) -> ContractResult:
        value = claim.get("value")
        window = claim.get("window")
        if value is None or not window:
            return self.abstained_result(claim, note="no value/window to check")
        lo, hi = window[0], window[1]
        if not (lo <= value <= hi):
            return self.violated(
                claim, f"peak center {value} outside measurement window [{lo}, {hi}]"
            )
        return self.passed(claim)


@register
class CLBeamVoltageContract(Contract):
    """Cathodoluminescence beam accelerating voltage operating bound."""

    name = "cl_beam_voltage_bounds"
    domain = "spectroscopy"
    applies_to_claims = ("cl_voltage", "beam_voltage", "accelerating_voltage")

    def check(self, claim: dict) -> ContractResult:
        value = claim.get("value")
        if value is None:
            return self.abstained_result(claim, note="no value to check")
        lo, hi = _CL_VOLTAGE_RANGE
        if not (lo <= value <= hi):
            return self.violated(
                claim, f"CL beam voltage {value} kV outside operating [{lo}, {hi}]"
            )
        return self.passed(claim)


@register
class PLExcitationContract(Contract):
    """Photoluminescence excitation wavelength must be in range and, by Stokes
    shift, shorter than the emission wavelength when emission is provided."""

    name = "pl_excitation_bounds"
    domain = "spectroscopy"
    applies_to_claims = ("pl_excitation", "excitation_wavelength")

    def check(self, claim: dict) -> ContractResult:
        value = claim.get("value")
        if value is None:
            return self.abstained_result(claim, note="no value to check")
        lo, hi = _PL_WAVELENGTH_RANGE
        if not (lo <= value <= hi):
            return self.violated(
                claim, f"excitation {value} nm outside [{lo}, {hi}]"
            )
        emission = claim.get("emission")
        if emission is not None and value >= emission:
            return self.violated(
                claim,
                f"excitation {value} nm not shorter than emission {emission} nm "
                "(violates Stokes shift)",
            )
        return self.passed(claim)


@register
class PLEmissionContract(Contract):
    """Photoluminescence emission wavelength operating bound."""

    name = "pl_emission_bounds"
    domain = "spectroscopy"
    applies_to_claims = ("pl_emission", "emission_wavelength")

    def check(self, claim: dict) -> ContractResult:
        value = claim.get("value")
        if value is None:
            return self.abstained_result(claim, note="no value to check")
        lo, hi = _PL_WAVELENGTH_RANGE
        if not (lo <= value <= hi):
            return self.violated(claim, f"emission {value} nm outside [{lo}, {hi}]")
        return self.passed(claim)
