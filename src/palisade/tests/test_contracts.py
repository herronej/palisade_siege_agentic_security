"""
Unit tests for the contract DSL (base class + registration + loader).

Acceptance criteria pinned here:

  * a `Contract` base class with `name`, `domain`, `applies_to_claims`,
    and `check(claim) -> ContractResult`;
  * the `@register` decorator adds a contract to the global registry;
  * `load_contract_library(contracts_dir, project)` discovers builtin and
    operator-supplied contracts at startup;
  * versioning via the class-level `__contract_version__` attribute;
  * registration and lookup by domain / claim-type.
"""

from __future__ import annotations

import textwrap

from palisade.contracts import (
    Contract,
    ContractRegistry,
    ContractResult,
    claim_type,
    default_registry,
    load_contract_library,
    register,
)


# -----------------------------------------------------------------
# Test contracts
# -----------------------------------------------------------------


class _DensityContract(Contract):
    name = "test_density_positive"
    domain = "test_salts"
    applies_to_claims = ("density", "rho")
    __contract_version__ = "3"

    def check(self, claim: dict) -> ContractResult:
        if claim.get("value", 1) <= 0:
            return self.violated(claim, "density must be positive")
        return self.passed(claim)


class _BandgapContract(Contract):
    name = "test_bandgap_range"
    domain = "test_spectroscopy"
    applies_to_claims = ("bandgap",)

    def check(self, claim: dict) -> ContractResult:
        return self.passed(claim)


# -----------------------------------------------------------------
# Base class surface
# -----------------------------------------------------------------


def test_contract_base_surface() -> None:
    c = _DensityContract()
    assert c.name == "test_density_positive"
    assert c.domain == "test_salts"
    assert c.applies_to_claims == ("density", "rho")
    result = c.check({"type": "density", "value": 2100.0})
    assert isinstance(result, ContractResult)
    assert result.ok is True
    assert result.contract == "test_density_positive"
    assert result.domain == "test_salts"


def test_check_returns_violation_with_reason() -> None:
    c = _DensityContract()
    result = c.check({"type": "density", "value": -5.0})
    assert result.ok is False
    assert "positive" in result.reason
    assert result.contract == "test_density_positive"


def test_applies_matches_claim_type_aliases() -> None:
    c = _DensityContract()
    assert c.applies({"type": "density"}) is True
    assert c.applies({"claim_type": "rho"}) is True
    assert c.applies({"quantity": "density"}) is True
    assert c.applies({"type": "viscosity"}) is False
    assert c.applies({}) is False


def test_claim_type_helper() -> None:
    assert claim_type({"type": "density"}) == "density"
    assert claim_type({"claim_type": "rho"}) == "rho"
    assert claim_type({"quantity": "bandgap"}) == "bandgap"
    assert claim_type({}) is None


# -----------------------------------------------------------------
# Versioning
# -----------------------------------------------------------------


def test_contract_versioning() -> None:
    # Explicit version is reported on the instance and in results.
    c = _DensityContract()
    assert c.version == "3"
    assert _DensityContract.__contract_version__ == "3"
    assert c.check({"type": "density", "value": 1.0}).version == "3"

    # Unversioned contracts default to "0".
    assert _BandgapContract().version == "0"
    assert _BandgapContract().check({"type": "bandgap"}).version == "0"


# -----------------------------------------------------------------
# Registry: registration + lookup by domain / claim-type
# -----------------------------------------------------------------


def test_registry_registration_and_lookup() -> None:
    reg = ContractRegistry()
    density = reg.register(_DensityContract())
    bandgap = reg.register(_BandgapContract())
    assert len(reg) == 2

    # by domain
    assert reg.by_domain("test_salts") == [density]
    assert reg.by_domain("test_spectroscopy") == [bandgap]
    assert reg.by_domain("nonexistent") == []
    assert reg.domains() == {"test_salts", "test_spectroscopy"}

    # by claim type (including aliases declared in applies_to_claims)
    assert reg.by_claim_type("density") == [density]
    assert reg.by_claim_type("rho") == [density]
    assert reg.by_claim_type("bandgap") == [bandgap]
    assert reg.by_claim_type("unknown") == []

    # by name
    assert reg.get("test_density_positive") is density
    assert reg.get("missing") is None


def test_registration_is_idempotent_by_name() -> None:
    reg = ContractRegistry()
    reg.register(_DensityContract())
    reg.register(_DensityContract())
    assert len(reg) == 1


def test_registry_check_and_coverage() -> None:
    reg = ContractRegistry()
    reg.register(_DensityContract())
    claims = [
        {"type": "density", "value": -1.0},   # violates
        {"type": "density", "value": 2100.0},  # passes
        {"type": "viscosity", "value": 5.0},   # not covered
    ]
    assert reg.applicable(claims[0]) == reg.by_claim_type("density")
    violations = reg.violations(claims)
    assert len(violations) == 1 and violations[0].ok is False
    # 2 of 3 claims are bounded by an applicable contract.
    assert reg.coverage(claims) == 2 / 3
    assert reg.coverage([]) == 0.0


def test_register_decorator_adds_to_default_registry() -> None:
    @register
    class _DecoratedContract(Contract):
        name = "test_decorated_unique_contract"
        domain = "test_decorated"
        applies_to_claims = ("gizmo",)

        def check(self, claim: dict) -> ContractResult:
            return self.passed(claim)

    reg = default_registry()
    assert reg.get("test_decorated_unique_contract") is not None
    assert _DecoratedContract in (type(c) for c in reg.by_domain("test_decorated"))


# -----------------------------------------------------------------
# Loader: startup discovery of builtin + external contracts
# -----------------------------------------------------------------


def test_load_contract_library_loads_builtins() -> None:
    reg = load_contract_library()
    # The builtin library spans the salt and HPC domains. (Spectroscopy
    # contracts exist but are intentionally not loaded as builtins yet.)
    assert {"molten_salts", "hpc"} <= reg.domains()
    assert reg.by_claim_type("density"), "expected a builtin density contract"
    assert reg.by_claim_type("hpc_resources"), "expected a builtin HPC contract"
    assert len(reg) >= 3


def test_load_contract_library_discovers_external_contracts(tmp_path) -> None:
    contract_file = tmp_path / "ext_contract.py"
    contract_file.write_text(textwrap.dedent('''
        from palisade.contracts.base import Contract, register

        @register
        class _ExternalContract(Contract):
            name = "external_widget_contract"
            domain = "external_test_domain"
            applies_to_claims = ("widget",)
            __contract_version__ = "9"

            def check(self, claim):
                return self.passed(claim)
    '''))

    reg = load_contract_library(str(tmp_path))
    contract = reg.get("external_widget_contract")
    assert contract is not None
    assert contract.domain == "external_test_domain"
    assert contract.version == "9"
    assert reg.by_claim_type("widget") == [contract]


def test_load_contract_library_accepts_settings_like_object() -> None:
    class _FakeSettings:
        contracts_dir = None

    reg = load_contract_library(_FakeSettings())
    assert {"molten_salts", "hpc"} <= reg.domains()


def test_load_contract_library_missing_dir_is_safe(tmp_path) -> None:
    reg = load_contract_library(str(tmp_path / "does_not_exist"))
    assert len(reg) >= 3
