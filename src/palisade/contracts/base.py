"""
PALISADE contract DSL.

A *contract* is a small, hand-authored, neuro-symbolic check that bounds a
scientific claim the agent emits -- a density, a viscosity, a bandgap, a
SLURM resource request. Contracts are the "hard constraint" layer: they do
not depend on the LLM, they are deterministic, and they live as plain
Python so domain scientists can review and extend them via PRs.

A *claim* is a plain dict the runtime extracts from agent output. It always
carries a claim type and the quantities the contract needs, e.g.::

    {"type": "density", "family": "flinak", "A": 2552.3, "B": 0.512,
     "T": 873.0, "value": 2104.6}

Contracts declare which claim types they apply to via `applies_to_claims`.
The registry runs every applicable contract over a claim and returns the
results; a `coverage` metric reports the fraction of claims bounded by at
least one active contract (proposal §7.3).

Authoring a contract::

    from palisade.contracts.base import Contract, ContractResult, register

    @register
    class MyContract(Contract):
        name = "my_check"
        domain = "molten_salts"
        applies_to_claims = ("density", "rho")

        def check(self, claim: dict) -> ContractResult:
            if claim["value"] < 0:
                return self.violated(claim, "density must be positive")
            return self.passed(claim)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable

logger = logging.getLogger(__name__)


def claim_type(claim: dict) -> str | None:
    """Extract the claim's type from any of the accepted keys."""
    for key in ("type", "claim_type", "quantity"):
        val = claim.get(key)
        if val:
            return str(val)
    return None


#: Marker key set on every claim pulled from untrusted text (``extract_claims``).
#: Contracts that would otherwise trust a self-supplied "trusted-context" field
#: (a heat-capacity ``reference`` override, a ``resolved_source`` provenance)
#: must refuse that field when this marker is present -- a poisoned chunk may
#: assert those fields, and only a trusted gate may populate them.
UNTRUSTED_KEY = "_untrusted"


def claim_is_untrusted(claim: dict) -> bool:
    """True if ``claim`` was extracted from untrusted text (not gate-built)."""
    return bool(claim.get(UNTRUSTED_KEY))


@dataclass
class ContractResult:
    """Outcome of running one contract against one claim.

    ``abstained`` distinguishes an *effective* pass (the contract ran its real
    check and the value satisfied it) from an *abstention* (the contract
    applied by type but could not actually constrain the value -- a missing
    value, an unrecognized family/salt/unit, no envelope). Abstentions are
    ``ok=True`` but do **not** count as coverage: ``covered`` means "a value was
    actually constrained," not merely "a contract's type matched" -- so a claim
    an attacker mislabels past every real check is reported as uncovered, not
    vacuously covered.
    """

    ok: bool
    contract: str
    domain: str = ""
    version: str = ""
    reason: str = ""
    claim: dict | None = None
    details: dict = field(default_factory=dict)
    abstained: bool = False

    @property
    def effective(self) -> bool:
        """True when the contract actually constrained the value (a real check)."""
        return not self.abstained


class Contract(ABC):
    """Base class for all contracts.

    Subclasses set `name`, `domain`, and `applies_to_claims`, and implement
    `check`. The `passed` / `violated` helpers build a `ContractResult`
    pre-filled with the contract's identity.

    Versioning: a scientist bumps the class-level `__contract_version__`
    whenever the bounds or logic change, so a recorded `ContractResult`
    (and any provenance derived from it) is traceable to the exact
    contract revision that fired. Unversioned contracts report ``"0"``.
    """

    name: str = ""
    domain: str = ""
    applies_to_claims: tuple[str,...] = ()
    #: Bumped by the author on every substantive change to the contract.
    __contract_version__: ClassVar[str] = "0"

    @property
    def version(self) -> str:
        """The contract's declared version (``__contract_version__``)."""
        return self.__contract_version__

    def applies(self, claim: dict) -> bool:
        ct = claim_type(claim)
        return ct is not None and ct in self.applies_to_claims

    @abstractmethod
    def check(self, claim: dict) -> ContractResult:
        """Return a ContractResult. Only called when `applies(claim)` is True."""

    # -- result helpers --------------------------------------------------
    def passed(self, claim: dict | None = None, **details: Any) -> ContractResult:
        return ContractResult(
            ok=True,
            contract=self.name,
            domain=self.domain,
            version=self.__contract_version__,
            claim=claim,
            details=details,
        )

    def violated(
        self, claim: dict | None, reason: str, **details: Any
    ) -> ContractResult:
        return ContractResult(
            ok=False,
            contract=self.name,
            domain=self.domain,
            version=self.__contract_version__,
            reason=reason,
            claim=claim,
            details=details,
        )

    def abstained_result(self, claim: dict | None = None, *, note: str, **details: Any) -> ContractResult:
        """A pass the contract could not actually check (missing / unrecognized
        input). ``ok=True`` but not counted as coverage -- see ``ContractResult``.
        """
        return ContractResult(
            ok=True,
            contract=self.name,
            domain=self.domain,
            version=self.__contract_version__,
            reason=note,
            claim=claim,
            details={"note": note, **details},
            abstained=True,
        )


def run_contract(contract: Contract, claim: dict) -> ContractResult:
    """Run one contract **fail-closed**.

    A contract may crash on a hostile claim -- a non-numeric ``value`` (``float``
    raises), a malformed regex in a self-scanning claim (``re.error``), a
    catastrophic-backtracking pattern (ReDoS). For a security layer a crash must
    not (a) take down the batch or (b) fail open, so it is caught and reported as
    a violation. This is the single choke point every caller runs contracts
    through.
    """
    try:
        return contract.check(claim)
    except Exception as exc:  # noqa: BLE001 -- deliberate: fail closed on any crash
        logger.warning("contract %r crashed on claim: %s", contract.name, exc)
        return ContractResult(
            ok=False,
            contract=contract.name,
            domain=contract.domain,
            version=contract.version,
            reason=f"contract raised {type(exc).__name__}: {exc}",
            claim=claim,
            details={"crashed": True},
        )


class ContractRegistry:
    """Holds the active contracts and runs them over claims.

    Registration is idempotent by contract name so re-importing the builtin
    modules (e.g. when several sidecars load the library) does not create
    duplicates.
    """

    def __init__(self) -> None:
        self._contracts: dict[str, Contract] = {}

    def register(self, contract: Contract) -> Contract:
        prev = self._contracts.get(contract.name)
        if prev is not None and type(prev) is not type(contract):
            # Given the external-loading path (operator ``*.py`` loaded after the
            # builtins), a name collision silently shadowing a builtin is an
            # integrity concern, not a benign re-import -- surface it.
            logger.warning(
                "contract name %r shadowed: %s overrides %s",
                contract.name, type(contract).__qualname__, type(prev).__qualname__,
            )
        self._contracts[contract.name] = contract
        return contract

    def all(self) -> list[Contract]:
        return list(self._contracts.values())

    def __len__(self) -> int:
        return len(self._contracts)

    def get(self, name: str) -> Contract | None:
        """Look up a single contract by its unique `name`."""
        return self._contracts.get(name)

    def by_domain(self, domain: str) -> list[Contract]:
        """All contracts in a domain (e.g. ``"molten_salts"``)."""
        return [c for c in self._contracts.values() if c.domain == domain]

    def by_claim_type(self, claim_type: str) -> list[Contract]:
        """All contracts that declare they apply to a claim type."""
        return [
            c for c in self._contracts.values() if claim_type in c.applies_to_claims
        ]

    def domains(self) -> set[str]:
        """The set of domains represented in the registry."""
        return {c.domain for c in self._contracts.values()}

    def applicable(self, claim: dict) -> list[Contract]:
        return [c for c in self._contracts.values() if c.applies(claim)]

    def check_claim(self, claim: dict) -> list[ContractResult]:
        return [run_contract(c, claim) for c in self.applicable(claim)]

    def check_claims(self, claims: Iterable[dict]) -> list[ContractResult]:
        results: list[ContractResult] = []
        for claim in claims:
            results.extend(self.check_claim(claim))
        return results

    def violations(self, claims: Iterable[dict]) -> list[ContractResult]:
        return [r for r in self.check_claims(claims) if not r.ok]

    def coverage(self, claims: Iterable[dict]) -> float:
        """Fraction of claims **effectively** bounded by >=1 contract.

        Effective, not merely applicable: a claim counts as covered only when a
        contract actually ran its check on it (a violation or an effective
        pass), not when a contract's type matched but it abstained (unknown
        family/salt/unit, missing value). An attacker who mislabels a claim past
        every real check is therefore reported as uncovered, not vacuously
        covered.
        """
        claims = list(claims)
        if not claims:
            return 0.0
        covered = 0
        for claim in claims:
            results = self.check_claim(claim)
            if any(r.effective for r in results):
                covered += 1
        return covered / len(claims)


# Module-level default registry populated by the @register decorator when the
# builtin contract modules are imported.
_DEFAULT_REGISTRY = ContractRegistry()


def default_registry() -> ContractRegistry:
    return _DEFAULT_REGISTRY


def register(cls: type[Contract]) -> type[Contract]:
    """Class decorator: instantiate the contract and add it to the default registry."""
    _DEFAULT_REGISTRY.register(cls())
    return cls
