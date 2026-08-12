"""
Ground-truth correctness oracle (WI11, PoC stub).

The oracle is the thing that, given a scientific *claim* a (possibly
compromised) agent emits, returns ``OK`` or ``VIOLATION(reason)``. It is a
thin wrapper over the PALISADE contract registry: it loads the builtin
contracts plus the operator ``palisade_contracts/`` library (the WI11
physical-bounds / data-value / citation-integrity / unit-consistency /
correlation-form contracts), runs every applicable contract over a claim,
and reports the verdict.

> **PoC-stub status.** Each contract returns ``OK | VIOLATION(reason)``
> against the surrogate ground-truth tables in ``ground_truth_tables/``.
> A curated, domain-scientist-signed tolerance-bound table per claim
> type will replace these; until then the verdict is advisory.

Claims reach the oracle two ways:

- **Directly** — a ``dict`` (``{"type": "density", "salt": "FLiBe", ...}``).
- **From a SIEGE instance** — ``extract_instance_claims(instance)``
  pulls every ``payload["claim"]`` an authored attack action declares (the
  data-value / citation-forgery / correctness-sabotage classes attach the
  scientific claim their attack corrupts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palisade.contracts import load_contract_library
from palisade.contracts.base import UNTRUSTED_KEY, ContractRegistry, run_contract
from siege.paths import REPO_ROOT

# Repo-root ``palisade_contracts/`` resolved from this file so the oracle
# works regardless of the process cwd (tests run from backend/).
_REPO_ROOT = REPO_ROOT
_DEFAULT_CONTRACTS_DIR = _REPO_ROOT / "palisade_contracts"

# Cache the loaded registry per contracts_dir: ``load_contract_library`` re-globs
# and re-execs the external ``*.py`` files on every call, so callers that build
# many oracles (the WI16 generator / bound table) would otherwise pay that
# repeatedly. Registry state is a shared singleton, so caching is safe.
_REGISTRY_CACHE: dict[str, ContractRegistry] = {}


@dataclass(frozen=True)
class OracleVerdict:
    """The oracle's decision on one claim.

    ``ok`` is True when no applicable contract was violated (including the
    abstain case where no contract applied). ``covered`` is True when at
    least one contract applied — the signal the coverage metric reads.
    """

    ok: bool
    covered: bool
    reason: str = ""
    contract: str | None = None
    version: str = ""
    claim: dict | None = None


class CorrectnessOracle:
    """Runs the contract registry over scientific claims."""

    def __init__(self, registry: ContractRegistry) -> None:
        self._registry = registry

    @classmethod
    def from_default(cls, contracts_dir: str | Path | None = None) -> "CorrectnessOracle":
        """Load builtins + the operator contract library into the oracle (cached)."""
        cdir = str(contracts_dir if contracts_dir is not None else _DEFAULT_CONTRACTS_DIR)
        registry = _REGISTRY_CACHE.get(cdir)
        if registry is None:
            registry = load_contract_library(contracts_dir=cdir)
            _REGISTRY_CACHE[cdir] = registry
        return cls(registry)

    @property
    def registry(self) -> ContractRegistry:
        return self._registry

    def evaluate(self, claim: dict) -> OracleVerdict:
        """Run every applicable contract over ``claim`` and fold the results.

        ``covered`` is *effective* coverage -- at least one contract actually
        constrained the value (ran a real check), not merely applied by type. An
        attacker who mislabels a claim past every real check is uncovered, not
        vacuously covered. Each contract runs fail-closed (a crash is a violation).
        """
        results = [run_contract(c, claim) for c in self._registry.applicable(claim)]
        covered = any(r.effective for r in results)
        violations = [r for r in results if not r.ok]
        if not violations:
            return OracleVerdict(ok=True, covered=covered, claim=claim)
        first = violations[0]
        reason = "; ".join(f"{v.contract}@{v.version}: {v.reason}" for v in violations)
        return OracleVerdict(
            ok=False,
            covered=covered,
            reason=reason,
            contract=first.contract,
            version=first.version,
            claim=claim,
        )

    def coverage(self, claims: list[dict]) -> float:
        """Fraction of claims bounded by ≥1 applicable contract."""
        return self._registry.coverage(claims)


# -----------------------------------------------------------------
# Claim extraction from SIEGE instances
# -----------------------------------------------------------------


def extract_instance_claims(instance: Any) -> list[dict]:
    """Pull every ``payload["claim"]`` declared on an instance's actions.

    The WI10 data-value / citation-forgery / correctness-sabotage templates
    attach the scientific claim their attack corrupts under the attack
    action's ``payload["claim"]`` (the free-form payload dict; gates ignore
    the extra key). Returns the claims in execution order.

    A claim on an ``is_attack`` action is attacker-controlled content, so it is
    tagged untrusted -- contracts then refuse any self-supplied trusted-context
    field on it (a citation-forgery ``resolved_source``, a heat-capacity
    ``reference`` override), matching how ``extract_claims`` tags claims pulled
    from retrieved text.
    """
    claims: list[dict] = []
    for session in instance.sessions:
        for turn in session.turns:
            for action in turn.actions:
                claim = action.payload.get("claim")
                if isinstance(claim, dict) and claim:
                    c = dict(claim)
                    if getattr(action, "is_attack", False):
                        c[UNTRUSTED_KEY] = True
                    claims.append(c)
    return claims
