"""
Slow-tier contract enforcement.

The shared machinery each gate's slow tier uses to run the contract
registry over the scientific claims in its context:

  * `extract_claims(text)` pulls structured claim envelopes out of free
    text (e.g. a retrieved RAG chunk that smuggles a fabricated value).
  * `evaluate_claims(registry, claims)` runs every applicable contract and
    reports the violations plus a coverage count.
  * `enforce(registry, claims, trust_scorer)` does the same and folds the
    coverage/violation counts into the session's trust scorer so the
    state API can surface domain-contract coverage.

A contract violation defaults to SEV2; a contract may escalate by putting
``incident_level`` in its `ContractResult.details` (the HPC path-scoping
contract uses SEV1). Gates translate the outcome into a `GateDecision`
(base `Gate.check_slow`) or an incident record (the G3 capability).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from palisade.contracts.base import (
    ContractRegistry,
    ContractResult,
    UNTRUSTED_KEY,
    claim_is_untrusted,
    run_contract,
)

# Re-exported from ``base`` for callers that import them here.
__all__ = [
    "UNTRUSTED_KEY",
    "claim_is_untrusted",
    "extract_claims",
    "evaluate_claims",
    "enforce",
    "ContractCheckOutcome",
]

# Default severity for a contract violation that does not declare its own.
_DEFAULT_VIOLATION_LEVEL = 2

# A poisoned/retrieved claim may be smuggled inline as
# ``[[CLAIM]]{...json...}[[/CLAIM]]``. This is the minimal structured
# extraction the slow tier understands; a richer NL claim extractor is a
# separate concern, but anything it produces flows through the same path.
_CLAIM_ENVELOPE = re.compile(r"\[\[CLAIM\]\](.*?)\[\[/CLAIM\]\]", re.DOTALL)


def extract_claims(text: str) -> list[dict]:
    """Pull ``[[CLAIM]]{json}[[/CLAIM]]`` envelopes out of `text`.

    Malformed envelopes are skipped (a poisoned chunk should never crash
    the slow tier). Returns the list of claim dicts, in order.
    """
    if not isinstance(text, str) or "[[CLAIM]]" not in text:
        return []
    claims: list[dict] = []
    for blob in _CLAIM_ENVELOPE.findall(text):
        try:
            obj = json.loads(blob.strip())
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict):
            claims.append(obj)
        elif isinstance(obj, list):
            claims.extend(c for c in obj if isinstance(c, dict))
    # These claims came from untrusted text; tag them so contracts refuse any
    # self-supplied trusted-context field (§ ``UNTRUSTED_KEY``).
    for claim in claims:
        claim[UNTRUSTED_KEY] = True
    return claims


def _violation_level(result: ContractResult) -> int:
    lvl = result.details.get("incident_level")
    # ``bool`` is an ``int`` subclass, so guard it: incident_level=True must not
    # silently read as SEV1.
    if isinstance(lvl, int) and not isinstance(lvl, bool):
        return int(lvl)
    return _DEFAULT_VIOLATION_LEVEL


@dataclass
class ContractCheckOutcome:
    """Result of running the registry over a batch of claims."""

    checked: int = 0
    covered: int = 0
    violations: list[ContractResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def incident_level(self) -> int | None:
        """The most severe (lowest-numbered) violation level, or None."""
        if not self.violations:
            return None
        return min(_violation_level(v) for v in self.violations)

    @property
    def coverage(self) -> float:
        return self.covered / self.checked if self.checked else 0.0

    @property
    def reason(self) -> str:
        return "; ".join(
            f"{v.contract}@{v.version}: {v.reason}" for v in self.violations
        )


def evaluate_claims(registry: ContractRegistry, claims) -> ContractCheckOutcome:
    """Run every applicable contract over each claim (no side effects).

    Each contract runs fail-closed (``run_contract``): a crash on a hostile
    claim becomes a violation, and never propagates out to take down the batch.
    ``covered`` counts *effective* coverage (a contract actually constrained the
    value), not mere applicability -- an abstaining contract does not count.
    """
    outcome = ContractCheckOutcome()
    for claim in claims or []:
        if not isinstance(claim, dict):
            continue
        outcome.checked += 1
        results = [run_contract(c, claim) for c in registry.applicable(claim)]
        if any(r.effective for r in results):
            outcome.covered += 1
        outcome.violations.extend(r for r in results if not r.ok)
    return outcome


def enforce(registry: ContractRegistry, claims, trust_scorer=None) -> ContractCheckOutcome:
    """`evaluate_claims`, then record the coverage/violation counts on the
    session trust scorer (when one is supplied) so the state API can
    surface domain-contract coverage."""
    outcome = evaluate_claims(registry, claims)
    if trust_scorer is not None and outcome.checked and hasattr(
        trust_scorer, "record_contract_check"
    ):
        trust_scorer.record_contract_check(
            outcome.checked, outcome.covered, len(outcome.violations)
        )
    return outcome
