"""
Report suites -- reweighting the corpus by adversary realism.

The corpus is gate-organized, but the *headline* numbers should reflect
VISTA's actual threat surface, not the gate grid. Two reweightings:

1. **Misuse/safety split.** VISTA has a single trusted principal (no real
   multi-tenant auth), so a malicious *prompt* is the user attacking their
   own session -- a model-misuse / safety concern, not a security
   boundary. The direct user-prompt B1 classes are therefore split into a
   ``MISUSE_SAFETY`` sub-suite and kept **out of the headline security
   ASR**. ``b1_9_attached_content`` is excluded from the split: it was
   re-mapped onto the indirect upload -> tool-return path (G2/G4), so it
   is a genuine security class.

2. **Science-correctness headline.** The wrong-science cluster
   (data-value poisoning, citation forgery, correctness sabotage) is the
   mission risk for a scientific assistant. It is promoted to its own
   ``SCIENCE_CORRECTNESS`` headline whose load-bearing metric is
   **contract coverage** (the defense is the contract layer / G6, not a
   gate-ablation ASR).

Everything else attack is ``SECURITY`` (the headline adversarial ASR);
the §B false-positive set is ``BENIGN`` (BU). The classification is
template-keyed and operator-adjustable -- edit the sets below.
"""

from __future__ import annotations

from enum import Enum


class ReportSuite(str, Enum):
    """Where a template's cells are reported."""

    SECURITY = "security"
    MISUSE_SAFETY = "misuse_safety"
    SCIENCE_CORRECTNESS = "science_correctness"
    BENIGN = "benign"


#: Direct user-prompt B1 classes -- self-attacks under a single trusted
#: principal, so model-safety/misuse rather than a security boundary.
#: ``b1_9`` is deliberately NOT here (re-mapped to the indirect
#: upload->tool-return path, a real security class).
MISUSE_SAFETY_TEMPLATES: frozenset[str] = frozenset(
    {
        "b1_1_direct_injection",
        "b1_2_credentialing",
        "b1_3_multi_turn_crescendo",
        "b1_4_obfuscated_encoded",
        "b1_5_adversarial_suffix",
        "b1_7_goal_hijack",
        "b1_8_refusal_suppression",
        "b1_10_dual_use_framing",
    }
)

#: The wrong-science cluster -- mission risk, defended by the contract
#: layer (G6); reported as a contract-coverage headline.
SCIENCE_CORRECTNESS_TEMPLATES: frozenset[str] = frozenset(
    {
        "b3_3_data_value_poisoning",
        "b3_4_citation_forgery",
        "b4_5_correctness_sabotage",
    }
)

#: The §B benign false-positive set (feeds BU). ``benign_workload`` is the
#: original 24-task control; ``benign_diverse`` is the W1 expanded control (~157
#: tasks spanning imperative-but-benign prompts, instruction-quoting retrievals,
#: dual-use-adjacent science, and edge-of-policy HPC jobs).
BENIGN_TEMPLATES: frozenset[str] = frozenset(
    {"benign_workload", "benign_diverse"}
)


def suite_for(template: str) -> ReportSuite:
    """Classify a template into its report suite.

    Anything not explicitly split out is ``SECURITY`` -- the headline
    adversarial ASR set.
    """
    if template in MISUSE_SAFETY_TEMPLATES:
        return ReportSuite.MISUSE_SAFETY
    if template in SCIENCE_CORRECTNESS_TEMPLATES:
        return ReportSuite.SCIENCE_CORRECTNESS
    if template in BENIGN_TEMPLATES:
        return ReportSuite.BENIGN
    return ReportSuite.SECURITY


#: Report display order -- headline (security) first, then the promoted
#: science-correctness cluster, then the misuse/safety sub-suite, then the
#: benign FP set.
SUITE_ORDER: tuple[ReportSuite, ...] = (
    ReportSuite.SECURITY,
    ReportSuite.SCIENCE_CORRECTNESS,
    ReportSuite.MISUSE_SAFETY,
    ReportSuite.BENIGN,
)

#: Human-readable section labels for the report.
SUITE_LABELS: dict[ReportSuite, str] = {
    ReportSuite.SECURITY: "Security -- headline adversarial ASR",
    ReportSuite.SCIENCE_CORRECTNESS: (
        "Science-correctness -- mission risk (contract coverage)"
    ),
    ReportSuite.MISUSE_SAFETY: (
        "Misuse / safety -- single trusted principal; NOT in the security ASR"
    ),
    ReportSuite.BENIGN: "Benign false-positive set (BU)",
}


__all__ = [
    "ReportSuite",
    "MISUSE_SAFETY_TEMPLATES",
    "SCIENCE_CORRECTNESS_TEMPLATES",
    "BENIGN_TEMPLATES",
    "SUITE_ORDER",
    "SUITE_LABELS",
    "suite_for",
]
