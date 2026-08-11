"""
Coverage + realism annotation tests (corpus revisions 5 & 6).

Pins that every B4 class (and the xc_4 B4-sink) is tagged with the
merged-G4 tier that defends it -- so the Semgrep-install dependency is
explicit -- and that the three down-scoped/conditional classes carry a
realism caveat, and that the report surfaces both.
"""

from __future__ import annotations

from siege.corpus_annotations import (
    DEFENSE_TIER,
    REALISM_NOTE,
    defense_tier,
    realism_note,
)
from siege.report_suites import ReportSuite
from siege.scorer import Cell
from siege.templates import TEMPLATE_BUILDERS


def test_defense_tier_covers_every_b4_class_and_xc4():
    real = set(TEMPLATE_BUILDERS)
    b4 = {t for t in real if t.startswith("b4_")}
    missing = b4 - set(DEFENSE_TIER)
    assert not missing, f"untagged B4 classes: {missing}"
    assert "xc_4_taint_laundering" in DEFENSE_TIER
    assert set(DEFENSE_TIER) <= real, "DEFENSE_TIER names unknown templates"


def test_tier1_semgrep_classes_flag_the_dependency():
    # The in-process-Python classes must name the Semgrep tier so the
    # install dependency (degrade-to-Tier-0) is explicit.
    for t in ("b4_1_malicious_code", "b4_2_network_exfil", "b4_3_typo_squat"):
        assert "Semgrep" in DEFENSE_TIER[t]
    # The always-on deterministic win:
    assert "Tier-0" in DEFENSE_TIER["b4_4_filesystem_write_abuse"]
    # b4_5 is contract-defended, not a G4 tier:
    assert "Contract" in DEFENSE_TIER["b4_5_correctness_sabotage"]
    # xc_4 laundering evades every tier (the point):
    assert "None" in DEFENSE_TIER["xc_4_taint_laundering"]


def test_realism_notes_downscope_intended_classes():
    assert set(REALISM_NOTE) == {
        "b4_7_sensitive_file_read",
        "b4_8_resource_exhaustion",
        "b5_9_social_engineering_confirmation",
    }
    assert set(REALISM_NOTE) <= set(TEMPLATE_BUILDERS)
    assert "empty target" in REALISM_NOTE["b4_7_sensitive_file_read"].lower()
    assert "undetected" in REALISM_NOTE["b4_8_resource_exhaustion"].lower()
    assert "approval" in REALISM_NOTE["b5_9_social_engineering_confirmation"].lower()


def test_helpers_return_none_for_unannotated():
    assert defense_tier("benign_workload") is None
    assert realism_note("b4_1_malicious_code") is None
    assert realism_note("benign_workload") is None


def test_report_surfaces_tier_and_realism():
    from siege.eval.siege_report import _cell_tables
    from siege.eval.siege_runner import SIEGEResult

    cells = (
        Cell(
            boundary="B4.7",
            template="b4_7_sensitive_file_read",
            config_name="baseline",
            n_attack=1,
            n_benign=0,
            asr=1.0,
            ua=None,
            bu=None,
            hard_win_rate=1.0,
            suite=ReportSuite.SECURITY.value,
        ),
    )
    result = SIEGEResult(
        configs=(),
        cells=cells,
        scores=(),
        n_instances=1,
        n_attack=1,
        n_benign=0,
        seed=42,
        metric_definitions={},
    )
    md = _cell_tables(result)
    assert "Defended by:" in md
    assert "Realism:" in md
    assert "empty target" in md.lower()


def test_cross_tenant_is_corpus_sourced():
    from siege.corpus_annotations import (
        CROSS_TENANT_TEMPLATES,
        is_cross_tenant,
        tenancy_note,
    )

    assert CROSS_TENANT_TEMPLATES <= set(TEMPLATE_BUILDERS)
    # Corpus-sourced classes are cross-tenant (shared global KB)...
    assert is_cross_tenant("b3_1_corpus_poisoning")
    assert is_cross_tenant("xc_4_taint_laundering")  # RAG-sourced supply
    # ...but per-(project,user) classes are single-tenant contained.
    assert not is_cross_tenant("b1_9_attached_content")  # per-user upload
    assert not is_cross_tenant("xc_5_history_forgery")  # own session history
    assert not is_cross_tenant("b4_8_resource_exhaustion")  # own MicroVM
    assert tenancy_note("b3_1_corpus_poisoning")
    assert tenancy_note("b1_9_attached_content") is None


def test_report_preconditions_and_tenancy_surface():
    from siege.eval.siege_report import (
        _cell_tables,
        _preconditions_block,
        format_siege_report,
    )
    from siege.eval.siege_runner import SIEGEResult

    cells = (
        Cell(
            boundary="B3.1",
            template="b3_1_corpus_poisoning",
            config_name="baseline",
            n_attack=1,
            n_benign=0,
            asr=1.0,
            ua=None,
            bu=None,
            hard_win_rate=0.0,
            suite=ReportSuite.SECURITY.value,
        ),
    )
    result = SIEGEResult(
        configs=(),
        cells=cells,
        scores=(),
        n_instances=1,
        n_attack=1,
        n_benign=0,
        seed=42,
        metric_definitions={},
    )
    pre = _preconditions_block(result)
    assert "Auth precondition" in pre
    assert "Multi-tenant profile" in pre
    assert "b3_1_corpus_poisoning" in pre  # listed as cross-tenant
    assert "Tenancy:" in _cell_tables(result)
    assert "Threat-model preconditions" in format_siege_report(result)
