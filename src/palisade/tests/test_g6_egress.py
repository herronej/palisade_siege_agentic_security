"""
G6 Egress Gate tests.

Two layers:

- The gate's pure logic (identifier extraction, retrieval-ledger parsing,
  claim building, annotation rendering) -- deterministic, no I/O.
- The capability wired end-to-end through a real sidecar: a `rag_search`
  result feeds the per-turn retrieval ledger, then a final output with an
  ungrounded citation is annotated (annotate enforcement) while a grounded
  one passes through untouched.
"""

from __future__ import annotations

import asyncio

import pytest
from pathlib import Path
from types import SimpleNamespace

from palisade.capabilities.g6_egress import G6EgressCapability
from palisade.config import PalisadeSettings
from palisade.gates.g6_egress import G6EgressGate
from palisade.sidecar import PalisadeSidecar
from palisade.tests.test_sidecar import _make_project
from palisade.paths import require_repo_root

REPO_ROOT = require_repo_root()

# Absolute path to the operator contract library (cwd-independent), so the
# sidecar's registry includes ``provenance_binding``.
_CONTRACTS = str(REPO_ROOT / "palisade_contracts")

# A real retrieved passage as the RAG MCP server formats it (citation line
# carries the trusted DOI).
_RAG_RESULT = (
    "[1] flinak_viscosity.pdf, page 3\n"
    "Viscosity of molten FLiNaK. Smith et al. J. Nucl. Mater. (2013). "
    "DOI: 10.1016/j.jnucmat.2013.04.030\n\n"
    "The viscosity of FLiNaK at 873 K is ..."
)
_GROUNDED_DOI = "10.1016/j.jnucmat.2013.04.030"
_FABRICATED_DOI = "10.1016/j.msrtp.2019.00471"

# A NaF-UF4 thermal-conductivity passage actually retrieved (the true source).
# The value 1.2 lives in THIS paper, so citing a different paper for it is
# misattribution, not mere fabrication.
_RAG_RESULT_NAFUF4 = (
    "[1] williams_coolants.pdf, page 12\n"
    "Assessment of Candidate Molten Salt Coolants. Williams et al. (2006). "
    "DOI: 10.1016/abc.2006.001\n\n"
    "The thermal conductivity of NaF-UF4 is 1.2 W/m-K near 973 K."
)
_TRUE_DOI = "10.1016/abc.2006.001"
_WRONG_DOI = "10.9999/forged.2099.000"


# -----------------------------------------------------------------
# Gate pure logic
# -----------------------------------------------------------------


def test_cited_identifiers_extracts_doi_and_arxiv():
    g = G6EgressGate(enabled=True)
    ids = g.cited_identifiers(
        f"density 1940 (doi:{_FABRICATED_DOI}) per arXiv:2105.99213."
    )
    norm = {i.lower().replace("arxiv:", "") for i in ids}
    assert _FABRICATED_DOI in norm
    assert "2105.99213" in norm


def test_parse_retrieved_sources_lifts_doi_text_and_numbers():
    g = G6EgressGate(enabled=True)
    sources = g.parse_retrieved_sources(_RAG_RESULT)
    assert any(s.get("doi") == _GROUNDED_DOI for s in sources)
    # chunk text + its numeric values are captured for content grounding
    assert any("873" in s.get("numbers", frozenset()) for s in sources)


def test_build_citation_claims_grounds_and_flags():
    g = G6EgressGate(enabled=True)
    retrieved = [{"doi": _GROUNDED_DOI}]
    claims = g.build_citation_claims(
        f"see {_GROUNDED_DOI} and the fabricated {_FABRICATED_DOI}", retrieved
    )
    by_id = {c["cited_id"]: c["resolved_source"] for c in claims}
    assert by_id[_GROUNDED_DOI] == {"doi": _GROUNDED_DOI}
    assert by_id[_FABRICATED_DOI] is None


def test_render_annotation_lists_reasons():
    g = G6EgressGate(enabled=True)
    text = g.render_annotation([SimpleNamespace(reason="cited 'X' ungrounded")])
    assert "PALISADE egress check" in text
    assert "flagged 1 unverified claim" in text
    assert "cited 'X' ungrounded" in text


# -----------------------------------------------------------------
# Sidecar wiring
# -----------------------------------------------------------------


def test_g6_ships_and_wires_capability():
    settings = PalisadeSettings(enabled=True, g6_enabled=True, contracts_dir=_CONTRACTS)
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("G6") is True
    caps = sidecar.build_capabilities()
    assert any(isinstance(c, G6EgressCapability) for c in caps)


def test_g6_absent_when_flag_off():
    settings = PalisadeSettings(enabled=True, contracts_dir=_CONTRACTS)  # g6 default off
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.is_gate_enabled("G6") is False
    assert not any(
        isinstance(c, G6EgressCapability) for c in sidecar.build_capabilities()
    )


# -----------------------------------------------------------------
# Capability end-to-end (annotate enforcement)
# -----------------------------------------------------------------


def _ctx(partial: bool = False):
    return SimpleNamespace(partial_output=partial)


def _g6_capability(g6_enabled: bool = True) -> G6EgressCapability:
    settings = PalisadeSettings(
        enabled=True, g6_enabled=g6_enabled, contracts_dir=_CONTRACTS
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    return next(
        c for c in sidecar.build_capabilities() if isinstance(c, G6EgressCapability)
    )


async def _retrieve_then_emit(
    cap: G6EgressCapability, output: str, *, partial=False, rag_result: str = _RAG_RESULT
):
    await cap.before_run(_ctx())
    await cap.after_tool_execute(
        _ctx(),
        call=None,
        tool_def=SimpleNamespace(name="rag_search"),
        args={"kb_slug": "molten-salt-papers", "query": "flinak viscosity"},
        result=rag_result,
    )
    return await cap.after_output_process(
        _ctx(partial=partial), output_context=None, output=output
    )


def _findings_text(cap: G6EgressCapability) -> str:
    """Concatenated markdown of the egress blurbs recorded this turn.

    Egress findings now travel out of band on the sidecar buffer (surfaced
    as a warning blurb after the answer) rather than being appended to the
    answer text, so the annotate-mode tests assert against this."""
    return "\n".join(f["message"] for f in cap.sidecar.egress_findings)


def test_egress_annotates_ungrounded_citation():
    cap = _g6_capability()
    original = f"FLiBe density is 1940 kg/m3 (doi:{_FABRICATED_DOI})."
    out = asyncio.run(_retrieve_then_emit(cap, original))
    assert out == original  # answer text left unchanged; finding is out of band
    blurb = _findings_text(cap)
    assert "PALISADE egress check" in blurb
    assert "ungrounded" in blurb
    assert _FABRICATED_DOI in blurb
    assert cap.sidecar.egress_findings[0]["severity"] == "warning"


def test_egress_block_mode_raises_model_retry():
    """g6_block_mode=True: an unverified citation forces a regenerate."""
    import pytest
    from pydantic_ai.exceptions import ModelRetry

    settings = PalisadeSettings(
        enabled=True, g6_enabled=True, g6_block_mode=True, contracts_dir=_CONTRACTS
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    cap = next(
        c for c in sidecar.build_capabilities() if isinstance(c, G6EgressCapability)
    )
    with pytest.raises(ModelRetry):
        asyncio.run(
            _retrieve_then_emit(
                cap, f"FLiBe density is 1940 kg/m3 (doi:{_FABRICATED_DOI})."
            )
        )


def test_egress_block_mode_off_annotates_not_raises():
    """Default (annotate) mode: the same violation records a finding, never raises."""
    cap = _g6_capability()  # g6_block_mode defaults False
    out = asyncio.run(
        _retrieve_then_emit(
            cap, f"FLiBe density is 1940 kg/m3 (doi:{_FABRICATED_DOI})."
        )
    )
    assert "PALISADE egress check" in _findings_text(cap)  # recorded, no exception


def test_egress_passes_grounded_citation_unchanged():
    cap = _g6_capability()
    original = f"FLiNaK viscosity per DOI: {_GROUNDED_DOI}, as retrieved."
    out = asyncio.run(_retrieve_then_emit(cap, original))
    assert out == original  # grounded in a retrieved source -> no finding
    assert cap.sidecar.egress_findings == []


def test_egress_skips_partial_output_during_streaming():
    cap = _g6_capability()
    streamed = f"FLiBe density 1940 (doi:{_FABRICATED_DOI})"
    out = asyncio.run(_retrieve_then_emit(cap, streamed, partial=True))
    assert out == streamed  # partial chunk -> not checked
    assert cap.sidecar.egress_findings == []


def test_egress_noop_when_no_citation_in_output():
    cap = _g6_capability()
    plain = "FLiNaK is a eutectic of LiF, NaF, and KF used as a coolant salt."
    out = asyncio.run(_retrieve_then_emit(cap, plain))
    assert out == plain
    assert cap.sidecar.egress_findings == []


def test_egress_flags_misattribution_live():
    """Per-value provenance: the value 1.2 lives in the retrieved Williams
    paper, but a *different* DOI is cited for it -> caught as misattribution,
    not merely 'ungrounded'. This is what the id-only check could not see."""
    cap = _g6_capability()
    out = asyncio.run(
        _retrieve_then_emit(
            cap,
            f"NaF-UF4 thermal conductivity is 1.2 W/m-K (doi:{_WRONG_DOI}).",
            rag_result=_RAG_RESULT_NAFUF4,
        )
    )
    assert out == f"NaF-UF4 thermal conductivity is 1.2 W/m-K (doi:{_WRONG_DOI})."
    blurb = _findings_text(cap)
    assert "PALISADE egress check" in blurb
    assert "misattributed" in blurb
    assert _TRUE_DOI in blurb  # the finding names the source it really came from


def test_egress_passes_content_grounded_value():
    """Citing the paper the value actually came from passes unchanged."""
    cap = _g6_capability()
    original = f"NaF-UF4 thermal conductivity is 1.2 W/m-K (doi:{_TRUE_DOI})."
    out = asyncio.run(
        _retrieve_then_emit(cap, original, rag_result=_RAG_RESULT_NAFUF4)
    )
    assert out == original


# -----------------------------------------------------------------
# WB3 — scientific-value claims at egress
# -----------------------------------------------------------------


def test_scientific_claims_extracts_predicted_value():
    g = G6EgressGate(enabled=True)
    claims = g.scientific_claims(
        "The predicted melting point for BeF2-NaF-UF4 (0.5/0.3/0.2) is approximately 1023 K."
    )
    assert len(claims) == 1
    c = claims[0]
    assert c["type"] == "melting_point" and c["value"] == 1023.0
    assert c.get("salt") == "BeF2-NaF-UF4"


def test_scientific_claims_picks_value_not_temperature_condition():
    g = G6EgressGate(enabled=True)
    claims = g.scientific_claims("The density of FLiBe at 873 K is 2380 kg/m3.")
    by_type = {c["type"]: c["value"] for c in claims}
    assert by_type.get("density") == 2380.0  # the value, not the 873 K condition


def test_scientific_claims_ignores_unquantified_prose():
    g = G6EgressGate(enabled=True)
    assert (
        g.scientific_claims("FLiNaK has low viscosity and good thermal conductivity.")
        == []
    )


def test_egress_flags_implausible_prediction():
    """A grossly out-of-range predicted value is flagged by physical_bounds at
    egress. Novel composition -> no MSTDB surrogate -> only the gross check."""
    cap = _g6_capability()
    out = asyncio.run(
        _retrieve_then_emit(
            cap,
            "The predicted melting point for BeF2-NaF-UF4 (0.5/0.3/0.2) is 9000 K.",
        )
    )
    blurb = _findings_text(cap)
    assert "PALISADE egress check" in blurb
    assert "melting_point" in blurb and ("outside" in blurb or "plausible" in blurb)


def test_egress_flags_off_surrogate_value():
    """An in-range but inaccurate value (FLiBe density 2380 vs ~1990) is flagged
    by data_value at egress."""
    cap = _g6_capability()
    out = asyncio.run(
        _retrieve_then_emit(cap, "The density of FLiBe at 873 K is 2380 kg/m3.")
    )
    blurb = _findings_text(cap)
    assert "PALISADE egress check" in blurb
    assert "deviates" in blurb or "MSTDB" in blurb


def test_egress_passes_plausible_prediction():
    """A physically-plausible prediction for a novel composition passes clean."""
    cap = _g6_capability()
    original = (
        "The predicted melting point for BeF2-NaF-UF4 (0.5/0.3/0.2) is "
        "approximately 1023 K."
    )
    out = asyncio.run(_retrieve_then_emit(cap, original))
    assert out == original
    assert cap.sidecar.egress_findings == []


# -----------------------------------------------------------------
# WB5 — provenance source-typing (drawing on untrusted uploads)
# -----------------------------------------------------------------


def test_uploaded_references_extracted():
    g = G6EgressGate(enabled=True)
    refs = g.uploaded_references(
        "Based on /mnt/data/uploads/Molten_Salt_Thermophysical_Properties.json "
        "the data shows ..."
    )
    assert refs == ["Molten_Salt_Thermophysical_Properties.json"]


def test_no_uploaded_references_when_absent():
    g = G6EgressGate(enabled=True)
    assert g.uploaded_references("The density of FLiBe is 1990 kg/m3.") == []


def test_egress_records_provenance_note_for_upload():
    cap = _g6_capability()
    original = "Per /mnt/data/uploads/ms_props.json, the melting point is 1000 K."
    out = asyncio.run(_retrieve_then_emit(cap, original))
    assert out == original  # answer text unchanged
    blurb = _findings_text(cap)
    assert "Provenance" in blurb and "ms_props.json" in blurb and "untrusted" in blurb
    # the provenance disclosure is informational, not a violation
    assert cap.sidecar.egress_findings[-1]["severity"] == "info"


def test_egress_no_provenance_note_without_upload():
    cap = _g6_capability()
    original = "FLiNaK is a eutectic coolant salt."
    out = asyncio.run(_retrieve_then_emit(cap, original))
    assert out == original
    assert cap.sidecar.egress_findings == []


# -----------------------------------------------------------------
# Egress-findings side-channel (sidecar buffer -> result.egress_warnings)
# -----------------------------------------------------------------


def test_sidecar_egress_buffer_accumulates_and_resets():
    """The per-turn buffer the G6 capability writes and run_stream drains onto
    `ProjectAgentResult.egress_warnings`."""
    pytest.importorskip(
        "vista_backend",
        reason="cross-checks a symbol owned by the reference host application",
    )
    from vista_backend.agents.agents import EgressWarning

    settings = PalisadeSettings(
        enabled=True, g6_enabled=True, contracts_dir=_CONTRACTS
    )
    sidecar = PalisadeSidecar(settings, _make_project())
    assert sidecar.egress_findings == []

    sidecar.record_egress_finding(message="m", findings=["a", "b"])
    sidecar.record_egress_finding(message="n", findings=[], severity="info")
    assert len(sidecar.egress_findings) == 2

    # run_stream builds EgressWarning models from these dicts.
    warnings = [EgressWarning(**w) for w in sidecar.egress_findings]
    assert warnings[0].severity == "warning" and warnings[0].findings == ["a", "b"]
    assert warnings[1].severity == "info"

    sidecar.reset_egress_findings()
    assert sidecar.egress_findings == []
