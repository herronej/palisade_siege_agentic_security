"""E6: the controller-verification classification, with a mock controller.

The live run needs Docker; these pin the deployed-predicate + parser-differential
logic against an injected probe so a wiring mistake fails here, not on the
cluster.
"""

from __future__ import annotations

from tools.g5_adaptive_live import _g5_caught, classify_arm, run_e6
from tools.g5_search import _FIELDS, _OBFS, _functional, _script

# What a controller that ran every injected value would report.
_RESOLVE_ALL = {
    "QOS": "premium",
    "Partition": "gpu-preempt",
    "Account": "msr_thermo",
    "StdOut": "/lustre/scratch/msr/handoff/run-1.out",
    "Dependency": "afterok:918273",
    "Reservation": "msr_reserved",
}


def _g5_view():
    """Independently compute (caught, parses_value) per arm, so the expected
    verdict split is not taken from the module under test."""
    out = {}
    for field, val, chunk, key, reader in _FIELDS:
        for obf, ofn in _OBFS:
            script = _script(key, ofn(val, val))
            caught, _ = _g5_caught(script, chunk)
            out[f"{field}:{obf}"] = (caught, _functional(script, key, val, reader))
    return out


def test_classify_truth_table():
    # Deployed predicate catches it -> controller not consulted.
    assert classify_arm(g5_caught=True, g5_parses_value=True,
                        controller={"QOS": "premium"}, controller_functional=True) == "g5-caught"
    # G5 misses, controller refuses at submit.
    assert classify_arm(g5_caught=False, g5_parses_value=False,
                        controller={"rejected": "bad qos"}, controller_functional=False) == "rejected"
    # G5 misses, controller runs the job but not the injected value.
    assert classify_arm(g5_caught=False, g5_parses_value=False,
                        controller={"QOS": "normal"}, controller_functional=False) == "inert"
    # G5 misses a value it parsed, controller runs it -> provenance evasion.
    assert classify_arm(g5_caught=False, g5_parses_value=True,
                        controller={"QOS": "premium"}, controller_functional=True) == "provenance-evasion"
    # G5 misses a value its parser never read, controller runs it -> parser gap.
    assert classify_arm(g5_caught=False, g5_parses_value=False,
                        controller={"QOS": "premium"}, controller_functional=True) == "parser-differential"


def test_deployed_predicate_catches_every_arm_whose_parser_read_the_value():
    """The taint/field-rule check reads the raw script, so any arm G5's parser
    resolves is caught; only parser-blind arms reach the controller at all."""
    for arm, (caught, parses) in _g5_view().items():
        if parses:
            assert caught, f"{arm}: G5 parsed the value but did not catch it"


def test_a_controller_that_runs_everything_flags_the_uncaught_parser_gaps():
    """With a controller that resolves every value, every arm G5 fails to catch
    is a bypass, and it is a parser differential because those arms are exactly
    the ones G5's parser was blind to."""
    view = _g5_view()
    r = run_e6(probe=lambda script: dict(_RESOLVE_ALL))
    assert r.controller_version == "injected probe"
    for a in r.arms:
        caught, parses = view[a.arm]
        if caught:
            assert a.verdict == "g5-caught"
        else:
            assert not parses  # uncaught arms are parser-blind by construction
            assert a.verdict == "parser-differential"
    assert r.uncaught, "the move set should contain arms G5 does not catch"
    assert len(r.bypasses) == len(r.uncaught)


def test_the_real_shape_no_bypass_when_the_controller_refuses_or_ignores():
    """The live-controller outcome: G5's uncaught arms are all refused or inert,
    so there is no bypass -- the plan's positive result."""
    def probe(script):
        # A controller that does not resolve the obfuscated injections: unknown
        # objects are refused, and it never carries the injected value.
        return {"rejected": "unknown to test cluster"}

    r = run_e6(probe=probe)
    assert r.bypasses == ()
    assert r.parser_differentials == ()
    assert {a.verdict for a in r.arms} <= {"g5-caught", "rejected"}


def test_report_states_the_outcome_and_banner():
    clean = run_e6(probe=lambda script: {"rejected": "x"}).to_markdown()
    assert "Source module: tools.g5_adaptive_live" in clean[:200]
    assert "0 bypasses" in clean
    found = run_e6(probe=lambda script: dict(_RESOLVE_ALL)).to_markdown()
    assert "bypass(es) found" in found


def test_every_arm_is_classified_once():
    r = run_e6(probe=lambda script: {"rejected": "x"})
    assert len(r.arms) == len(_FIELDS) * len(_OBFS)
