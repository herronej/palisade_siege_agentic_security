"""E1a adaptive: folding error corrections into the full sweeps."""

from __future__ import annotations

import pytest

from tools.prompt_monitor_adaptive import merge


def _run(strategy, seed, hard, errors, templates=None):
    return {
        "strategy": strategy,
        "templates": templates,
        "seeds": [
            {"seed": seed, "hard_ids": list(hard),
             "monitor_stats_delta": {"errors": errors}}
        ],
    }


def test_a_correction_replaces_its_class_as_a_unit():
    full = _run("pair", 3, ["b5_11_injected_qos_00", "xc_5_forged_tool_result_03"], 6)
    fix = _run("pair", 3, ["xc_5_forged_tool_result_03", "xc_5_forged_user_consent_04"],
               0, templates=["xc_5_history_forgery"])
    merged, notes = merge([full, fix])
    assert len(merged) == 1
    hard = merged[0]["seeds"][0]["hard_ids"]
    assert sorted(hard) == [
        "b5_11_injected_qos_00",
        "xc_5_forged_tool_result_03",
        "xc_5_forged_user_consent_04",
    ]
    assert "6 monitor error(s)" in notes[0] and "2 -> 3" in notes[0]


def test_a_correction_can_also_lower_the_count():
    """Replacement is by class, not a union: a win the rerun does not repeat is
    dropped, so the correction cannot only ever add."""
    full = _run("tap", 0, ["xc_5_forged_tool_result_03"], 5)
    fix = _run("tap", 0, [], 0, templates=["xc_5_history_forgery"])
    merged, _ = merge([full, fix])
    assert merged[0]["seeds"][0]["hard_ids"] == []


def test_other_seeds_and_strategies_are_untouched():
    runs = [
        _run("pair", 3, ["xc_5_a_00"], 6),
        _run("pair", 4, ["xc_5_a_00"], 6),
        _run("tap", 3, ["xc_5_a_00"], 7),
        _run("pair", 3, [], 0, templates=["xc_5_history_forgery"]),
    ]
    merged, _ = merge(runs)
    by_key = {(r["strategy"], r["seeds"][0]["seed"]): r["seeds"][0]["hard_ids"] for r in merged}
    assert by_key[("pair", 3)] == []
    assert by_key[("pair", 4)] == ["xc_5_a_00"]
    assert by_key[("tap", 3)] == ["xc_5_a_00"]


def test_a_correction_without_a_full_run_is_an_error():
    with pytest.raises(SystemExit):
        merge([_run("pair", 9, [], 0, templates=["xc_5_history_forgery"])])


def test_the_valid_comparison_leaves_out_the_excluded_instances():
    from tools.prompt_monitor_adaptive import to_markdown

    def full(strategy, seed, hard, errors):
        r = _run(strategy, seed, hard, errors)
        r.update(budget=12, scored_n=4, proposer_model="p", monitor_model="m",
                 policy_sha256="0" * 64, monitor_stats={"errors": errors})
        return r

    runs = [
        full("pair", 0, ["a_00", "b_01", "xc_5_z_03"], 0),
        full("pair", 1, ["a_00"], 3),
    ]
    md = to_markdown(runs, [1], exclude=["xc_5_z_03"], exclusion_note="endpoint failed")
    assert "## Valid comparison" in md and "endpoint failed" in md
    # Seed 0 has 2 of the 3 kept instances, seed 1 has 1: mean 1.5.
    assert "| PAIR | 1.5 [1, 2] (50%) |" in md
    # Only the error-free seed is reported as valid over the whole set.
    assert "PAIR seed 0: 3/4" in md and "PAIR seed 1:" not in md.split("## Valid comparison")[1].split("## PAIR")[0]


def test_one_correction_can_replace_two_classes():
    """TAP seed 1 lost the endpoint during xc_4, so its rerun covers xc_4 and
    xc_5 together and both classes are replaced."""
    full = _run("tap", 1, ["b4_9_x_00", "xc_4_substring_slice_04", "xc_5_a_00"], 22)
    fix = _run("tap", 1, ["xc_4_base64_exec_01", "xc_4_substring_slice_04", "xc_5_b_01"],
               0, templates=["xc_4_taint_laundering", "xc_5_history_forgery"])
    merged, notes = merge([full, fix])
    assert sorted(merged[0]["seeds"][0]["hard_ids"]) == [
        "b4_9_x_00", "xc_4_base64_exec_01", "xc_4_substring_slice_04", "xc_5_b_01",
    ]
    assert "xc_4_taint_laundering, xc_5_history_forgery" in notes[0]


def _report_run(strategy, seed, hard, endpoint=None):
    run = _run(strategy, seed, hard, 0)
    run.update(
        budget=12, scored_n=32, proposer_model="openai/gpt-oss-120b",
        monitor_model="openai-chat:openai/gpt-oss-120b", policy_sha256="ab" * 32,
        monitor_stats={"model_calls": 10},
    )
    if endpoint is not None:
        run["endpoint"] = endpoint
    return run


def test_report_names_its_host_its_static_report_and_extra_rows():
    from tools.prompt_monitor_adaptive import to_markdown

    endpoint = {
        "host": "openrouter.ai", "base_path": "/api/v1", "model": "openai/gpt-oss-120b",
        "openrouter_routing": {"only": ["deepinfra/bf16", "akashml/bf16"]},
    }
    runs = [_report_run("pair", 0, ["a"], endpoint), _report_run("tap", 0, [], endpoint)]
    row = "| PALISADE `full`, this host | 13/32 | 30.0 [29, 31] | 29.0 [28, 30] |"
    md = to_markdown(runs, [5, 6], static_report="prompt_monitor_baseline_openrouter.md",
                     extra_rows=[row])
    assert "Served by `openrouter.ai` pinned to `deepinfra/bf16`, `akashml/bf16`." in md
    assert "(`prompt_monitor_baseline_openrouter.md`)" in md
    assert row in md
    assert md.index("PALISADE `+all` (Table II)") < md.index(row)


def test_report_says_when_the_host_was_not_recorded():
    from tools.prompt_monitor_adaptive import to_markdown

    md = to_markdown([_report_run("pair", 0, [])], [5])
    assert "Served by not recorded (these runs predate host logging)." in md


def test_report_names_the_host_of_a_correction_too():
    from tools.prompt_monitor_adaptive import to_markdown

    full = _report_run("pair", 1, ["b1_9_json_00"])
    fix = _report_run("pair", 1, ["xc_5_forged_tool_result_01"], {
        "host": "openrouter.ai", "openrouter_routing": {"only": ["cerebras/fp16"]},
    })
    fix["templates"] = ["xc_5_history_forgery"]
    md = to_markdown([full, fix], [5])
    assert ("Served by not recorded (these runs predate host logging); "
            "`openrouter.ai` pinned to `cerebras/fp16`.") in md
    assert "| 1 | 2/32 |" in md
