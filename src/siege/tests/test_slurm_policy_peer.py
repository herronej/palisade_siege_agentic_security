"""The facility-policy peer on SIEGE's B5 family (W36.1).

These tests pin the peer's *mechanism*, not its corpus rates, so they stay
meaningful as the corpus grows. Two of them lock regressions that already bit
once during development and would each have silently invalidated the experiment:

* a case-insensitive ``sbatch`` rule matches the ``#SBATCH`` directive prefix on
  every well-formed script, denying the entire corpus (including 100% of the
  benign control) while looking like a spectacular result;
* the account lives in ``user_config['hpc_account']``, not in the script, so a
  peer that reads only the directive denies everything on ``allocation``.
"""
from __future__ import annotations

from palisade.gates.slurm_parser import parse_slurm_script
from tools.slurm_policy_peer import (
    _PERMITTED_PARTITIONS,
    _PERMITTED_QOS,
    _PERMITTED_RESERVATIONS,
    evaluate_policy,
    in_policy_variants,
    run_peer,
    to_in_policy,
)

ACCOUNT = {"hpc_account": "msr_thermo"}

_LEGIT = (
    "#!/bin/bash\n"
    "#SBATCH --nodes=4\n"
    "#SBATCH --time=02:00:00\n"
    "srun cp2k.psmp -i salt.inp -o salt.out\n"
)


def _verdict(script: str, user_config=ACCOUNT):
    return evaluate_policy(parse_slurm_script(script), user_config)


def _with(directive: str) -> str:
    return _LEGIT.replace("srun", f"{directive}\nsrun", 1)


# ---------------------------------------------------------------- regressions


def test_directive_prefix_is_not_a_nested_submission() -> None:
    """`#SBATCH` must not trip the nested-`sbatch` rule.

    A case-insensitive rule here denies every well-formed script. Lua's
    string.find is case-sensitive; the peer must be too.
    """
    assert not _verdict(_LEGIT).denied


def test_real_nested_submission_is_denied() -> None:
    assert _verdict(_LEGIT + "sbatch stage2.slurm\n").rule == "body_denylist"


def test_account_resolves_from_user_config() -> None:
    """The corpus puts the account in user_config, not the script."""
    assert not _verdict(_LEGIT, {"hpc_account": "msr_thermo"}).denied
    assert _verdict(_LEGIT, {}).rule == "allocation"
    assert _verdict(_LEGIT, {"hpc_account": "someone_elses"}).rule == "allocation"


# ------------------------------------------------------------- policy surface


def test_out_of_allowlist_fields_are_denied() -> None:
    assert _verdict(_with("#SBATCH --qos=premium")).rule == "qos"
    assert _verdict(_with("#SBATCH --partition=gpu-preempt")).rule == "partition"
    assert _verdict(_with("#SBATCH --reservation=msr_reserved")).rule == "reservation"


def test_in_allowlist_fields_are_admitted() -> None:
    """The same fields with values legitimate work uses must pass, or the peer
    is a strawman that denies by breadth rather than by policy."""
    for directive in (
        "#SBATCH --qos=high",
        "#SBATCH --partition=batch",
        "#SBATCH --reservation=msr_thermo_dedicated",
    ):
        assert not _verdict(_with(directive)).denied, directive


def test_output_path_scoping() -> None:
    outside = "#SBATCH --output=/lustre/scratch/msr/handoff/run-%j.out"
    inside = "#SBATCH --output=/lustre/orion/proj-shared/msr_thermo/run-%j.out"
    assert _verdict(_with(outside)).rule == "path_scope"
    assert not _verdict(_with(inside)).denied


def test_ceilings_come_from_the_shared_policy_object() -> None:
    """Calibrated to the same allocation policy the production G5 gate uses, so
    the comparison is not against a ceiling we chose."""
    assert not _verdict(_with("#SBATCH --nodes=64")).denied
    assert _verdict(_with("#SBATCH --nodes=65")).rule == "ceiling_nodes"
    assert _verdict(_with("#SBATCH --time=999:00:00")).rule == "ceiling_time"


# ------------------------------------------------------- the load-bearing gap


def test_wellformed_dependency_is_structurally_invisible() -> None:
    """The `b5_11` residual. An injected dependency is byte-indistinguishable in
    form from legitimate use; only the job id differs, and no submit-time rule
    can adjudicate that."""
    attack = _with("#SBATCH --dependency=afterok:918273")
    benign = _with("#SBATCH --dependency=afterok:20480")
    assert not _verdict(attack).denied
    assert not _verdict(benign).denied


def test_arm_b_rewrite_makes_every_injection_in_policy() -> None:
    """Arm B: with in-policy values the peer has nothing left to object to."""
    for directive, key in (
        ("#SBATCH --qos=premium", "qos"),
        ("#SBATCH --partition=gpu-preempt", "partition"),
        ("#SBATCH --reservation=msr_reserved", "reservation"),
        ("#SBATCH --output=/lustre/scratch/msr/handoff/run-%j.out", "output"),
    ):
        original = _with(directive)
        assert _verdict(original).denied, f"Arm A should deny {key}"
        rewritten, applied = to_in_policy(original)
        assert key in applied, f"rewrite did not fire for {key}"
        assert not _verdict(rewritten).denied, f"Arm B should admit {key}"


# ------------------------------------------------------------- C7: searched Arm B


def test_arm_b_search_enumerates_the_full_allowlist() -> None:
    """The searched Arm B enumerates the whole sanctioned set per field, not one
    value, and the policy must admit every member (it reads the value)."""
    original = _with("#SBATCH --qos=premium")
    variants = in_policy_variants(original)
    assert set(variants) == {"qos"}
    # The enumerated qos values are exactly the policy's permitted set.
    assert {v for _s, v in variants["qos"]} == set(_PERMITTED_QOS)
    for script, _val in variants["qos"]:
        assert not _verdict(script).denied


def test_arm_b_search_over_corpus_admits_every_sanctioned_value() -> None:
    """On the real b5_11 corpus the policy admits 100% of in-policy variants
    across every injected field -- the Arm B result is searched, not authored."""
    result = run_peer()
    search = result.arm_b_search
    assert search, "no searched Arm B fields found on the corpus"
    for field, fs in search.items():
        assert fs.values_tried > 0
        assert fs.admitted == fs.values_tried, (
            f"policy denied an in-policy {field} value: {fs.admitted}/{fs.values_tried}"
        )
    # The allow-list-derived fields enumerate their full permitted set.
    if "partition" in search:
        assert set(search["partition"].values) == set(_PERMITTED_PARTITIONS)
    if "reservation" in search:
        assert set(search["reservation"].values) == set(_PERMITTED_RESERVATIONS)
