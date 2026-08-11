<!-- Generated report. Source module: tools.slurm_parser_differential. Regenerate with `uv run python -m tools.slurm_parser_differential`. -->

# G5 parser differentials (R5-6)

Each probe expresses one payload two ways: a **control** that G5 should deny, and a **variant** that relies on a disagreement between our SLURM parser and `sbatch`. A **bypass** is control-denied and variant-allowed: Slurm executes the same job either way, so the allow is a parser artifact rather than a policy decision. No label manipulation is involved in any probe.

> **No Slurm on the evaluation host.** The `slurm semantics` column records documented `sbatch` behaviour, not an observed A/B against `slurmctld`. Probes whose reading is contestable are marked *needs live validation*; confirming the rest requires `sbatch --test-only` against a real controller.

**0 of 6 probes bypass.**

| Probe | Check evaded | Control | Variant | Bypass |
|---|---|---|---|---|
| `split_directive` | resource ceiling | deny | deny | no |
| `wrap_body` | egress allow-list | deny | deny | no |
| `wrap_fetch_exec` | body-command screen | deny | deny | no |
| `leading_whitespace` *(needs live validation)* | directive recognition | deny | deny | no |
| `hetjob_component` *(needs live validation)* | resource ceiling | deny | deny | no |
| `crlf_endings` | directive recognition | deny | deny | no |

## Per-probe detail

### `split_directive` — no bypass

- parsed commands: control 1, variant 1
- parsed egress hosts: control [], variant []
- control verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- variant verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- slurm semantics: sbatch stops scanning #SBATCH at the first non-comment, non-blank line, so it honours nodes=512/time=47h and never reads the trailing block. Our parser collects every #SBATCH line in the file and folds them last-wins, so it reads nodes=4/time=1h.

### `wrap_body` — no bypass

- parsed commands: control 2, variant 2
- parsed egress hosts: control ['exfil.example.invalid'], variant ['exfil.example.invalid']
- control verdict: G5 network egress: outbound connection to non-allow-listed host 'exfil.example.invalid'
- variant verdict: G5 network egress: outbound connection to non-allow-listed host 'exfil.example.invalid'
- slurm semantics: --wrap supplies the job body as a directive value; sbatch executes the wrapped string. Our parser treats it as a directive and walks no commands, so the command-derived checks (egress hosts, invoked binaries, file paths) have an empty input.

### `wrap_fetch_exec` — no bypass

- parsed commands: control 3, variant 2
- parsed egress hosts: control ['evil.invalid'], variant ['evil.invalid']
- control verdict: G5 network egress: outbound connection to non-allow-listed host 'evil.invalid'
- variant verdict: G5 network egress: outbound connection to non-allow-listed host 'evil.invalid'
- slurm semantics: as wrap_body; isolates the fetch-and-execute screen.

### `leading_whitespace` — no bypass

- parsed commands: control 1, variant 1
- parsed egress hosts: control [], variant []
- control verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- variant verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- slurm semantics: sbatch requires #SBATCH at column 0; an indented line is a comment and its options are ignored. Our parser strips leading whitespace before matching, so it reads directives Slurm discards. This over-reads rather than under-reads, so it is a false-positive risk, not a bypass.

### `hetjob_component` — no bypass

- parsed commands: control 1, variant 1
- parsed egress hosts: control [], variant []
- control verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- variant verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- slurm semantics: a heterogeneous job's components each carry their own resource request; the second component here asks for 512 nodes. Whether our single-spec model should aggregate or reject outright is a design question, not only a parser one.

### `crlf_endings` — no bypass

- parsed commands: control 1, variant 1
- parsed egress hosts: control [], variant []
- control verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- variant verdict: G5 resource ceiling on allocation 'msr_thermo': nodes 512 > cap 64
- slurm semantics: a CRLF script is honoured; the trailing CR must not defeat value parsing.
