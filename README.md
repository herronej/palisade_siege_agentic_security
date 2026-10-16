# PALISADE and SIEGE

Security mediation and adversarial evaluation for AI agents at HPC facilities.

This repository is the artifact for **Security Mediation and Adversarial
Evaluation for Agentic AI at HPC Facilities**, by Emily Herron, Jesse Hines,
Vanessa Lama, Junqi Yin, Jack Hutchins, Amir Sadovnik and Feiyi Wang (Oak Ridge
National Laboratory). To appear in the SC26 Workshops proceedings, 1st
International Workshop on Agentic AI for HPC (AgenticAI4HPC'26), Chicago,
November 2026. See [Citation](#citation).

---

## Overview

AI agents at DOE user facilities retrieve documents, run code and submit batch
jobs. Two failures there have the same shape. An injected instruction that
steers a privileged action and an unsupported citation that reaches the
scientist are both a value arriving at a consequential action from an origin
that action should not accept.

**PALISADE** (Provenance-Aware Labeling and Information-flow Security for
Agentic Deployment Environments) is a sidecar that sits between an agent and
those actions. It labels every value by where it came from, outside the model's
reach, and admits a value at a privileged action only if its label permits.
Detectors that inspect content run alongside, but they can only add denials.
The decision to admit is a deterministic check over labels.

**SIEGE** (Scientific-agent Infrastructure Exploitation and Guardrail
Evaluation) is the benchmark for it: 205 attack instances in 42 classes, a
181-task benign control, and static and adaptive modes. It treats HPC job
submission as an enforced, measured security boundary.

---

## Key terms

| Term | Meaning |
|---|---|
| Sink | An action with a consequential effect: the file, shell and scheduler tools, and the answer returned to the scientist. |
| Soft win | An attack action a gate admits, meaning it got past inspection. |
| Hard win | An untrusted value admitted at a sink, or a forged citation reaching the answer. Only a hard win breaks the information-flow model. |
| Sink-capable | An attack instance that can reach a sink at all. 32 of the 150 security instances are; hard-win counts below use this denominator. |
| Production bound | Hard wins scored with labels as the runtime reconstructs them, by matching text. |
| Recorded-label bound | Hard wins scored with the labels the corpus author attached, standing in for perfect label tracking. |
| Credited admission | An adaptive-search win: a rewrite admitted at a sink while untrusted taint is live. It does not confirm the rewrite still carries the attack. |

---

## How PALISADE works

PALISADE places one gate at each boundary the deployment exposes.

| Gate | Boundary | What it checks |
|---|---|---|
| G1 | Prompt | Attack signatures over de-obfuscated text; CUI and PII markers |
| G2 | Tool | Tool allow-list, descriptor hash, JSON-Schema validation, sink predicate |
| G3 | Retrieval | Corpus allow-list, sensitivity tier against session trust, query injection, manifest hash |
| G4 | Code | High-risk constructs, sandbox path scoping, execution indicators, optional Semgrep |
| G5 | Slurm submission | Twelve policy checks over the parsed script, including resource ceilings, lifecycle hooks, chained-job dependencies, and which source proposed each scheduler field |
| G6 | Answer (egress) | Every cited identifier must appear in the turn's retrieval ledger. Annotates a violating answer by default; blocks in the recommended posture |

**Labels.** Every value carries four labels (source, taint, sensitivity tier,
dual-use marker) and a provenance chain. Taint is assigned by origin, not
content: every prompt, tool return and retrieved chunk is registered as
untrusted, so rewriting a payload cannot change its taint. The planning model
never writes a label. Dual-use is the one label a model assigns, from the
quarantined model's classification.

**Sink check.** A privileged sink admits an argument only if none of its taint
comes from outside the trusted principals: the authenticated user and the
system and operator channels.

**The limit.** A sidecar on an unmodified agent loop cannot observe dataflow,
so labels follow values by matching text. A sink argument inherits a label if
it reuses an untrusted value verbatim or keeps a distinctive run of it (at
least 12 characters containing a path, URL or shell metacharacter, or a
spaceless letter-and-digit string), including under base64, hex or ROT13
encoding. A paraphrase arrives unlabeled. The one exception is a scheduler
field with a closed vocabulary (QoS, partition, account, reservation): G5 reads
the parsed value and checks which source proposed it, so a paraphrase is still
attributed.

---

## Results

On authored attacks, the deterministic tier stops 19 of the 32 sink-capable
instances at 0.28 ms per turn. Adaptive search recovers admission, but a hand
audit finds that about half of the admitted rewrites no longer carry the
attack, and every admission search added reached the sink without a label.
**The deployed configuration is not adaptively secure, and the failure lies in
label propagation, not in the sink check.**

### Authored attacks (static corpus, seed 42)

| Configuration | Hard wins, of 32 |
|---|---|
| Undefended | 32 |
| `full` and the recommended posture (deterministic tier) | 13 |
| `+all` (adds the model-backed tier and Semgrep) | 4 |
| Recorded-label bound, deterministic tier | 4 |

`+all` closes nine more instances but raises a benign turn from 0.28 ms to
about 2.5 s, so the recommended posture is the deterministic tier: `full` with
G6 set to block, the G1 weaponization signature disabled,
`sticky_high_stakes` off, and lifecycle-hook denials routed at SEV2 (paper,
Section V-G).

### Adaptive attacks (PAIR and TAP, budget 12 proposals, five seeds)

| Measure | Result |
|---|---|
| Credited admissions, gpt-oss-120b proposer | means of 28.0 to 31.0 of 32 across PALISADE configurations |
| Credited admissions, Mistral Small 4 proposer | 22.8 of 32 for PAIR and TAP both, against the recommended posture |
| Admitted rewrites still carrying the attack (hand audit, four runs) | about 14 per run (56 of 119), against 13 unsearched |
| Hard wins under the recorded-label bound | 4 of 32 for every attack and seed, by construction (an implementation check, paper Section V-E) |

### Cost

| Measure | Result |
|---|---|
| Benign controls blocked, recommended posture | 3.3% (6 of 181) |
| AgentDojo v1.2.1 benign tasks broken, deterministic proxy | 17% of those the undefended agent completes |
| Fast-tier latency | 0.28 ms per benign turn |

### What the results mean

The two bounds separate the sink check from label propagation. Under the
production bound, search finds paraphrases that reach the sink unlabeled. Under
the recorded-label bound the sink reads the label the corpus author attached,
and the residual stays at 4 of 32 by construction: that run checks that no
rewrite reaches the sink check through argument content. The difference
between the two bounds is the label-propagation residual.

The four instances that survive the recorded-label bound are all `xc_1`
cross-boundary chains, whose sink argument is authored fresh and shares no text
with the poisoned read. No label reaches their sink under either bound, so the
sink check never evaluates them. Closing them needs dataflow tracking or
keeping untrusted values out of the planner's context (paper, Section VI-B).
The model-backed tier closes three of the four.

---

## Quickstart

Requires Python 3.14.3 and [uv](https://docs.astral.sh/uv/). Everything below
runs on one workstation with no facility allocation and no account.

```bash
uv sync                                          # install both packages
uv run pytest                                    # 2000+ deterministic tests, no network
uv run python -m tools.soft_hard_decomposition   # replay the corpus through the gate stack
uv run python -m tools.corpus_datasheet          # regenerate the corpus datasheet
```

The sandbox-mediation probe and the B5 sweeps need a local Docker or Podman
daemon. The `+semgrep` column needs the Semgrep binary; without it the
static-analysis step does not run and the column returns `full`'s 13 of 32
rather than 12.

[`REPRODUCING.md`](REPRODUCING.md) covers every reported result, including
which need a served model and which reproduce offline.

---

## Layout

```
src/palisade/          the sidecar
  sidecar.py             hook wiring; master flag off => empty hook list
  config.py              every toggle; PALISADE_<FIELD> from the environment
  host.py                the host-application contract
  paths.py               repo anchors used by the analysis modules
  gates/                 G1 prompt · G2 tool · G3 retrieval · G4 code
                         G5 scheduler · G6 egress, plus the Slurm parser
  capabilities/          PydanticAI capability shims + label propagation
  contracts/             the 11-contract domain library + bundled Semgrep rules
  trust.py incidents.py  sticky-floor trust scorer and incident playbook
  provenance.py          one chokepoint; every decision emitted as an event

siege/src/siege/       the benchmark  (separate distribution)
  corpus/                205 attack instances, one directory per class
  controls/              the 181-task benign control
  templates/             the authoring templates each class derives from
  redteam/               the adaptive adversary (PAIR/TAP drivers, optimizers)
  eval/                  the replay harness and per-gate runners
  scorer.py              soft-win / hard-win adjudication
  ablation_matrix.py     the nine ablation configurations (six cumulative,
                         three independent additions to full)

tools/                 analysis modules; one per reported number
docs/palisade/         generated reports + the result-to-module map
docker/slurm/          single-node slurmctld (Slurm 23.11.4) for the B5 sweeps
palisade_contracts/    operator contract directory, loaded beside the built-ins
```

`tools/` is not packaged into the wheel. Run its modules from a checkout as
`python -m tools.<module>`, which is how the paper cites them.

---

## Installing PALISADE into a host application

The repository is a uv workspace with two packages. A host depends on
`palisade` alone (444 KB, no attack corpus); `siege` stays out of the
deployment.

```toml
# the host's pyproject.toml
dependencies = ["palisade==1.0.0"]

[tool.uv.sources]
palisade = { git = "https://github.com/herronej/palisade_siege_agentic_security", tag = "v1.0.0", subdirectory = "." }
```

Pin a tag, not a branch. The reported numbers come from a release tag, and a
gate stack that changes underneath a deployment is an unreviewed security
change.

---

## Using PALISADE as a sidecar

PALISADE wraps an existing agentic framework rather than replacing it. With
the master flag off, the hook list is empty and the agent is byte-identical to
the unguarded baseline.

```python
from pydantic_ai import Agent
from palisade.config import PalisadeSettings
from palisade.sidecar import PalisadeSidecar

settings = PalisadeSettings(enabled=True, g1_enabled=True, g5_enabled=True)
sidecar = PalisadeSidecar(settings, my_project)   # your own project object
agent = Agent(..., capabilities=sidecar.build_capabilities())
```

`my_project` is your own object, passed as is. The sidecar reads only `id`,
`name`, `knowledge_bases` and `tools`, as stated by the `typing.Protocol`
`palisade.host.HostProject`, so any object with those attributes works without
a base class or adapter. Hosts without such an object can use the concrete
`HostProjectModel`.

Configure PALISADE through `PALISADE_<FIELD>` environment variables (see
`.env.sample`), or nest `PalisadeSettings` in your own settings object.

The reference deployment is VISTA, a FastMCP/PydanticAI platform whose prompt
interface, tool layer, retrieval, sandboxed execution, Slurm submission and
citation egress exercise six of the eight boundaries in the threat model (B0
through B5). The tests under `src/palisade/tests/host_integration/` run
PALISADE inside VISTA and skip when it is absent. No reported result depends on
them.

---

## Safety and release

Public attack payloads are **detection-pattern triggers with placeholder hosts,
not working exploits**. All dual-use content is surrogate-only.

The adaptive machinery is public too: the PAIR/TAP optimizer drivers, the
transform libraries and the reward-ranked strategy library. The paper argues
that a defense tested only against fixed attacks says nothing about its worst
case, and readers cannot check that argument against our system if the attacks
that move it are withheld. The reported PAIR and TAP runs start from each
instance's authored payload, use no pre-ranked pool, and exclude the encode and
round-trip transforms (paper, Appendix D-A).

[`docs/palisade/RELEASE_SAFETY.md`](docs/palisade/RELEASE_SAFETY.md) lists what
the release contains, what is not redistributable, and how the payload claim
above is verified.

---

## Citation

If you use PALISADE or SIEGE, please cite the paper:

```bibtex
@inproceedings{herron2026palisade,
  author    = {Herron, Emily and Hines, Jesse and Lama, Vanessa and Yin, Junqi and
               Hutchins, Jack and Sadovnik, Amir and Wang, Feiyi},
  title     = {Security Mediation and Adversarial Evaluation for Agentic {AI}
               at {HPC} Facilities},
  booktitle = {Proceedings of the SC '26 Workshops of the International
               Conference for High Performance Computing, Networking,
               Storage and Analysis},
  series    = {SC Workshops '26},
  publisher = {IEEE},
  address   = {Chicago, IL, USA},
  year      = {2026},
  note      = {1st International Workshop on Agentic AI for HPC
               (AgenticAI4HPC'26). To appear}
}
```

For the software itself, see [`CITATION.cff`](CITATION.cff).

## Acknowledgment

This software was authored by UT-Battelle, LLC under Contract No.
DE-AC05-00OR22725 with the U.S. Department of Energy. See [`NOTICE`](NOTICE).
