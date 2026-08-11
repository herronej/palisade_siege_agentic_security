# PALISADE and SIEGE

**PALISADE** (Provenance-Aware Labeling and Information-flow Security for Agentic
Deployment Environments) is a privileged security-mediation sidecar for agentic
scientific platforms. It interposes a gate at each boundary a deployment exposes
and enforces an information-flow model in which every value carries four label
axes — source, taint, sensitivity tier, dual-use marker — plus a provenance
chain. The load-bearing admission decisions are **deterministic predicates over
labels assigned outside the model**; the content-inspecting tiers are
subordinate to them and can only add denials.

**SIEGE** (Scientific-agent Infrastructure Exploitation and Guardrail
Evaluation) is its companion benchmark: 205 attack instances across 42 classes
against a 181-task benign control, with a static mode and an adaptive mode. It
scores an evaded inspection tier (**soft win**) separately from a value reaching
a privileged sink with its label stripped (**hard win**). Only the latter
compromises the information-flow model.

This repository is the artifact for *Security Mediation and Adversarial
Evaluation for Agentic AI in Scientific Workflows*.

> **Release status.** This repository is not yet published. `LICENSE` is a
> placeholder pending UT-Battelle software-release review, and the archival DOI
> has not been minted. See [`NOTICE`](NOTICE) for the open items.

---

## What the evaluation found

We state the adaptive result up front rather than hold it, because it bounds
everything else.

| | |
|---|---|
| Hard wins at the recommended deterministic tier, authored payloads | **13 of 32** sink-capable instances |
| Hard wins at the strongest configuration we run (`+all`) | **4 of 32** |
| Hard wins under PAIR / TAP, 12-query budget, five seeds | **28.0–31.0 of 32** against *every* configuration |
| Hard wins under the recorded-label bound | **4 of 32**, invariant to attack and seed |
| Benign controls blocked at the recommended posture | **3.3%**, for a 3.3 pp utility loss |
| Deterministic fast-tier cost | **0.28 ms** per benign turn |

**At a twelve-query budget the deployed configuration is not adaptively
secure.** The two bounds locate why. Under the production bound the runtime
reconstructs a label by matching argument content, and optimization recovers
most of the capable set. Under the recorded-label bound — the sink reading the
label the corpus author attached — no attack and no seed moves the residual.
Every recovered win is a value that arrived at the sink with *no label to
adjudicate*; none is a label the predicate read and admitted.

That gap is the **label-propagation residual**. PALISADE propagates by content
match rather than through an interpreter that observes dataflow, so the closure
is partial by construction. The trade is deployability against closure, and it
is the paper's central limitation, not an incidental one.

The four instances surviving even the recorded-label bound are a genuine defect
the evaluation surfaced in our own system: a privileged sink **fails open on an
unlabelled value**. See `docs/palisade/README.md` for where that is measured.

---

## Layout

```
src/palisade/          the sidecar
  sidecar.py             hook wiring; master flag off => empty hook list
  config.py              every toggle; PALISADE_<FIELD> from the environment
  host.py                the host-application contract (see below)
  paths.py               repo anchors used by the analysis modules
  gates/                 G1 prompt · G2 tool · G3 retrieval · G4 code
                         G5 scheduler · G6 egress, plus the Slurm parser
  capabilities/          PydanticAI capability shims + label propagation
  contracts/            the 11-contract domain library + bundled Semgrep rules
  trust.py incidents.py  sticky-floor trust scorer and incident playbook
  provenance.py          one chokepoint; every decision emitted as an event

src/siege/             the benchmark
  corpus/                205 attack instances, one directory per class
  controls/              the 181-task benign control
  templates/             the authoring templates each class derives from
  redteam/               the adaptive adversary (PAIR/TAP drivers, optimizers)
  eval/                  the replay harness and per-gate runners
  scorer.py              soft-win / hard-win adjudication
  ablation_matrix.py     the nine configurations of the cumulative ladder

tools/                 analysis modules; one per reported number
docs/palisade/         generated reports + the result-to-module map
docker/slurm/          single-node slurmctld (Slurm 23.11.4) for the B5 sweeps
palisade_contracts/    operator contract directory, loaded beside the built-ins
```

`tools/` is deliberately not packaged into the wheel — the analysis modules run
from a checkout as `python -m tools.<module>`, which is how the paper cites
them.

---

## Quickstart

Requires Python 3.14.3 and [uv](https://docs.astral.sh/uv/). Everything below
runs on one workstation with no facility allocation and no account.

```bash
uv sync --extra repro
```

Run the test suite (2000+ tests, deterministic, no network):

```bash
uv run pytest
```

Replay the corpus through the unmodified gate stack:

```bash
uv run python -m tools.soft_hard_decomposition
```

Regenerate the corpus datasheet:

```bash
uv run python -m tools.corpus_datasheet
```

Full reproduction instructions, including which results need a served model and
which reproduce offline, are in [`REPRODUCING.md`](REPRODUCING.md).

---

## Using PALISADE as a sidecar

PALISADE wraps an existing agentic framework rather than owning one. It is
strictly modular: **with the master flag off the hook list is empty and the
agent is byte-identical to the unguarded baseline.**

```python
from palisade.config import PalisadeSettings
from palisade.host import HostProject
from palisade.sidecar import PalisadeSidecar

settings = PalisadeSettings(enabled=True, g1_enabled=True, g5_enabled=True)
sidecar = PalisadeSidecar(project=HostProject(id=..., name="my-project"), settings=settings)
agent = Agent(..., capabilities=sidecar.build_capabilities())
```

`palisade.host` states the minimum shape the sidecar reads from its host — the
project it acts within and the principal it acts for. A host application may
pass its own richer models instead, provided they carry those attributes.

The reference deployment is VISTA, a FastMCP/PydanticAI platform whose
retrieval, sandboxed execution, Slurm submission and citation egress exercise
six of the eight boundaries the threat model names. The tests under
`src/palisade/tests/host_integration/` exercise PALISADE inside that host and
skip automatically without it; nothing the paper reports depends on them.

---

## Running SIEGE against a different defense

The corpus is declarative — one file is one instance, one directory is one
class — and the runner takes any gate stack exposing the same admission
interface. SIEGE is meant to be run against defenses other than ours. See
`src/siege/README.md` for the admission interface and
`docs/palisade/RELEASE_SAFETY.md` for what the public corpus does and does not
contain.

---

## Safety and release tiering

Public attack payloads are **detection-pattern triggers with placeholder hosts,
not working exploits**, and all dual-use content is surrogate-only. A separate
tier — the trained attack policies, the transform libraries and the
reward-ranked evasive pools, which together would constitute a turnkey evasive
corpus — is gated to vetted partners.

[`docs/palisade/RELEASE_SAFETY.md`](docs/palisade/RELEASE_SAFETY.md) states what
sits in each tier, what is not redistributable, and how the claim above is
verified rather than asserted.

---

## Citation

See [`CITATION.cff`](CITATION.cff). The archival DOI is minted at camera-ready;
**the state behind the reported numbers is a release tag, not a branch head.**

## Acknowledgment

This software was authored by UT-Battelle, LLC under Contract No.
DE-AC05-00OR22725 with the U.S. Department of Energy. See [`NOTICE`](NOTICE).
# palisade_siege_agentic_security
