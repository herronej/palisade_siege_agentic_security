# Release safety and artifact tiering

SIEGE is an attack corpus. Releasing it has to measure the defense without
handing an attacker a turnkey evasive corpus, so the artifact is split into
three tiers. This document states what is in each, and — for the claims that
are checkable — how they are checked rather than asserted.

This supersedes the earlier `redteam_artifact_manifest.md`, which named five
paths that never existed (`redteam/transforms/`, `redteam/oracles/
effect_equivalence.py`, `redteam/chained.py`, `redteam/oscillation.py`,
`policies/`). The tiering below names paths that are in this repository.

---

## Tier 1 — Public (this repository)

The methodology, the mechanism, and the measurements.

| Artifact | Path |
|---|---|
| The six gate implementations | `src/palisade/gates/`, `src/palisade/capabilities/` |
| The information-flow model: labels, propagation, sink predicates | `src/palisade/capabilities/propagation.py`, `registry.py` |
| Trust scorer, incident playbook, provenance bus | `src/palisade/trust.py`, `incidents.py`, `provenance.py` |
| The contract library (11 contracts) and bundled Semgrep rules | `src/palisade/contracts/` |
| The 205-instance attack corpus in 42 classes | `src/siege/corpus/b*/`, `src/siege/corpus/xc*/` |
| The 181-task benign control | `src/siege/corpus/benign_workload/`, `benign_diverse/` |
| Per-class ground truth and adjudication | `src/siege/oracles/`, `src/siege/scorer.py` |
| The replay harness and the nine-configuration ablation | `src/siege/eval/`, `src/siege/ablation_matrix.py` |
| All analysis modules behind reported numbers | `tools/` |
| The facility `job_submit.lua` policy comparison | `tools/artifacts/job_submit.lua` |
| The containerized single-node `slurmctld` | `docker/slurm/` |

## Tier 2 — Gated to vetted partners

The components that together would constitute a turnkey evasive corpus. These
are **present in this repository today and must be removed before it is made
public** — see the checklist at the end.

| Artifact | Path | Why gated |
|---|---|---|
| Trained attack policies (bandit state, learned proposal distributions) | `src/siege/redteam/policy.py`, `attacker.py` | Learned attack distributions |
| Transform libraries — semantics-preserving evasion operators | `src/siege/redteam/realizer.py`, `redteam/attacks/llm/strategy_library.py`, `attacks/llm/intent_laundering.py`, `attacks/code/correctness_sabotage.py`, `attacks/embedding/` | The operators a propagation search would drive |
| Reward-ranked evasive pools (80 instances, 4 classes × 20) | `src/siege/corpus_generated/` | Reward-ranked full-config evaders |
| The reward model that ranks them | `src/siege/redteam/reward.py`, `generator.py` | Ranks evasions by success |
| The optimizing drivers | `src/siege/redteam/llm_optimizer.py`, `embedding_optimizer.py` | PAIR/TAP search over the above |

**What a reviewer loses without Tier 2.** Every result in the paper is
executable from Tier 1 *except* the reward-ranked initialization of the
adaptive drivers behind Section V-E, which reproduces from the public transform
set at a lower starting point. The paper says this; it is true of this split.

The transform libraries would also drive a search of the propagation closure —
the experiment Section VI-A names as the one the evaluation most obviously
calls for. They are released to vetted partners for that purpose. No such
experiment is reported here.

## Tier 3 — Not redistributable

| Artifact | Status |
|---|---|
| MSTDB-TP thermophysical database | Obtained under its own terms. This artifact ships the ingestion manifest and content hashes, **not** the database. |
| Slurm | Not vendored; `docker/slurm/` builds from upstream images at 23.11.4. |
| Detector baseline weights (Prompt-Guard-86M, Prompt Guard 2) | Downloaded from upstream under their own licenses. |
| AgentDojo | Optional PyPI dependency at 0.1.35; not vendored. |

---

## The payload-safety claim, and how it is checked

The paper claims: *public attack payloads are detection-pattern triggers with
placeholder hosts rather than working exploits, and all dual-use content is
surrogate-only.*

The host half of that claim is mechanically checkable, and `tools/tests/
test_release_safety.py` checks it: every network identifier appearing in the
public corpus must be drawn from a reserved range that cannot route to a real
system —

- **RFC 2606 / 6761 reserved names**: `.example`, `.invalid`, `.test`,
  `.localhost`, and `example.com` / `.net` / `.org`
- **RFC 5737 documentation addresses**: `192.0.2.0/24`, `198.51.100.0/24`,
  `203.0.113.0/24`
- **RFC 1918 private addresses**: `10/8`, `172.16/12`, `192.168/16`

Four violations were found and fixed during release preparation, each of which
would have shipped a payload pointed at a real-world identifier:

| Was | Now | Where | Severity |
|---|---|---|---|
| `stratum+tcp://gulf.moneroocean.stream:10128` | `pool.attacker.example` | `b5_1_mining` | **live mining pool** |
| `stratum+ssl://xmr.2miners.com:12121` | `pool.attacker.example` | `b5_1_mining`, `siege.eval.g5_attacks` | **live mining pool** |
| `45.9.148.7` (a routable public address) | `198.51.100.45` | `b5_2_credential_exfil`, `siege.eval.g5_attacks` | routable |
| `attacker.io`, `mining.attacker.io`, `x.attacker.io` (a registrable `.io` domain) | `attacker.example` and friends | `b4_1_malicious_code`, `b4_2_network_exfil`, `gen_b5_smuggling`, `siege.eval.g4_attacks` | registrable |

The first two are the serious ones: a public corpus carrying a working stratum
URL is a payload aimed at live infrastructure, which is precisely what the
paper's claim says the corpus does not contain. They were caught by the check
below, not by inspection, which is the argument for having the check.

**These fixes are in the release repository only.** The authoring source in
VISTA still carries the original strings; regenerating the corpus from there
would reintroduce all four. Backport before the next regeneration.

**Deliberately retained.** Real mining-pool hostnames (`pool.minexmr.com`,
`supportxmr.com`) appear in `src/palisade/gates/denylists.py` and the tests
that exercise it. These are *detection signatures* — the strings G5 matches
against — not payload destinations. A defensive denylist has to name what it
blocks, and removing them would silently weaken the gate. They are the only
real-world hosts in the public tier, and they appear only on the defense side.

**Not mechanically checkable.** That a payload is a detection-pattern trigger
rather than a working exploit is a property of intent and construction, not of
syntax. The corpus is authored, not collected, and each instance declares a
programmatic success criterion the harness checks; no instance carries a
functioning exploit chain. This rests on the authoring procedure documented in
`corpus_datasheet.md`, not on an automated check.

**Dual-use content.** Surrogate-only. No instance carries a real synthesis
route, agent, or protocol. The dual-use axis is deployed but closes no attack
in the reported measurements (Section VI-A), and the corpus exercises it only
through framing.

---

## Pre-publication checklist

Work through this before the repository is made public.

- [ ] **Remove Tier 2.** Excise the paths in the gated table above, and confirm
      the remaining tree still imports and tests clean. The adaptive drivers'
      *interfaces* stay public — only the trained state, the transform
      operators and the ranked pools leave.
- [ ] **Re-run the payload scan.** `uv run pytest tools/tests/test_release_safety.py`
- [ ] **Confirm no credentials.** No `.env` is tracked; verify with
      `git log -p | grep -iE 'api[_-]?key|sk-[A-Za-z0-9]{20,}'` over the full
      history, not just the tip. This repository was created fresh rather than
      filtered from VISTA's history, so there is no prior history to scrub —
      keep it that way.
- [ ] **Confirm MSTDB-TP is absent.** Only the manifest and hashes ship.
- [ ] **Settle the license** (see `LICENSE`) and record the ORNL release number
      in `NOTICE`.
- [ ] **Mint the DOI**, tag the release, and update `CITATION.cff`. The state
      behind the reported numbers must be a tag, not a branch head.
- [ ] **Set up the gated-access contact** on the artifact landing page, so the
      Tier 2 request path in the paper resolves to something.
- [ ] **Backport the payload scrub to VISTA.** The four fixes above exist only
      in this repository; the authoring source still carries the originals.
- [ ] **Re-run the full suite** (`uv run pytest`) and confirm it is green after
      Tier 2 removal, since some `tools/tests/` exercise the gated modules.
