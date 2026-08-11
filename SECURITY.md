# Security

## What this repository is

PALISADE is a defensive mechanism; SIEGE is an attack corpus built to measure
it. The corpus is authored to trip detection patterns, not to work: payloads
name reserved placeholder hosts and dual-use content is surrogate-only. See
[`docs/palisade/RELEASE_SAFETY.md`](docs/palisade/RELEASE_SAFETY.md) for the
tiering and how that claim is checked.

## What PALISADE does not claim

Read this before deploying it. The paper is explicit and so are we:

- **It is not adaptively secure at the budget we tested.** PAIR and TAP at a
  twelve-query budget return hard wins to 28.0–31.0 of 32 sink-capable
  instances against every configuration we run, including the deterministic
  tier we recommend.
- **Label propagation is partial by construction.** Labels are resolved at
  sinks by content match, not by dataflow lineage, so a value transformed past
  the content guard arrives unlabelled and is admitted. This is the residual
  every recovered adaptive win exploits.
- **A privileged sink fails open on an unlabelled value.** The deployed
  high-stakes check denies an argument it resolves to an untrusted label and is
  silent on one it resolves to *no* label. The remediation is stated in
  Section V-D and is not deployed, because its cost is unpriced.
- **Separation is process-level on one host.** An attacker escaping the sandbox
  as the same user reaches the scheduler connection ungated. Closing this needs
  the mediation boundary to be an isolation boundary too; we specify but do not
  evaluate it.
- **Contracts are coverage-defined.** Except for citation binding, a claim
  outside declared coverage passes. We report effective coverage, not a
  guarantee.
- **No gate at B6 (instrument) or B7 (agent/federation).** No result is claimed
  for them.

Deploying PALISADE reduces an adversary's leverage in a content space where
search is cheap. It does not eliminate it.

## Reporting a vulnerability

For a vulnerability **in PALISADE itself** — a way past a gate that the paper's
threat model says should hold, a label the registry mishandles, a sink that
admits what its predicate should refuse — please report it privately rather
than opening a public issue. Contact the maintainers at the address on the
artifact landing page.

Findings that fall inside the stated limitations above are expected, not
vulnerabilities. A search that moves the propagation closure is the experiment
we ask for; please report it as a result, and we will cite it.

## Why the adaptive machinery is public

The trained attack policies, transform libraries and reward-ranked evasive
pools ship with everything else. Gating them was considered and dropped.

The argument for gating was that together they constitute a turnkey evasive
corpus. The argument against — which won — is that this project's own claim is
that a defense evaluated only against fixed attacks measures nothing about its
worst case. Withholding the attacks that move our numbers would make that claim
unfalsifiable by anyone but us, against a defense we are asking people to
trust. The operators are paraphrase and encoding transforms rather than novel
capability, and the adaptive work we position against released theirs.

What that decision does **not** cover: payloads remain detection-pattern
triggers with placeholder hosts, dual-use content remains surrogate-only, and
`tools/tests/test_release_safety.py` enforces the host half of that over every
public instance — the generated pools and the adversary modules included — on
every run.
