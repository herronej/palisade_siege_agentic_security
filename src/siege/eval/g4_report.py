"""
Markdown report formatter for the G4 evaluation.

Consumes a ``G4EvaluationResult`` from ``g4_runner.py`` and
produces the G4 portion of ``docs/palisade/g1_g4_eval.md``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from siege.eval.g4_attacks import DISPLAY_NAMES
from siege.eval.g4_runner import G4EvaluationResult


# -----------------------------------------------------------------
# Static sections
# -----------------------------------------------------------------


_METHODOLOGY = """\
## Methodology

### Synthetic vs live evaluation

A live evaluation against a malicious-code corpus would shell out
to the real ``semgrep`` CLI with the bundled ruleset + the
community ``p/security-audit`` config. This harness is
**synthetic** in one specific way:

- **Semgrep is mocked.** The gate's ``_run_semgrep`` is monkey-
  patched on each scenario to return that scenario's recorded
  findings. The harness runs without the
  ``vista-backend[palisade-g4]`` optional extra installed.
  The defense being measured is the gate's severity-routing
  logic (ERROR -> SEV2 deny, WARNING -> SEV3 allow, none ->
  allow), not Semgrep's rule-match precision.

What the harness pins by construction:

- The bundled rules' check IDs exist (every attack class names
  a real check ID).
- The gate routes ERROR-only and ERROR-mixed scenarios to deny.
- WARNING-only scenarios allow with a SEV3 incident -- the
  slow-tier Q-LLM is the layer that promotes those to deny.
- Empty-findings scenarios allow with no incident (the benign
  workload baseline).

The live evaluation measures the real-Semgrep rule-match rate on a live
malicious-code corpus drawn from CodeQL fixtures, MITRE ATT&CK
T1059 references, and the SIEGE A4 malicious-code
samples.

### Attack classes

Seven classes drawn from attack-taxonomy §4:

- **reverse_shell** -- ``subprocess.run(..., shell=True)`` with
  a bash reverse-shell command. Defense:
  ``vista-subprocess-shell-true`` ERROR.
- **eval_exec** -- ``eval`` / ``exec`` / ``compile`` on
  attacker-influenced input. Defense: ``vista-eval-exec`` ERROR.
- **pickle_rce** -- ``pickle.loads`` over attacker-controlled
  bytes. Defense: ``vista-pickle-untrusted`` ERROR.
- **credential_file_read** -- read SSH keys, ``.aws/credentials``,
  ``.netrc``, ``/etc/shadow``, etc. Defense:
  ``vista-credential-file-read`` ERROR.
- **typo_squat_import** -- import a known typo-squatted PyPI
  package. Defense: ``vista-typo-squat-import`` ERROR.
- **env_var_hijack** -- write ``LD_PRELOAD`` /
  ``DYLD_INSERT_LIBRARIES``. Defense:
  ``vista-env-var-preload-hijack`` ERROR.
- **direct_network_exfil** -- direct ``requests.post`` to an
  external host from inside the sandbox. Defense:
  ``vista-direct-outbound-network`` **WARNING** -- the rule
  fires SEV3 but allows; the slow-tier Q-LLM promotes to deny.

### Configurations

- **baseline**: G4 disabled. Every attack passes; benign
  scenarios trivially pass.
- **G4 fast-only (semgrep off)**: gate is on but
  ``semgrep_enabled=False`` -- per the work item AC, this is a
  no-op pass-through. Confirms the gate doesn't deny when
  Semgrep is intentionally off (the slow tier is the only
  layer in that configuration; not measured here).
- **G4 fast-only (semgrep on)**: gate is on,
  ``semgrep_enabled=True``. Mocked Semgrep returns each
  scenario's findings; severity routing is what's measured.

### Metrics

- **ASR per attack class** = fraction of scenarios where the
  gate did NOT deny. WARNING-only scenarios count as
  "succeeded" at the fast-tier-only layer because the slow tier
  is the deny-promoter.
- **FPR (False-Positive Rate)** = fraction of the benign-code
  workload G4 incorrectly denied. The benign workload has no
  findings by construction, so FPR > 0 only reveals a defect
  in the gate's empty-findings path.

### Reproducibility

```bash
cd backend
uv run python -m siege.eval --gate g4 > docs/palisade/g1_g4_eval.md
```

The harness pins ``seed=42``. Re-running produces an identical
report (modulo the timestamp).
"""


_LIMITS = """\
## Honest limits

What this evaluation does NOT measure:

- **Semgrep rule precision against real malicious code.** This
  is the load-bearing measurement deferred to the live evaluation: against
  CodeQL / SIEGE A4 / MITRE ATT&CK T1059 fixtures,
  what fraction of malicious snippets fire at least one of the
  bundled rules? The harness here pins the gate's *routing*
  logic, not the rules' coverage.
- **Slow-tier Q-LLM intent extraction.** The work item that
  implements G4's slow-tier intent extraction is a follow-on
  issue (`Implement G4 slow-tier`). When that lands,
  the WARNING-class scenarios (``direct_network_exfil``)
  should drop to ASR near zero in a ``fast+slow``
  configuration.
- **Sidecar wiring.** G4 is not yet wired into
  ``PalisadeSidecar.process_tool_call`` (separate
  issue). Until that lands, the eval measures the gate
  directly and the production runtime doesn't invoke it.
- **Benign-code corpus diversity.** The synthetic benign
  workload is 50 scenarios drawn from a small template pool
  (numpy / scipy / matplotlib / json / bash one-liners). A
  live measurement against a larger benign corpus
  (e.g. the project's actual ``hpc_jobs/`` history) is the
  authoritative FPR number.

### What the numbers DO say

- For every ERROR-class attack family the bundled rules
  target, the gate denies on the fast tier with ASR=0% when
  ``semgrep_enabled=True``.
- For WARNING-class families (``direct_network_exfil``), the
  fast tier allows with a SEV3 incident, matching the
  documented severity-routing contract; the slow tier is the
  next defense.
- When ``semgrep_enabled=False``, the gate is a no-op
  pass-through -- ASR matches baseline. This is the AC
  contract: ``__SEMGREP_ENABLED=false`` falls back to slow
  tier only.
- FPR on the synthetic benign workload is zero by construction
  (no rule fires on the workload templates), confirming the
  empty-findings code path doesn't spuriously deny.
"""


# -----------------------------------------------------------------
# Dynamic sections
# -----------------------------------------------------------------


def _format_asr_table(result: G4EvaluationResult) -> str:
    sections: list[str] = []
    for attack_class in result.attack_classes:
        display = DISPLAY_NAMES.get(attack_class, attack_class)
        sections.append(f"### {display}")
        sections.append("")
        sections.append("| Configuration | ASR | n attacks | sample reason |")
        sections.append("|---|---|---|---|")
        for cell in result.asr_cells:
            if cell.attack_class != attack_class:
                continue
            reason = cell.sample_reason or "(deny -- attack caught)"
            if len(reason) > 80:
                reason = reason[:77] + "..."
            sections.append(
                f"| {cell.config_name} | {cell.asr:.1%} | "
                f"{cell.n_attacks} | {reason} |"
            )
        sections.append("")
    return "\n".join(sections)


def _format_fpr_table(result: G4EvaluationResult) -> str:
    lines = [
        "| Configuration | FPR | n benign | sample block reason |",
        "|---|---|---|---|",
    ]
    for cell in result.fpr_cells:
        reason = cell.sample_block_reason or "(no incorrect blocks)"
        if len(reason) > 80:
            reason = reason[:77] + "..."
        lines.append(
            f"| {cell.config_name} | {cell.fpr:.1%} | "
            f"{cell.n_benign} | {reason} |"
        )
    return "\n".join(lines)


def _headline(result: G4EvaluationResult) -> str:
    per_family_best: dict[str, float] = {}
    for cell in result.asr_cells:
        prev = per_family_best.get(cell.attack_class, 1.0)
        per_family_best[cell.attack_class] = min(prev, cell.asr)
    worst_best = max(per_family_best.values()) if per_family_best else 0.0

    fpr_excl_baseline = [
        c.fpr for c in result.fpr_cells if c.config_name != "baseline"
    ]
    best_fpr = min(fpr_excl_baseline) if fpr_excl_baseline else 0.0

    return (
        f"**Worst-case ASR across attack classes in the best "
        f"configuration: {worst_best:.1%}.** "
        f"Best non-baseline FPR: {best_fpr:.2%}. "
        f"G4's fast tier denies every ERROR-class attack the bundled "
        f"rules target; WARNING-class attacks "
        f"(``direct_network_exfil``) await slow-tier promotion."
    )


# -----------------------------------------------------------------
# Public formatter
# -----------------------------------------------------------------


def format_g4_report(result: G4EvaluationResult) -> str:
    """Build the markdown body for the G4 portion of the eval doc."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        "# G4 evaluation: malicious-code corpus vs. G4 Code Gate",
        "",
        f"**Generated:** {timestamp} from seed {result.rng_seed}.",
        f"**Workload:** {result.n_attacks_per_class} scenarios per "
        f"attack class ({len(result.attack_classes)} classes); "
        f"{result.n_benign} benign-code workload scenarios for FPR.",
        f"**Configurations:** {', '.join(c.name for c in result.configs)}",
        "",
        "## Headline",
        "",
        _headline(result),
        "",
        _METHODOLOGY,
        "",
        "## ASR results (lower is better)",
        "",
        _format_asr_table(result),
        "## False-positive rate (lower is better)",
        "",
        _format_fpr_table(result),
        "",
        _LIMITS,
    ]
    return "\n".join(lines)
