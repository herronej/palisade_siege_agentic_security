<!-- Generated report. Source module: tools.utility_dos. Regenerate with `uv run python -m tools.utility_dos`. -->

# Utility / DoS dashboard (W23.5)

Operational framing of the defense's cost on legitimate work and the attack-side lockout dynamics. All measurements under the **deterministic fast-tier `full PALISADE`** config (trust scorer active, capability bound active, all gates live, no Q-LLM). CIs are exact-binomial (Clopper-Pearson) 95%.

## Benign side (W1 expanded control, n=181)

| Outcome | Count | Rate | 95% CI |
|---|---:|---:|---|
| Task completed | 173/181 | 95.6% | 91.5%–98.1% |
| Blocked — retryable (non-SEV1) | 1/181 | 0.6% | 0.0%–3.0% |
| Blocked — lockout (SEV1) | 7/181 | 3.9% | 1.6%–7.8% |

**Task completion = 95.6%** (= 1 − FPR). Of the 8 blocked benign tasks, **7** are SEV1 (lockout: the sticky trust scorer terminates the session until re-authentication) and **1** are lower-severity (retryable: the scientist rephrases or adjusts and resubmits without session state loss).

### Lockout signatures (benign SEV1)

| Signature | Count |
|---|---:|
| `G5 lifecycle-hook injection` | 5 |
| `G1 dual-use` | 2 |

These are the same conservative signatures the FPR analysis names as tunable policy knobs — tightening them raises attack coverage at the cost of more benign lockouts; loosening them does the reverse. The lockout rate moves with the same knobs that set the FPR.

## Attack side (SIEGE corpus, n=205)

| Metric | Count | Rate |
|---|---:|---:|
| Attack blocked (all attack actions denied) | 126/205 | 61.5% |
| Utility preserved (legitimate task allowed) | 205/205 | 100.0% |
| Utility lost (legitimate task also blocked) | 0/205 | 0.0% |
| Attack-induced lockout (SEV1 on attack) | 13/205 | 6.3% |
| Lockout + utility loss | 0/13 of lockouts | 0% |

**Attack-induced lockout: 13/205 = 6.3%** of attack instances trigger a SEV1 that would terminate the session. Of those lockouts, **0/13 = 0%** also block the instance's legitimate-task action — the DoS amplification: catching the attacker blocks legitimate work too until re-auth.

## Synthesis: the operational tradeoff

- **Benign lockout floor: 3.9%** (7/181) — the false-lockout rate the sticky mechanism imposes on clean traffic. This is the Table E6 number (7/181 = 3.9%).
- **Attack-induced lockout: 6.3%** (13/205) — the fraction of attacks that trigger a session-terminating SEV1.
- **Retries (non-lockout blocks): 0.6%** of benign tasks need a retry/adjustment (lower-severity blocks the scientist can recover from without re-auth).
- **Re-authentication cost:** every lockout (benign or attack-induced) requires an explicit re-auth to restore capabilities. The lockout is intentionally expensive (it removes the probe-then-strike oscillation window), but it means a benign false-SEV1 carries the same re-auth overhead as a real attack.

### DoS interpretation

An adversary who triggers a SEV1 denial forces a session lockout — an **intended** consequence of the sticky trust scorer (blocking the oscillation attack). This is not a denial-of-service *vulnerability*; it is the cost of the sticky-floor guarantee. The operator trades a 3.9% benign-lockout floor for the property that no amount of benign probing can re-open a breached high-stakes capability within a session. The lockout rate is tunable via the same gate signatures that set the FPR (Section W1).
