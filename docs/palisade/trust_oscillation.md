<!-- Generated report. Source module: tools.trust_oscillation. Regenerate with `uv run python -m tools.trust_oscillation`. -->

# Sticky vs decaying trust -- probe-then-strike (E6)

The real `TrustScorer` under both variants, driven through a sweep of benign recovery windows. The strike targets the high-stakes `G5` capability, floored by a SEV1 incident; each window is a run of benign probe calls before the strike, testing whether trust recovers enough to re-open it.

| Variant | Probe queries before strike | Strike lands (ASR) | High-stakes capability retained after benign lull? |
|---|---|---|---|
| Sticky (PALISADE) | never | 0% (no) | **Yes** |
| Decaying-trust | 10 | 100% (yes) | No |

**C4 holds: no benign-probe timing re-opens the floored capability under the sticky scorer, so the probe-then-strike oscillation the decaying variant admits is unavailable against PALISADE.**

The sticky scorer removes the probe-then-strike oscillation available against decaying-trust designs: benign activity never lifts a high-stakes floor within the session (only an explicit re-authentication does), so an attacker cannot launder trust back up to land a strike.

## Per-window detail

Each row is a recovery-window length (benign probe calls); `strike lands` is whether the high-stakes capability is back at `NORMAL` after that lull. The decaying table's built-in "calibration bug" verdict is the test being *sensitive*: for the sticky deployment a landing strike would be a bug, which is exactly the decaying baseline's demonstrated vulnerability here.

### B2 trust-oscillation (sticky_high_stakes=True, cap=G5)

baseline (immediate strike) lands: **False**

| recovery window (clean calls) | strike lands |
|---|---|
| 0 | False |
| 1 | False |
| 3 | False |
| 10 | False |
| 50 | False |
| 200 | False |
| 1000 | False |

**C4 confirmed (no timing beats baseline)**

### B2 trust-oscillation (sticky_high_stakes=False, cap=G5)

baseline (immediate strike) lands: **False**

| recovery window (clean calls) | strike lands |
|---|---|
| 0 | False |
| 1 | False |
| 3 | False |
| 10 | True |
| 50 | True |
| 200 | True |
| 1000 | True |

**calibration bug surfaced (a recovery window lands the strike)**
