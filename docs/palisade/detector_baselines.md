<!-- Generated report. Source module: tools.detector_baselines. Regenerate with `uv run python -m tools.detector_baselines`. -->

# Detection-only baselines: real Prompt-Guard-86M vs the regex denylist (W4.1)

Reviewer R1-M5 asked that the paper stop leading with a *reimplemented* detection-only baseline and report a **real** deployed detector. This runs Meta Prompt-Guard-86M (an mDeBERTa classifier over `{BENIGN, INJECTION, JAILBREAK}`) beside the 9-pattern regex denylist over the same 205-instance corpus, on the identical hard-win metric, each paired with its benign FPR on the W1 expanded control (181 tasks).

## Headline

| Defense | HW (all 205) | HW (security 150) | Benign FPR | 95% CI |
|---|---|---|---|---|
| Undefended | 18% | 21% | --- | --- |
| denylist-detector (regex reimplementation) | 14% | 16% | 0.0% | [0.0%, 2.0%] |
| prompt-guard (any non-BENIGN) (real model, max surface catch) | 2% | 3% | 66.9% | [59.5%, 73.7%] |
| prompt-guard (jailbreak-only) (real model, Meta low-FPR screen) | 16% | 19% | 7.7% | [4.3%, 12.6%] |
| **PALISADE (full)** | **2%** | **3%** | **4.4%** | [1.9%, 8.5%] |

A detection-only defense blocks an attack iff it flags the attack content; where it misses, the value reaches the sink as undefended (`baseline_hard_win = undefended_hard_win AND not flagged`). Benign FPR is the fraction of the 181 benign tasks the detector flags (would refuse) -- the same benign control the gate stack's 4.4% is measured on.

## Per detector

### `denylist-detector` (regex reimplementation)

- **Hard-win rate:** 18% undefended -> **14%** (all 205); 21% -> **16%** (150 security subset). PALISADE: 2% / 3%.
- **Surface catch:** 20% of all attacks; 22% of the undefended hard-win instances.
- **Benign FPR:** 0/181 = **0.0%** (95% CI [0.0%, 2.0%]) on the W1 control.

| Family | n | Undef HWR | Baseline HWR | PALISADE HWR | catch-on-HW |
|---|---|---|---|---|---|
| b1 | 45 | 11% | 0% | 0% | 100% |
| b3 | 38 | 21% | 21% | 0% | 0% |
| b4 | 45 | 11% | 9% | 0% | 20% |
| b5 | 55 | 9% | 9% | 0% | 0% |
| xc | 22 | 64% | 55% | 18% | 14% |

### `prompt-guard (any non-BENIGN)` (real model, max surface catch)

- **Hard-win rate:** 18% undefended -> **2%** (all 205); 21% -> **3%** (150 security subset). PALISADE: 2% / 3%.
- **Surface catch:** 80% of all attacks; 86% of the undefended hard-win instances.
- **Benign FPR:** 121/181 = **66.9%** (95% CI [59.5%, 73.7%]) on the W1 control.

| Family | n | Undef HWR | Baseline HWR | PALISADE HWR | catch-on-HW |
|---|---|---|---|---|---|
| b1 | 45 | 11% | 0% | 0% | 100% |
| b3 | 38 | 21% | 0% | 0% | 100% |
| b4 | 45 | 11% | 0% | 0% | 100% |
| b5 | 55 | 9% | 9% | 0% | 0% |
| xc | 22 | 64% | 0% | 18% | 100% |

### `prompt-guard (jailbreak-only)` (real model, Meta low-FPR screen)

- **Hard-win rate:** 18% undefended -> **16%** (all 205); 21% -> **19%** (150 security subset). PALISADE: 2% / 3%.
- **Surface catch:** 20% of all attacks; 11% of the undefended hard-win instances.
- **Benign FPR:** 14/181 = **7.7%** (95% CI [4.3%, 12.6%]) on the W1 control.

| Family | n | Undef HWR | Baseline HWR | PALISADE HWR | catch-on-HW |
|---|---|---|---|---|---|
| b1 | 45 | 11% | 11% | 0% | 0% |
| b3 | 38 | 21% | 21% | 0% | 0% |
| b4 | 45 | 11% | 11% | 0% | 0% |
| b5 | 55 | 9% | 9% | 0% | 0% |
| xc | 22 | 64% | 45% | 18% | 29% |

## Takeaway -- the content-detection trade-off is dominated by the capability bound

A content detector has one knob, its flag sensitivity, and it moves attack catch and benign refusal together:

- the **regex reimplementation** (the number the manuscript currently leads with) sits at ~14% hard-win / ~0% FPR -- it flags only unambiguous injection markers, sparing benign work but leaving most hard-wins standing;
- the **real Prompt-Guard-86M at full sensitivity** (any non-BENIGN label) reaches PALISADE's hard-win rate -- but only by flagging almost every imperative, benign or not, so its benign FPR is catastrophic;
- backed off to Meta's **jailbreak-only** low-FPR screen, its benign FPR falls to a deployable single-digit rate and its hard-win rate springs back up to roughly the regex baseline.

No posture of the content detector reaches PALISADE's corner: a low hard-win rate **and** a low benign FPR at once. The capability bound reaches it because it reads provenance, not content -- it blocks the tag-dropped value arriving at a privileged sink without touching the benign imperative that trips every content detector. That separation is the structural contribution, and it is unavailable to any defense that scores text.
