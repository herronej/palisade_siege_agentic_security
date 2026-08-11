<!-- Generated report. Source module: tools.grounded_but_wrong. Regenerate with `uv run python -m tools.grounded_but_wrong`. -->

# Grounded-but-wrong: the reference-tolerance blind spot (W8.2)

The egress **grounding** predicate binds a reported value to a source retrieved this turn; it is *value-independent*, so a claim citing a real retrieved source (`10.1016/j.jnucmat.2013.04.030`) passes grounding at every perturbation below. The only value-accuracy check is then the subordinate reference-value contract (`data_value_mstdb_roundtrip`), which round-trips the number against the surrogate MSTDB-TP table and flags a deviation beyond the per-property tolerance. This driver sweeps each reference value through a signed relative perturbation and reports where the contract catches.

**Blind-spot bound.** A poisoned value that cites a real source and stays within the per-property tolerance passes every content contract. Across the 15 surrogate references whose true value the contracts admit, the tolerance -- and therefore the grounded-but-wrong half-width -- ranges from **5%** (density, melting point) to **25%** (viscosity). The contract catches at exactly its tolerance and nowhere tighter (crossover == tolerance for all 15: True); grounding holds at every perturbation (True), so it never supplies the catch.

## Per-reference blind spot

| salt | property | reference | tol | blind-spot half-width | worst grounded error | boundary catch |
|---|---|---:|---:|---:|---:|---|
| FLiBe | viscosity | 5.6 mPa.s | 25% | 24.5% | ±1.37 mPa.s | `data_value_mstdb_roundtrip` |
| FLiNaK | viscosity | 2.9 mPa.s | 20% | 19.5% | ±0.566 mPa.s | `data_value_mstdb_roundtrip` |
| FLiBe | thermal_conductivity | 1 W/(m.K) | 15% | 15% | ±0.15 W/(m.K) | `data_value_mstdb_roundtrip` |
| FLiNaK | thermal_conductivity | 0.92 W/(m.K) | 15% | 15% | ±0.138 W/(m.K) | `data_value_mstdb_roundtrip` |
| FLiBe | heat_capacity | 2386 J/(kg.K) | 10% | 10% | ±239 J/(kg.K) | `data_value_mstdb_roundtrip` |
| FLiNaK | heat_capacity | 1880 J/(kg.K) | 10% | 10% | ±188 J/(kg.K) | `data_value_mstdb_roundtrip` |
| KF-NaF-UF4 | heat_capacity | 1200 J/(kg.K) | 10% | 10% | ±120 J/(kg.K) | `data_value_mstdb_roundtrip` |
| LiF-BeF2 | thermal_conductivity | 1 W/(m.K) | 10% | 10% | ±0.1 W/(m.K) | `data_value_mstdb_roundtrip` |
| FLiBe | density | 1990 kg/m3 | 5% | 5% | ±99.5 kg/m3 | `data_value_mstdb_roundtrip` |
| FLiBe | melting_point | 732 K | 5% | 4.5% | ±32.9 K | `data_value_mstdb_roundtrip` |
| FLiNaK | density | 2034 kg/m3 | 5% | 4.5% | ±91.5 kg/m3 | `data_value_mstdb_roundtrip` |
| FLiNaK | melting_point | 727 K | 5% | 4.5% | ±32.7 K | `data_value_mstdb_roundtrip` |
| KF-NaF-UF4 | melting_point | 875 K | 5% | 5% | ±43.8 K | `data_value_mstdb_roundtrip` |
| LiF-BeF2 | density | 1990 kg/m3 | 5% | 5% | ±99.5 kg/m3 | `data_value_mstdb_roundtrip` |
| NaF-UF4 | melting_point | 910 K | 5% | 5% | ±45.5 K | `data_value_mstdb_roundtrip` |

The half-width is the largest single-sided perturbation that still passes; the worst grounded error is that half-width in physical units -- the maximum a design-consequential value can be corrupted while remaining grounded and contract-clean.

### References excluded (true value already flagged)

For these the surrogate reference itself is caught by a gross-plausibility bound, so no grounded-but-wrong band exists to measure (the gross bound would also flag the true value -- a coincidentally over-tight envelope, out of scope for the tolerance blind spot).

| salt | property | reference | flagged by |
|---|---|---:|---|
| NaF-UF4 | density | 4200 kg/m3 | `physical_bounds` |

## Pooled catch-rate curve

Fraction of the reference claims whose perturbed value is caught, by perturbation magnitude (pooled over both signs and all references). It rises from 0% inside every tolerance to 100% outside every tolerance; the transition band is the spread of per-property tolerances.

| \|perturbation\| | claim-perturbations | caught | catch rate |
|---|---:|---:|---:|
| 0–1% | 45 | 0 | 0% |
| 1–5% | 240 | 0 | 0% |
| 5–10% | 300 | 132 | 44% |
| 10–15% | 300 | 214 | 71% |
| 15–25% | 600 | 537 | 90% |
| 25–50% | 1500 | 1500 | 100% |
| 50–∞ | 630 | 630 | 100% |

The reference-value contract cannot detect a corruption smaller than its tolerance: that band is the grounded-but-wrong residual the structural guarantee does not close, because the guarantee promises grounding, not correctness. It is a coverage limitation of the subordinate detection layer, tunable by tightening the per-property tolerance at a false-positive cost on legitimately noisy references.
