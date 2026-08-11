"""
PALISADE evaluation tools (WI20).

Gate-touching robustness tooling that must import the gate internals to test
them -- so it lives here, outside the ``redteam`` read-only guard, and only ever
*calls* the gates (never mutates them):

- ``fuzz_slurm_parser`` -- parser / IOC bypass stress (bypass rate).
- ``competitive_robustness`` -- the WI20 driver: external baseline, parser-fuzz,
  Family-A anomaly evasion (real G3 detector), replay-vs-live, held-out subset,
  and the G5/B5 concrete rows, assembled into the manuscript results doc.
"""
