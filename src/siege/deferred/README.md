# SIEGE — deferred scope (out of scope for this release)

This directory holds SIEGE work that the **v0.2** scope defers out
of the active harness. Nothing here is on the default run path; it is kept
so the work is not lost when the deferred capabilities land.

| Piece | Why deferred | Revived by |
|---|---|---|
| `persistent_memory_store.py` (SQLite / Redis) | The active harness uses the in-process turn buffer (`siege/turn_buffer.py`); VISTA has no persistent cross-session memory yet. | Persistent cross-session memory (out of scope) |
| `templates/b3_5_minja.py` + `corpus/b3_5_minja/` | MINJA / MemoryGraft multi-session poisoning needs the persistent store above. | Persistent memory store (out of scope) |
| `templates/b1_6_cui_extraction.py` + `corpus/b1_6_cui_extraction/` | The deployed capability model carries all four axes `(source, taint, sensitivity, dual-use)` — `CapabilityTag.sensitivity` (`SensitivityTier` OPEN/INTERNAL/CUI/EXPORT_CONTROLLED) is live and G3 enforces per-KB tiers. What is deferred is this B1 CUI-*extraction* attack class, out of scope until the deployment ingests controlled data; the benchmark `CapabilitySpec` exercises the 3-axis `(source, taint, dual-use)` subset accordingly. | CUI / sensitivity-tier corpus (controlled-data deployment) |
| `docker/deferred/siege-memory/` | The Redis memory-store substrate container, removed from the active two-container set. | Persistent memory store (out of scope) |

To revive a template, move it back under `siege/templates/`, re-add
it to `templates/__init__.py:TEMPLATE_BUILDERS`, move its corpus back under
`siege/corpus/`, and (for B1.6) reinstate the `sensitivity` field on
`CapabilitySpec` and the instance JSON-Schema.
