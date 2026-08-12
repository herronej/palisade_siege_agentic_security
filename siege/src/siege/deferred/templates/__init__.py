"""
Deferred template generators (parked, out of scope).

``b3_5_minja`` (multi-session memory poisoning) and ``b1_6_cui_extraction``
(CUI/sensitivity-tier extraction) are out of the v0.2 active corpus. They
are kept runnable here so the work can be revived once persistent
cross-session memory is added and the sensitivity tier is reinstated.
"""

from __future__ import annotations

from collections.abc import Callable

from siege.schemas import Instance
from siege.deferred.templates import b1_6_cui_extraction, b3_5_minja

#: Deferred template key -> builder.
DEFERRED_TEMPLATE_BUILDERS: dict[str, Callable[[], list[Instance]]] = {
    b1_6_cui_extraction.TEMPLATE: b1_6_cui_extraction.build,
    b3_5_minja.TEMPLATE: b3_5_minja.build,
}


__all__ = ["DEFERRED_TEMPLATE_BUILDERS"]
