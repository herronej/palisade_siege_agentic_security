"""Optional gate instrumentation, inverted so the sidecar owns no host code.

A host application often wants per-gate timing — how long each tier costs on a
benign turn is the operating-point number PALISADE's own evaluation reports.
The obvious way to get it is to call the host's metrics recorder from inside
the gate dispatch, and that is exactly what a sidecar must not do: PALISADE is
installed *into* hosts, so an import pointing at one would make the package
uninstallable anywhere else.

So the dependency is inverted, the same way :mod:`palisade.host` inverts the
project and principal objects. PALISADE declares the shape it will call
(:class:`GateRecorder`), defaults to a no-op, and the host injects its own
recorder at startup:

    from palisade.instrumentation import set_gate_recorder
    set_gate_recorder(my_recorder)      # anything satisfying GateRecorder

The default recorder answers ``active()`` with ``False``, so an uninstrumented
deployment takes no clock readings and builds no payloads — the timing code
costs a single attribute lookup per tier. That matters because the thing being
measured is sub-millisecond: an instrument that cost as much as the check
would change the number it reports.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GateRecorder(Protocol):
    """What PALISADE calls to record a gate tier's timing.

    Structural: a host passes its own recorder unchanged, provided it carries
    these two methods. The signatures match the shape a metrics funnel
    normally already has, so a host usually needs no adapter.
    """

    def active(self, min_level: Any, override: str | None = None) -> bool:
        """Is this probe live? Called before any clock is read.

        Returning ``False`` must be cheap: it is on the path of every gate
        check, including in deployments that never enable instrumentation.
        """

    def gate(
        self,
        *,
        gate: str,
        tier: str,
        duration_ms: float,
        allow: bool,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        """Record one tier's wall time. ``tier`` is ``"fast"`` or ``"slow"``."""


class NullGateRecorder:
    """The default: never active, records nothing.

    Chosen over ``None`` so the call sites stay branch-free and read the same
    whether or not a host injected anything.
    """

    def active(self, min_level: Any, override: str | None = None) -> bool:
        return False

    def gate(
        self,
        *,
        gate: str,
        tier: str,
        duration_ms: float,
        allow: bool,
        payload: dict[str, Any] | None = None,
    ) -> None:
        return None


_recorder: GateRecorder = NullGateRecorder()


def set_gate_recorder(recorder: GateRecorder | None) -> None:
    """Install the host's recorder, or restore the no-op with ``None``.

    Idempotent and safe to call at startup. Passing ``None`` is the documented
    way for a test to undo an injection without reaching into module state.
    """
    global _recorder
    _recorder = NullGateRecorder() if recorder is None else recorder


def get_gate_recorder() -> GateRecorder:
    """The installed recorder, or the no-op default."""
    return _recorder


__all__ = [
    "GateRecorder",
    "NullGateRecorder",
    "set_gate_recorder",
    "get_gate_recorder",
]
