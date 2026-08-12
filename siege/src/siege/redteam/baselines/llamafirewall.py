"""
Meta **LlamaFirewall** as a SIEGE detection baseline (reviewer item 11).

The panel already carries Prompt Guard, which is LlamaFirewall's *injection
scanner*, but not the system: LlamaFirewall composes six scanners and the paper
should not claim the system on the strength of one of them. This adapter runs
the real package (``llamafirewall``, Meta, Apache-2.0) through the same
``Detector`` protocol the other baselines implement, so it is scored on the
paper's own axes -- hard-win rate over the 205 attack instances and benign
false-positive rate over the 181-task control -- rather than on its own.

**Which scanners run.** Four of the six are local and are enabled by default:
``PROMPT_GUARD`` (the injection classifier), ``REGEX`` (its bundled pattern
rules), ``HIDDEN_ASCII`` (invisible-unicode smuggling), and ``CODE_SHIELD``
(static analysis of generated code, which is the scanner that overlaps our G4
Semgrep tier). ``AGENT_ALIGNMENT`` is excluded because it needs a served
model -- including it would make the comparator a model-backed defense and put
it on the wrong side of the fast/slow split this paper draws. ``PII_DETECTION``
is excluded because SIEGE scores privileged effects, not disclosure, so it can
only add false positives on the benign control. Both exclusions are stated in
the report so the comparator is not silently weakened.

**Decision mapping.** LlamaFirewall returns ALLOW, HUMAN_IN_THE_LOOP_REQUIRED,
or BLOCK. We count both non-ALLOW decisions as flagged, which is the reading
most favourable to the comparator: a deployment that escalates to a human has
not admitted the value, and scoring HITL as an allow would credit us with
closures LlamaFirewall arguably makes.

**Compatibility shim.** ``llamafirewall`` 1.0.3 imports ``HfFolder`` from
``huggingface_hub``, removed in 1.x. It is used only to test whether a token
exists before an optional ``login()``, so a minimal stand-in restores the
import without touching scan behaviour and without pinning the hub back for the
rest of the stack (transformers and the Prompt Guard 2 adapter both need it
current).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from siege.redteam.baselines.detectors import DetectorVerdict

#: Local scanners. AGENT_ALIGNMENT (served model) and PII_DETECTION (disclosure,
#: not a privileged effect) are deliberately out; see the module docstring.
DEFAULT_SCANNERS: tuple[str, ...] = (
    "PROMPT_GUARD",
    "REGEX",
    "HIDDEN_ASCII",
    "CODE_SHIELD",
)


def _install_hf_shim() -> None:
    """Restore ``huggingface_hub.HfFolder`` for ``llamafirewall`` 1.0.3.

    The scanner calls ``HfFolder.get_token()`` and, when it returns nothing,
    ``login()``. Returning the token the current API exposes preserves that
    behaviour exactly; when no token is configured we return None, which is the
    same answer the removed class gave.
    """
    import huggingface_hub as hub

    if hasattr(hub, "HfFolder"):
        return

    class HfFolder:  # noqa: D401 - a stand-in, not a new abstraction
        @staticmethod
        def get_token() -> str | None:
            try:
                return hub.get_token()
            except Exception:  # noqa: BLE001 - absence is a valid answer
                return None

    hub.HfFolder = HfFolder  # type: ignore[attr-defined]


@dataclass
class LlamaFirewallAdapter:
    """LlamaFirewall behind the ``Detector`` protocol."""

    scanners: tuple[str, ...] = DEFAULT_SCANNERS
    #: Which role the scanned text is presented as. USER is the injection-facing
    #: path; TOOL is what a tool-return scan would use.
    role: str = "USER"

    def __post_init__(self) -> None:
        self.name = "llamafirewall (" + "+".join(
            s.lower().replace("_", "-") for s in self.scanners
        ) + ")"
        self._fw: Any | None = None

    def available(self) -> bool:
        return self._build() is not None

    def _build(self) -> Any | None:
        if self._fw is not None:
            return self._fw
        try:
            _install_hf_shim()
            from llamafirewall import LlamaFirewall, Role, ScannerType

            self._fw = LlamaFirewall(
                scanners={
                    Role[self.role]: [ScannerType[s] for s in self.scanners]
                }
            )
        except Exception:  # noqa: BLE001 - an unavailable baseline is reported, not fatal
            self._fw = None
        return self._fw

    def flag(self, text: str) -> DetectorVerdict:
        fw = self._build()
        if fw is None:
            return DetectorVerdict(
                flagged=False, reason="llamafirewall unavailable", detector=self.name
            )
        from llamafirewall import Role, UserMessage

        try:
            result = fw.scan(UserMessage(content=text or ""))
        except Exception as exc:  # noqa: BLE001
            # A scanner error is not a block: counting it as one would inflate
            # the comparator the same way a fail-closed slow-tier error inflates
            # ours, and we report that hazard for our own numbers.
            return DetectorVerdict(
                flagged=False,
                reason=f"llamafirewall error: {type(exc).__name__}",
                detector=self.name,
            )
        decision = getattr(result.decision, "name", str(result.decision))
        return DetectorVerdict(
            flagged=decision != "ALLOW",
            reason=f"llamafirewall {decision} score={getattr(result, 'score', 0.0):.3f}",
            detector=self.name,
        )


__all__ = ["LlamaFirewallAdapter", "DEFAULT_SCANNERS"]
