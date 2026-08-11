"""
G4 Sandbox / Code Gate -- the single gate on the sandbox-code boundary (B4).

G4 fires on the tool calls that ship code or file writes into the agent
sandbox: ``run_bash`` (bash command body) and ``create_file`` (file path +
content, language inferred from the path extension). It runs three tiers,
cheapest first:

- **Tier 0 -- deterministic (always-on, no dependencies).** Path
  confinement for ``create_file`` (writes must resolve under the allowed
  sandbox roots) and execution-IOC scanning for ``run_bash``
  (reverse-shell / pipe-to-shell / crypto-miner / sensitive-credential
  reads), reusing G5's mining + sensitive-path denylists so the sandbox
  and HPC surfaces share one source of truth. A violation denies SEV2.
  This tier runs whenever the gate is enabled, independent of Semgrep --
  it is the deterministic floor, and it runs *before* the "nothing to
  scan" short-circuit so path confinement applies even to non-code files.

- **Tier 1 -- Semgrep (optional dependency).** When ``semgrep_enabled``
  and the ``semgrep`` CLI is available, the extracted code is scanned with
  the community ``p/security-audit`` ruleset + the PALISADE-specific
  bundle under ``palisade/contracts/semgrep/``. Any ERROR-severity
  finding -> SEV2 deny; WARNING-only -> SEV3 incident, allow; no findings
  -> allow.

- **Tier 2 -- Q-LLM code-intent (slow tier).** ``extract_code_intent``
  asks the quarantined LLM what the code is trying to do and compares it
  against G1's recorded user intent (dual-use / high-stakes-category
  mismatch). Shared with G5 (same code-intent agent).

History: this gate is the merge of the former G4 (Semgrep + Q-LLM) and the
former G7 (deterministic sandbox path/IOC) gates -- one boundary, one gate,
expressed with the fast/slow tier pattern every other PALISADE gate
uses. ``G4CodeGate`` remains as a backward-compatible alias.

## Per-tool overrides

Each tool's MCP descriptor can declare which rule IDs to disable
for code shipped to that tool. The format is a list of rule IDs
restricted to the ``vista-*`` namespace (community rules from
``p/security-audit`` can't be disabled per-tool, only globally).
At construction the gate accepts ``per_tool_overrides`` as a
``Mapping[str, Sequence[str]]``; the entries are validated lazily
when a tool is invoked. The lazy validation is the load-bearing
property of the Bell-LaPadula default-deny posture: an entry that
fails validation denies *that tool's call*, not the entire gate.

## Semgrep invocation

The gate shells out to the ``semgrep`` CLI via ``asyncio.subprocess``
so the gate stays compatible with the optional-dependency story --
deployments that haven't installed ``vista-backend[palisade-g4]``
should construct the gate fine but get a deterministic "semgrep
not available" path when ``semgrep_enabled=True``. The ``--json``
output is parsed into ``SemgrepFinding`` dataclasses; the gate
does NOT depend on Semgrep's internal Python API.

Failure modes are explicit:

- Semgrep configured but **not installed** (``FileNotFoundError``) ->
  degrade to the Tier-0 result (allow + SEV3 visibility). The
  deterministic floor already ran; an absent optional scanner must not
  fail-close the whole sandbox.
- Semgrep present but the scan **fails** (non-zero with no parseable
  JSON) -> deny SEV2. A present-but-unreliable scanner is not the same
  as an absent one.
- Semgrep timeout -> deny SEV2 (bounded at the capability hook layer).
- Semgrep produces a parseable response with zero findings -> allow.
- A Tier-0 path/IOC violation -> deny SEV2.

The defaults match the rest of PALISADE: when a *present* security
classifier can't reach a trustworthy verdict, default-deny rather than
fall through to allow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from palisade.capabilities import DualUseMarker
from palisade.quarantine import (
    HIGH_STAKES_CODE_CATEGORIES,
    CodeIntentExtraction,
    run_code_intent_extraction_with_self_consistency,
)
from palisade.gates.base import Gate, GateContext, GateDecision
from palisade.gates.denylists import (
    DEFAULT_MINING_BINARY_DENYLIST,
    DEFAULT_MINING_SIGNATURES,
    DEFAULT_SCIENTIFIC_IMPORT_ALLOWLIST,
    DEFAULT_SENSITIVE_PATH_PATTERNS,
    DEFAULT_TYPO_SQUAT_DENYLIST,
)


logger = logging.getLogger(__name__)


# Default low-confidence threshold below which the slow tier
# defaults to deny. Matches G1's intent-extraction default for
# consistency; operators with measured Q-LLM calibration data can
# lower it.
_DEFAULT_CODE_INTENT_CONFIDENCE_THRESHOLD: float = 0.5


# -----------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------


DEFAULT_SEMGREP_CONFIG: str = "p/security-audit"
"""
The community ruleset an operator can *opt into* (in addition to the
always-loaded PALISADE-bundled rules) by setting
``PalisadeSettings.semgrep_config`` / the gate's ``semgrep_config``.
It is **not** the default: a ``p/`` registry spec makes Semgrep fetch
from semgrep.dev, so it is unsuitable for air-gapped / CUI deployments.
The default (``semgrep_config=None``) runs the bundled rules only --
fully offline.
"""


_BUNDLED_RULES_SUBPATH = "contracts/semgrep"


def bundled_rules_dir() -> Path:
    """
    Return the absolute path to the package-bundled PALISADE
    Semgrep rules. 
    """
    return (Path(__file__).parent.parent / _BUNDLED_RULES_SUBPATH).resolve()


DEFAULT_SEMGREP_TIMEOUT_SECONDS: float = 30.0
"""
Per-invocation Semgrep timeout. A timed-out run defaults to deny
SEV2 (consistent with the rest of PALISADE's fail-closed
posture).
"""


# Language inference for `create_file`.
_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".js": "javascript",
    ".ts": "typescript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
}


_VALID_SEMGREP_SEVERITIES: frozenset[str] = frozenset(
    {"ERROR", "WARNING", "INFO"}
)


# -----------------------------------------------------------------
# Tier 0 -- deterministic sandbox checks (merged from the former G7)
# -----------------------------------------------------------------
#
# Dependency-free path-confinement + execution-IOC checks. They reuse G5's
# mining / sensitive-path denylists so the sandbox and HPC surfaces share
# one source of truth.

#: Directories the agent may write to via ``create_file``. ``uploads`` holds
#: user-provided + agent-authored data files; ``output`` holds generated
#: artifacts (plots, reports). Everything else under the ``/mnt`` mount is
#: off-limits.
DEFAULT_WRITE_ROOTS = ("/mnt/data/uploads", "/mnt/data/output")
#: Read-only subtrees the agent must never write to, even within the working
#: directory: its own skill code (``skills/``) and the RAG corpus / ChromaDB
#: store (``knowledge-bases/``, ``rag_db/``). A *relative* path like
#: ``skills/loader.py`` or ``data/knowledge-bases/.../chroma.sqlite3`` does not
#: escape the workdir with ``..``, so the upward-escape check misses it -- but
#: writing there is skill tampering / filesystem corpus poisoning (B4.4).
#: Matched as a path *segment* (so ``skills_summary.md`` is unaffected).
DEFAULT_PROTECTED_WRITE_SEGMENTS: frozenset[str] = frozenset(
    {"skills", "knowledge-bases", "rag_db"}
)
#: Tools whose path argument the Tier-0 check scopes.
WRITE_TOOLS = ("create_file",)
#: Tools whose command body the Tier-0 check scans for execution IOCs.
RUN_TOOLS = ("run_bash",)

# Reverse-shell and pipe-to-shell command IOCs. The sandbox has public
# network egress, so these are the remote-control / exfiltration surface.
_REVERSE_SHELL_RE = re.compile(
    r"/dev/(?:tcp|udp)/|\bnc(?:at|\.traditional)?\b[^\n;|&]*-e\b|"
    r"\bsocat\b[^\n]*\bexec|bash\s+-i\b[^\n]*/dev/tcp",
    re.IGNORECASE,
)
_PIPE_TO_SHELL_RE = re.compile(
    r"(?:curl|wget|fetch)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z)?sh\b|"
    r"\bbase64\s+-{1,2}d\w*\b[^\n|]*\|\s*(?:ba)?sh\b|"
    r"\beval\s+[\"']?\$\((?:curl|wget)",
    re.IGNORECASE,
)


def _normalize(path: str) -> str:
    return posixpath.normpath(path.strip())


def write_path_violation(
    path: Any, allowed_roots: tuple[str, ...] = DEFAULT_WRITE_ROOTS
) -> str | None:
    """Return a violation reason if ``path`` is not a confined sandbox write,
    else None.

    - Absolute paths must resolve under one of ``allowed_roots``.
    - Relative paths must not escape upward (``..``); they land in the
      sandbox working directory, which is inside the mount.
    """
    if not isinstance(path, str) or not path.strip():
        return "create_file called with an empty path"
    norm = _normalize(path)
    # Protected read-only subtrees (agent skills / RAG corpus store) -- denied
    # whether the path is relative or absolute, since a relative target inside
    # the workdir (skills/loader.py, data/knowledge-bases/...) does not trip the
    # upward-escape or sandbox-root checks below (B4.4).
    protected = next(
        (s for s in norm.split("/") if s in DEFAULT_PROTECTED_WRITE_SEGMENTS),
        None,
    )
    if protected is not None:
        return (
            f"create_file path {path!r} writes into the protected read-only "
            f"subtree {protected!r} (agent skills / RAG corpus store)"
        )
    if posixpath.isabs(norm):
        for root in allowed_roots:
            r = _normalize(root)
            if norm == r or norm.startswith(r + "/"):
                return None
        return (
            f"create_file path {path!r} is outside the writable sandbox roots "
            f"{list(allowed_roots)}"
        )
    # Relative path: reject upward escape; otherwise it stays in the workdir.
    if norm == ".." or norm.startswith("../"):
        return f"create_file relative path {path!r} escapes the working directory"
    return None


def run_bash_violation(command: Any) -> str | None:
    """Return a reason if ``command`` matches a sandbox-execution IOC, else
    None. Reuses G5's mining-binary / mining-signature / sensitive-path
    denylists so the sandbox and HPC surfaces share one source of truth."""
    if not isinstance(command, str) or not command.strip():
        return None
    low = command.lower()
    if _REVERSE_SHELL_RE.search(command):
        return (
            "run_bash command contains a reverse-shell pattern "
            "(/dev/tcp, nc -e, socat exec)"
        )
    if _PIPE_TO_SHELL_RE.search(command):
        return (
            "run_bash command pipes remote or decoded content into a shell "
            "(curl|bash, base64 -d|bash, eval $(curl ...))"
        )
    miner = {t for t in re.findall(r"[a-z0-9_.-]+", low)} & DEFAULT_MINING_BINARY_DENYLIST
    if miner:
        return f"run_bash command invokes a known crypto-miner ({sorted(miner)[0]!r})"
    for sig in DEFAULT_MINING_SIGNATURES:
        if sig in low:
            return f"run_bash command contains a crypto-mining signature ({sig!r})"
    for pat in DEFAULT_SENSITIVE_PATH_PATTERNS:
        if pat.lower() in low:
            return f"run_bash command reads a sensitive credential path ({pat!r})"
    return None


# Active / executable content smuggled into a *data* artifact (B4.6). A
# scientific datacard, report, or plot the agent writes (.md / .html / .svg /
# .json / .csv / ...) has no legitimate need for a <script> block, an inline
# event handler, a javascript: URI, a network fetch, or a cookie read -- those
# are an exfil / stored-XSS payload riding in a benign-looking output file.
# (Recognized *code* extensions are excluded here -- they go to the Semgrep
# tier, which has a real parser; this catches the non-code files Semgrep skips.)
_ACTIVE_CONTENT_RE = re.compile(
    r"<\s*script\b|"
    r"<\s*iframe\b|"
    r"\bon(?:load|error|click|mouseover|focus|submit|toggle|animationstart)"
    r"\s*=|"
    r"javascript:\s*\S|"
    r"document\s*\.\s*cookie|"
    r"\b(?:fetch|XMLHttpRequest|navigator\s*\.\s*sendBeacon)\s*\(",
    re.IGNORECASE,
)


def create_file_content_violation(path: Any, content: Any) -> str | None:
    """Return a reason if a ``create_file`` writes active/executable content
    into a non-code *data* artifact, else None.

    The Tier-0 path check confines *where* a file lands; this confines *what*
    a data file may contain. Code files (recognized extensions) are left to
    the Semgrep tier, which can parse them; this closes the gap where a
    ``<script>``/``onload=``/``fetch(evil)`` payload rides in a ``.md`` / ``.svg``
    / ``.html`` the scanner skips for lack of a parser (B4.6)."""
    if not isinstance(content, str) or not content.strip():
        return None
    ext = (
        "".join(Path(path).suffixes[-1:]).lower()
        if isinstance(path, str) and "." in path
        else ""
    )
    # Recognized code extension -> Semgrep tier owns the content scan.
    if ext in _LANGUAGE_BY_EXT:
        return None
    if _ACTIVE_CONTENT_RE.search(content):
        return (
            f"create_file writes active content (script / event-handler / "
            f"javascript: URI / network-fetch / cookie read) into the data "
            f"artifact {path!r}, which has no legitimate executable payload"
        )
    return None


# create_file should produce exactly one file (its own). Content that itself
# writes *executable shell scripts* is a file-writing fan-out scaffold -- the
# B4.6 multi_file_writer vector (a ``.py`` that loops ``open('a.sh','w')`` ...)
# -- which Semgrep's data-flow rules miss and which run_bash would later
# execute. A scientific analysis script has no reason to emit shell scripts.
_SHELL_SCRIPT_LITERAL_RE = re.compile(r"""['"][^'"\s]*\.(?:sh|bash|zsh)['"]""")
_FILE_WRITE_OP_RE = re.compile(
    r"\bopen\s*\([^)]*,\s*['\"][wa]|\.write_text\s*\(|\bwith\s+open\s*\(",
    re.IGNORECASE,
)


def create_file_fanout_violation(content: Any) -> str | None:
    """Return a reason if ``create_file`` content writes executable shell
    scripts (a file-writing fan-out), else None (B4.6 multi_file_writer)."""
    if not isinstance(content, str) or not content.strip():
        return None
    if _SHELL_SCRIPT_LITERAL_RE.search(content) and _FILE_WRITE_OP_RE.search(
        content
    ):
        return (
            "create_file content writes executable shell script(s) -- a "
            "file-writing fan-out, not a single artifact (create_file must "
            "produce exactly one file)"
        )
    return None


# Typo-squatted dependency imports (B4.3). Matched two ways: an exact hit on
# the known-squat denylist, and -- the generalization beyond a fixed list -- a
# single-edit (Damerau/OSA distance 1) near-miss of a legitimate scientific
# package that is not itself a known-good name. Top-level module only.
_IMPORT_RE = re.compile(r"(?m)^\s*(?:import|from)\s+([A-Za-z_][\w]*)")


def _osa_distance(a: str, b: str, *, cap: int = 2) -> int:
    """Optimal string alignment (restricted Damerau-Levenshtein) distance,
    short-circuiting above ``cap`` since the caller only cares about <= 1."""
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev2: list[int] | None = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if (
                prev2 is not None
                and i > 1
                and j > 1
                and a[i - 1] == b[j - 2]
                and a[i - 2] == b[j - 1]
            ):
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def import_typo_squat_violation(code: Any) -> str | None:
    """Return a reason if ``code`` imports a typo-squatted package, else None.

    Always-on (Tier-0): the deterministic floor catches the same squats as the
    Semgrep ``typo_squat.yml`` rule even when Semgrep is absent, and the
    near-miss heuristic generalizes past the fixed denylist."""
    if not isinstance(code, str) or not code.strip():
        return None
    for match in _IMPORT_RE.finditer(code):
        module = match.group(1).lower()
        if module in DEFAULT_TYPO_SQUAT_DENYLIST:
            return f"import of a known typo-squatted package ({module!r})"
        if module in DEFAULT_SCIENTIFIC_IMPORT_ALLOWLIST or len(module) < 5:
            continue
        for good in DEFAULT_SCIENTIFIC_IMPORT_ALLOWLIST:
            if len(good) >= 4 and _osa_distance(module, good) == 1:
                return (
                    f"import {module!r} is a single-character near-miss of the "
                    f"scientific package {good!r} (typo-squat)"
                )
    return None


# -----------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------


@dataclass(frozen=True)
class SemgrepFinding:
    """
    A single Semgrep finding, normalized into the fields G4 actually
    consumes. The full Semgrep result dict is preserved in
    ``extra`` for downstream diagnostics; the typed fields are the
    contract gates and tests pin against.
    """

    check_id: str
    severity: str
    message: str
    start_line: int
    end_line: int
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedCode:
    """
    Result of ``_extract_code(payload)``: the code blob about to be
    scanned plus the language Semgrep should be told to parse it as.
    """

    scan: bool
    code: str = ""
    language: str = ""
    reason: str = ""


# -----------------------------------------------------------------
# G4CodeGate
# -----------------------------------------------------------------


class G4SandboxCodeGate(Gate):
    """
    G4 Sandbox / Code Gate.

    The single gate on the B4 sandbox-code boundary. ``check_fast`` runs
    Tier 0 (deterministic path-confinement + execution IOCs, always-on)
    then Tier 1 (Semgrep, optional dependency, degrades to Tier 0 when not
    installed); ``check_slow`` runs Tier 2 (Q-LLM code-intent). Subsumes
    the former G7 sandbox gate; ``G4CodeGate`` is a back-compat alias.
    """

    name = "G4"

    def __init__(
        self,
        *,
        enabled: bool = True,
        semgrep_enabled: bool = False,
        semgrep_config: str | None = None,
        bundled_rules_dir: Path | None = None,
        per_tool_overrides: Mapping[str, Sequence[str]] | None = None,
        semgrep_timeout: float = DEFAULT_SEMGREP_TIMEOUT_SECONDS,
        semgrep_executable: str = "semgrep",
        scan_tools: frozenset[str] = frozenset(
            {"run_bash", "create_file"}
        ),
        allowed_write_roots: tuple[str, ...] = DEFAULT_WRITE_ROOTS,
        code_intent_extraction_agent: Any | None = None,
        code_intent_confidence_threshold: float = _DEFAULT_CODE_INTENT_CONFIDENCE_THRESHOLD,
        code_intent_self_consistency_samples: int = 1,
        high_stakes_code_categories: frozenset[str] | None = None,
        slow_tier_fail_open: bool = False,
    ) -> None:
        super().__init__(enabled=enabled)
        self._semgrep_enabled = semgrep_enabled
        self._semgrep_config = semgrep_config
        self._bundled_rules_dir = (
            bundled_rules_dir
            if bundled_rules_dir is not None
            else globals()["bundled_rules_dir"]()
        )
        self._per_tool_overrides: dict[str, Any] = dict(
            per_tool_overrides or {}
        )
        self._semgrep_timeout = semgrep_timeout
        self._semgrep_executable = semgrep_executable
        self._scan_tools: frozenset[str] = scan_tools
        # Tier-0 (deterministic sandbox) write-root confinement.
        self._roots: tuple[str, ...] = tuple(allowed_write_roots)

        # Slow-tier configuration. The agent is supplied at
        # construction by the sidecar 
        self._code_intent_extraction_agent: Any | None = (
            code_intent_extraction_agent
        )
        self._code_intent_confidence_threshold = code_intent_confidence_threshold
        self._code_intent_self_consistency_samples = max(
            1, int(code_intent_self_consistency_samples)
        )
        # Low-confidence (uncertain) slow-tier verdict -> SEV3 advisory allow
        # (defer to Tier-0/Semgrep + capability bound, §8.2) when True; the §C8
        # benign-FP posture. Confident mismatches still deny. Default False.
        self._slow_tier_fail_open = bool(slow_tier_fail_open)
        self._high_stakes_code_categories: frozenset[str] = (
            high_stakes_code_categories
            if high_stakes_code_categories is not None
            else HIGH_STAKES_CODE_CATEGORIES
        )

    # -----------------------------------------------------------------
    # Read-only views
    # -----------------------------------------------------------------

    @property
    def semgrep_enabled(self) -> bool:
        return self._semgrep_enabled

    @property
    def semgrep_config(self) -> str | None:
        return self._semgrep_config

    @property
    def bundled_rules_dir(self) -> Path:
        return self._bundled_rules_dir

    @property
    def semgrep_timeout(self) -> float:
        return self._semgrep_timeout

    @property
    def scan_tools(self) -> frozenset[str]:
        return self._scan_tools

    @property
    def allowed_write_roots(self) -> tuple[str, ...]:
        return self._roots

    @property
    def write_tools(self) -> tuple[str, ...]:
        return WRITE_TOOLS

    @property
    def run_tools(self) -> tuple[str, ...]:
        return RUN_TOOLS

    def check_write_path(self, path: Any) -> str | None:
        """Tier-0 ``create_file`` path-scope check (pure, deterministic)."""
        return write_path_violation(path, self._roots)

    def check_run_bash(self, command: Any) -> str | None:
        """Tier-0 ``run_bash`` execution-IOC check (pure, deterministic)."""
        return run_bash_violation(command)

    def _check_deterministic(
        self, tool_name: str, args: Mapping[str, Any]
    ) -> str | None:
        """Tier 0: deterministic path/IOC check. Returns a violation
        reason, or None when the call is within sandbox policy."""
        if tool_name in WRITE_TOOLS:
            path, content = args.get("path"), args.get("content")
            return (
                self.check_write_path(path)
                or create_file_content_violation(path, content)
                or create_file_fanout_violation(content)
                or import_typo_squat_violation(content)
            )
        if tool_name in RUN_TOOLS:
            command = args.get("command")
            return self.check_run_bash(command) or import_typo_squat_violation(
                command
            )
        return None

    @property
    def code_intent_extraction_agent(self) -> Any | None:
        return self._code_intent_extraction_agent

    @property
    def code_intent_confidence_threshold(self) -> float:
        return self._code_intent_confidence_threshold

    @property
    def high_stakes_code_categories(self) -> frozenset[str]:
        return self._high_stakes_code_categories

    def attach_code_intent_extraction_agent(self, agent: Any) -> None:
        """
        Store the PydanticAI code-intent Agent for the slow-tier path.
        """
        self._code_intent_extraction_agent = agent

    # -----------------------------------------------------------------
    # Fast-tier check
    # -----------------------------------------------------------------

    async def _check_fast_when_enabled(
        self,
        payload: Any,
        ctx: GateContext,
    ) -> GateDecision:
        """
        Extract code from the payload, run Semgrep, route by
        finding severity.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                f"G4CodeGate expected payload dict with 'tool_name'/'args', "
                f"got {type(payload).__name__}"
            )
        tool_name = payload.get("tool_name")
        args = payload.get("args")
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            raise TypeError(
                "G4CodeGate payload must be "
                "{'tool_name': str, 'args': dict}; got "
                f"tool_name={type(tool_name).__name__}, "
                f"args={type(args).__name__}"
            )

        # Out-of-scope tool -> allow without scanning. G4 is only
        # interested in code-bearing / sandbox tools.
        if tool_name not in self._scan_tools:
            return GateDecision(
                allow=True,
                reason=f"G4: tool {tool_name!r} not in scan_tools (skip)",
            )

        # ----- Tier 0: deterministic path/IOC checks (always-on) -----
        # Runs regardless of Semgrep availability, and *before* the
        # "nothing to scan" short-circuit below, so create_file path
        # confinement applies even to files with no recognized code
        # extension. This is the deterministic floor of the gate.
        tier0_violation = self._check_deterministic(tool_name, args)
        if tier0_violation is not None:
            return GateDecision(
                allow=False,
                reason=f"G4 Tier-0: {tier0_violation}",
                incident_level=2,
            )

        # Per-tool override (Bell-LaPadula: malformed entry denies).
        try:
            disabled = self._disabled_rules_for_tool(tool_name)
        except ValueError as exc:
            return GateDecision(
                allow=False,
                reason=(
                    f"G4 malformed per-tool override for {tool_name!r}: "
                    f"{exc}"
                ),
                incident_level=2,
            )

        extracted = _extract_code(tool_name, args)
        if not extracted.scan or not extracted.code.strip():
            return GateDecision(
                allow=True,
                reason=(
                    f"G4: nothing to scan on call to {tool_name!r} "
                    f"({extracted.reason or 'empty code'})"
                ),
            )

        # ----- Tier 1: Semgrep (optional dependency) -----
        # Tier 0 already passed. When Semgrep is disabled, that
        # deterministic floor is the whole fast tier -> allow.
        if not self._semgrep_enabled:
            return GateDecision(
                allow=True,
                reason=(
                    f"G4: deterministic tier only "
                    f"(semgrep_enabled=False) on {tool_name!r}"
                ),
            )

        try:
            findings = await self._run_semgrep(
                extracted.code, extracted.language
            )
        except FileNotFoundError as exc:
            # Semgrep configured but not installed: DEGRADE to the Tier-0
            # result (which already passed) rather than fail-closed. The
            # deterministic floor still ran; surface a SEV3 so the gap is
            # never silent.
            logger.warning(
                "PALISADE G4: %s; running the deterministic tier only", exc
            )
            return GateDecision(
                allow=True,
                reason=(
                    f"G4: semgrep configured but not installed; "
                    f"deterministic-tier-only on {tool_name!r}"
                ),
                incident_level=3,
            )
        except Exception as exc:
            # Semgrep present but the scan failed (unparseable output,
            # etc.): a present-but-unreliable scanner is not the same as an
            # absent one -> keep the fail-closed default-deny.
            logger.warning(
                "PALISADE G4: semgrep invocation failed (%s: %s); "
                "default-deny",
                type(exc).__name__, exc,
            )
            return GateDecision(
                allow=False,
                reason=(
                    f"G4: semgrep invocation failed "
                    f"({type(exc).__name__}: {exc}); default-deny"
                ),
                incident_level=2,
            )

        # Filter out disabled rules.
        if disabled:
            findings = [
                f for f in findings if f.check_id not in disabled
            ]

        return self._decide_from_findings(tool_name, findings)

    # -----------------------------------------------------------------
    # Severity routing
    # -----------------------------------------------------------------

    def _decide_from_findings(
        self,
        tool_name: str,
        findings: list[SemgrepFinding],
    ) -> GateDecision:
        """
        Route Semgrep findings into a single ``GateDecision``.

        Any ERROR finding denies SEV2 (the gate names the
        highest-severity rule plus the count of additional rules so
        the structured log lets ops triage without re-running
        Semgrep). WARNING findings allow but record a SEV3 incident
        with the same shape.
        """
        errors = [f for f in findings if f.severity == "ERROR"]
        warnings = [f for f in findings if f.severity == "WARNING"]

        if errors:
            primary = errors[0]
            other_ids = sorted({f.check_id for f in errors[1:]})
            reason_extra = (
                f" (+{len(other_ids)} more: {other_ids[:4]})"
                if other_ids else ""
            )
            return GateDecision(
                allow=False,
                reason=(
                    f"G4 semgrep ERROR on {tool_name!r}: "
                    f"{primary.check_id} at L{primary.start_line}: "
                    f"{primary.message[:120]}{reason_extra}"
                ),
                incident_level=2,
            )
        if warnings:
            primary = warnings[0]
            other_ids = sorted({f.check_id for f in warnings[1:]})
            reason_extra = (
                f" (+{len(other_ids)} more: {other_ids[:4]})"
                if other_ids else ""
            )
            return GateDecision(
                allow=True,
                reason=(
                    f"G4 semgrep WARNING on {tool_name!r}: "
                    f"{primary.check_id} at L{primary.start_line}: "
                    f"{primary.message[:120]}{reason_extra}"
                ),
                incident_level=3,
            )
        return GateDecision(
            allow=True,
            reason=f"G4 semgrep ok on {tool_name!r}: 0 findings",
        )

    # -----------------------------------------------------------------
    # Per-tool override resolution
    # -----------------------------------------------------------------

    def disabled_rules_for_tool(
        self, tool_name: str
    ) -> frozenset[str] | None:
        """
        Public accessor: resolve the disabled rule IDs for
        ``tool_name``, raising ``ValueError`` on a malformed override.

        `G4CodeCapability.prepare_tools` calls this to enforce the
        Bell-LaPadula default-deny on a per-tool basis (a malformed
        override drops only that tool from the toolset).
        """
        return self._disabled_rules_for_tool(tool_name)

    def _disabled_rules_for_tool(
        self, tool_name: str
    ) -> frozenset[str] | None:
        """
        Return the set of rule IDs disabled for ``tool_name``, or
        ``None`` if no override exists.
        """
        raw = self._per_tool_overrides.get(tool_name)
        if raw is None:
            return None
        if isinstance(raw, (str, bytes)):
            raise ValueError(
                f"override must be a sequence of rule IDs, "
                f"not {type(raw).__name__}"
            )
        if not isinstance(raw, (list, tuple, set, frozenset)):
            raise ValueError(
                f"override must be a list/set; got {type(raw).__name__}"
            )
        rules: set[str] = set()
        for entry in raw:
            if not isinstance(entry, str):
                raise ValueError(
                    f"override entries must be strings; got "
                    f"{type(entry).__name__}"
                )
            if not entry.startswith("vista-"):
                raise ValueError(
                    f"override rule id {entry!r} must be in the "
                    f"`vista-*` namespace (community rules cannot be "
                    f"disabled per-tool)"
                )
            rules.add(entry)
        return frozenset(rules)

    # -----------------------------------------------------------------
    # Slow-tier: Q-LLM code-intent extraction
    # -----------------------------------------------------------------

    async def extract_code_intent(
        self,
        payload: Any,
        ctx: GateContext,
        decision: GateDecision,
    ) -> GateDecision:
        """
        G4 slow-tier: ask the Q-LLM what the code is trying to do,
        and compare against G1's recorded user-intent.

        Behavior:

        - **Pass-through** when the gate is disabled, the
          ``ctx.quarantine_agent`` is None (operator did not enable
          slow tier), the code-intent agent isn't attached, the
          fast-tier already denied (``decision.allow=False``), or
          the payload has no scannable code.
        - **Low-confidence intent** -> SEV2 default-deny. A Q-LLM
          that cannot reach a coherent intent on emitted code is
          the load-bearing signal for an obfuscated payload.
        - **Dual-use mismatch** -> SEV2. The code's
          ``dual_use_flag`` is non-NONE and the user prompt's G1
          intent did not declare the same marker.
        - **High-stakes-category mismatch** -> SEV2. The code
          performs operations in
          ``high_stakes_code_categories`` (default:
          ``credential_access`` / ``data_exfiltration``) and the
          user prompt's G1 intent does not lexically mention that
          category.
        - **Match** -> allow with the existing decision augmented
          to carry the extracted intent for downstream provenance.

        Why a separate ``extract_code_intent`` (rather than relying
        only on ``_check_slow_when_enabled``): like G1's
        ``extract_intent``, this is the public entrypoint the
        sidecar will call directly when wiring G4 into
        ``process_tool_call``. The base ``Gate.check_slow``
        dispatch (which guards on ``ctx.quarantine_agent``) still
        works because ``_check_slow_when_enabled`` delegates here.
        """
        if (
            not self.enabled
            or self._code_intent_extraction_agent is None
            or not decision.allow
        ):
            return decision

        if not isinstance(payload, dict):
            return decision
        tool_name = payload.get("tool_name")
        args = payload.get("args")
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            return decision

        # Only code-bearing tools get scanned. Other tools fall
        # through with the input decision unchanged.
        if tool_name not in self._scan_tools:
            return decision

        extracted = _extract_code(tool_name, args)
        if not extracted.scan or not extracted.code.strip():
            return decision

        prompt = _CODE_INTENT_PROMPT_TEMPLATE.format(
            tool_name=tool_name,
            language=extracted.language,
            code=extracted.code,
        )

        intent: CodeIntentExtraction = (
            await run_code_intent_extraction_with_self_consistency(
                self._code_intent_extraction_agent,
                prompt,
                samples=self._code_intent_self_consistency_samples,
            )
        )

        # ----- Low-confidence path -> advisory allow (fail-open) or deny -----
        if intent.confidence < self._code_intent_confidence_threshold:
            if self._slow_tier_fail_open:
                # Uncertain advisory verdict -> defer to the deterministic
                # Tier-0/Semgrep + capability bound (§8.2); SEV3 advisory, allow
                # (the §C8 ≤2% benign-FP posture). A confident mismatch below
                # still denies.
                return decision.replace_with(
                    reason=(
                        f"{decision.reason}; G4 slow-tier uncertain (confidence "
                        f"{intent.confidence:.2f} < "
                        f"{self._code_intent_confidence_threshold:.2f} on "
                        f"{tool_name!r}); fail-open advisory, deferred to the "
                        f"deterministic layer"
                    ),
                    incident_level=3,
                )
            return decision.replace_with(
                allow=False,
                reason=(
                    f"G4 slow-tier: code-intent confidence "
                    f"{intent.confidence:.2f} below threshold "
                    f"{self._code_intent_confidence_threshold:.2f} on "
                    f"{tool_name!r}; default-deny"
                ),
                incident_level=2,
            )

        # ----- Read G1's recorded user intent --------------------
        g1_summary, g1_dual_use = _read_g1_intent(ctx)

        # ----- Mismatch detection -------------------------------
        mismatch, mismatch_reason = _detect_intent_mismatch(
            code_intent=intent,
            g1_intent_summary=g1_summary,
            g1_dual_use=g1_dual_use,
            high_stakes_categories=self._high_stakes_code_categories,
        )
        if mismatch:
            return decision.replace_with(
                allow=False,
                reason=(
                    f"G4 slow-tier: code-intent mismatch on {tool_name!r}: "
                    f"{mismatch_reason} (code_intent="
                    f"{intent.intent_summary!r})"
                ),
                incident_level=2,
            )

        # ----- Clean intent -> allow with annotated reason ------
        category_text = ",".join(sorted(intent.categories)) or "(none)"
        return decision.replace_with(
            reason=(
                f"{decision.reason}; G4 slow-tier ok "
                f"(categories={category_text}, "
                f"dual_use={intent.dual_use_flag.value}, "
                f"confidence={intent.confidence:.2f})"
            ),
        )

    async def _check_slow_when_enabled(
        self,
        payload: Any,
        ctx: GateContext,
        decision: GateDecision,
    ) -> GateDecision:
        """
        Delegate to ``extract_code_intent`` so callers using the
        base ``Gate.check_slow`` dispatch get the same behavior as
        callers invoking ``extract_code_intent`` directly.
        """
        return await self.extract_code_intent(payload, ctx, decision)

    # -----------------------------------------------------------------
    # Semgrep invocation
    # -----------------------------------------------------------------

    async def _run_semgrep(
        self, code: str, language: str
    ) -> list[SemgrepFinding]:
        """
        Invoke the semgrep CLI on ``code`` and parse the JSON
        output into ``SemgrepFinding`` objects.
        """
        executable = shutil.which(self._semgrep_executable)
        if executable is None:
            raise FileNotFoundError(
                f"semgrep executable {self._semgrep_executable!r} not "
                f"found on PATH; install with "
                f"`pip install vista-backend[palisade-g4]`"
            )

        # Write the code to a temp file so Semgrep can pick up the
        # language from the suffix 
        suffix = _suffix_for_language(language)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name
        try:
            cmd = [
                executable,
                "--quiet",
                "--json",
                "--no-git-ignore",
            ]
            # Optional opt-in ruleset (e.g. a `p/` registry pack) layered on
            # first; None -> bundled rules only, so the scan stays offline.
            if self._semgrep_config:
                cmd += ["--config", self._semgrep_config]
            cmd += [
                "--config", str(self._bundled_rules_dir),
                tmp_path,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # No internal timeout here: G4CodeCapability bounds the
            # whole fast check with `anyio.fail_after(semgrep_timeout)`
            # at the hook layer (the same primitive PydanticAI uses for
            # native hook timeouts), so the deadline lives with the hook
            # rather than buried in the runner. A timeout cancels this
            # await; the capability converts it to a default-deny SEV2.
            stdout_b, stderr_b = await proc.communicate()

            # Semgrep exits non-zero when it has findings, so the
            # return code is not a reliable success signal. The
            # contract is: stdout is parseable JSON.
            try:
                payload = json.loads(stdout_b.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as exc:
                stderr_text = stderr_b.decode("utf-8", errors="replace")[:400]
                raise RuntimeError(
                    f"semgrep produced unparseable output "
                    f"(rc={proc.returncode}): {exc}; stderr={stderr_text!r}"
                ) from exc

            return _parse_semgrep_results(payload)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# -----------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Slow-tier prompt template + helpers
# -----------------------------------------------------------------


_CODE_INTENT_PROMPT_TEMPLATE = """\
Tool: {tool_name}
Language: {language}
Code:
---
{code}
---
"""


def _read_g1_intent(
    ctx: GateContext,
) -> tuple[str, DualUseMarker]:
    """
    Read G1's recorded user intent from the capability registry.

    Returns ``(intent_summary, dual_use_flag)`` extracted from the
    ``"user:prompt"`` tag's metadata. When the tag is absent or
    the fields are missing, returns ``("", DualUseMarker.NONE)`` so
    the comparator treats the user-intent side as "no information."

    No information on the user-intent side is *not* automatically a
    mismatch: a deployment that hasn't enabled G1 slow tier still
    needs to allow benign code to run. The comparator's mismatch
    decision is between "code declares high-stakes categories" and
    "user-intent declares them too" -- if both sides are empty,
    that's a match.
    """
    tag = ctx.capability_registry.get("user:prompt")
    if tag is None:
        return "", DualUseMarker.NONE

    metadata = tag.metadata or {}
    summary = str(metadata.get("intent_summary", "") or "")

    raw_flag = metadata.get("intent_dual_use_flag")
    if isinstance(raw_flag, str):
        try:
            dual_use = DualUseMarker(raw_flag.lower())
        except ValueError:
            dual_use = DualUseMarker.NONE
    elif isinstance(raw_flag, DualUseMarker):
        dual_use = raw_flag
    else:
        # Fall back to the tag's own `dual_use` field, which G1's
        # slow tier copies into the tag when it updates.
        dual_use = tag.dual_use

    return summary, dual_use


def _detect_intent_mismatch(
    *,
    code_intent: CodeIntentExtraction,
    g1_intent_summary: str,
    g1_dual_use: DualUseMarker,
    high_stakes_categories: frozenset[str],
) -> tuple[bool, str]:
    """
    Apply the G4 intent-mismatch comparator.

    Returns ``(is_mismatch, reason)``. Two routes to a mismatch:

    1. **Dual-use mismatch.** The code's ``dual_use_flag`` is
       non-NONE and G1's recorded dual-use marker either differs
       or is NONE. Different dual-use markers are mismatches
       regardless of which side is "more restrictive" -- the
       semantic check is "does the user's stated intent match the
       code's apparent intent," not a lattice join.
    2. **High-stakes-category mismatch.** The code claims a
       high-stakes category (``credential_access`` /
       ``data_exfiltration`` by default) and G1's recorded intent
       summary does not lexically mention the category.

    The lexical-mention check is intentionally conservative. We
    use the category name with underscores stripped (e.g.,
    ``"credential access"``) and look for that token in the lower-
    cased user-intent summary. A user prompt like "summarize the
    paper" doesn't mention credentials, so credential-access code
    triggers a mismatch. A prompt like "extract credentials from
    the auth log file and summarize" does mention credentials and
    passes -- the deterministic lexical check accepts the user's
    declared intent at face value, on the assumption that an
    attacker capable of jailbreaking past G1's regex *and* the
    Q-LLM's intent extraction has already lost a layer of defense.
    """
    # ----- Dual-use comparison ----------------------------------
    if code_intent.dual_use_flag is not DualUseMarker.NONE:
        if g1_dual_use is DualUseMarker.NONE:
            return True, (
                f"code dual_use={code_intent.dual_use_flag.value} but "
                f"user intent declared no dual-use marker"
            )
        if code_intent.dual_use_flag is not g1_dual_use:
            return True, (
                f"code dual_use={code_intent.dual_use_flag.value} "
                f"differs from user dual_use={g1_dual_use.value}"
            )

    # ----- High-stakes-category comparison ----------------------
    code_categories = set(code_intent.categories)
    flagged = high_stakes_categories & code_categories
    if not flagged:
        return False, ""

    user_text = g1_intent_summary.lower()
    undeclared = []
    for category in sorted(flagged):
        # Match the category as a phrase (e.g. "credential access").
        # A future enhancement: regex with word boundaries, or an
        # operator-supplied per-category vocabulary.
        token = category.replace("_", " ")
        if token in user_text:
            continue
        undeclared.append(category)

    if undeclared:
        return True, (
            f"code performs high-stakes operation(s) {undeclared} "
            f"not declared in user intent {g1_intent_summary!r}"
        )

    return False, ""


def _extract_code(
    tool_name: str, args: Mapping[str, Any]
) -> ExtractedCode:
    """
    Extract the code blob + language from a code-bearing tool call.
    """
    if tool_name == "run_bash":
        command = args.get("command")
        if not isinstance(command, str):
            return ExtractedCode(
                scan=False,
                reason=f"run_bash args lack a string `command`",
            )
        return ExtractedCode(scan=True, code=command, language="bash")

    if tool_name == "create_file":
        content = args.get("content")
        path = args.get("path")
        if not isinstance(content, str):
            return ExtractedCode(
                scan=False,
                reason=f"create_file args lack a string `content`",
            )
        if not isinstance(path, str):
            return ExtractedCode(
                scan=False,
                reason=f"create_file args lack a string `path`",
            )
        ext = "".join(Path(path).suffixes[-1:]).lower() if "." in path else ""
        language = _LANGUAGE_BY_EXT.get(ext)
        if language is None:
            return ExtractedCode(
                scan=False,
                reason=(
                    f"create_file path {path!r} has unrecognized "
                    f"extension {ext!r}; no Semgrep parser to use"
                ),
            )
        return ExtractedCode(scan=True, code=content, language=language)

    return ExtractedCode(
        scan=False,
        reason=f"tool {tool_name!r} not code-bearing",
    )


def _suffix_for_language(language: str) -> str:
    """
    Return a file suffix Semgrep recognizes for the given language.
    """
    table = {
        "python": ".py",
        "bash": ".sh",
        "javascript": ".js",
        "typescript": ".ts",
        "ruby": ".rb",
        "go": ".go",
        "rust": ".rs",
    }
    return table.get(language, ".txt")


def _parse_semgrep_results(
    payload: Mapping[str, Any]
) -> list[SemgrepFinding]:
    """
    Convert Semgrep's ``--json`` output into ``SemgrepFinding`` list.

    Semgrep's JSON shape is:
    ``{"results": [{"check_id": ..., "start": {"line": int},
    "end": {"line": int}, "extra": {"severity": ..., "message": ...}}, ...]}``.
    """
    raw_results = payload.get("results") or []
    if not isinstance(raw_results, list):
        return []

    out: list[SemgrepFinding] = []
    for entry in raw_results:
        if not isinstance(entry, Mapping):
            continue
        try:
            check_id = str(entry["check_id"])
            extra = entry.get("extra") or {}
            severity = str(extra.get("severity", "INFO")).upper()
            if severity not in _VALID_SEMGREP_SEVERITIES:
                severity = "INFO"
            message = str(extra.get("message", ""))
            start = entry.get("start") or {}
            end = entry.get("end") or {}
            start_line = int(start.get("line", 0))
            end_line = int(end.get("line", start_line))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "PALISADE G4: skipping malformed semgrep finding "
                "(%s: %s); entry=%r",
                type(exc).__name__, exc, entry,
            )
            continue
        out.append(
            SemgrepFinding(
                check_id=check_id,
                severity=severity,
                message=message,
                start_line=start_line,
                end_line=end_line,
                extra=dict(extra),
            )
        )
    return out


# -----------------------------------------------------------------
# Module-level public API
# -----------------------------------------------------------------


# Backward-compatible alias. The gate was formerly ``G4CodeGate`` (Semgrep
# + Q-LLM only); it now also subsumes the deterministic sandbox path/IOC
# checks of the former G7 gate. External importers keep working.
G4CodeGate = G4SandboxCodeGate


__all__ = [
    "DEFAULT_SEMGREP_CONFIG",
    "DEFAULT_SEMGREP_TIMEOUT_SECONDS",
    "DEFAULT_WRITE_ROOTS",
    "ExtractedCode",
    "G4CodeGate",
    "G4SandboxCodeGate",
    "RUN_TOOLS",
    "SemgrepFinding",
    "WRITE_TOOLS",
    "bundled_rules_dir",
    "run_bash_violation",
    "write_path_violation",
]
