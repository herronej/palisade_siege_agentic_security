"""The payload-safety claim, checked rather than asserted.

The paper states that public attack payloads are detection-pattern triggers
with *placeholder hosts* rather than working exploits. The host half of that
is mechanically checkable: every network identifier reachable in the public
corpus must be drawn from a range that cannot route to a real system.

Two real-world identifiers were found and fixed during release preparation
(see docs/palisade/RELEASE_SAFETY.md). This test is what keeps them from
coming back through a regenerated pool or a new class.
"""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path

import pytest

from siege.paths import CONTROLS_DIR, CORPUS_DIR, SIEGE_DIR

# RFC 2606 / 6761 reserved TLDs and second-level names.
RESERVED_SUFFIXES = (
    ".example", ".invalid", ".test", ".localhost",
    "example.com", "example.net", "example.org",
)

# Networks that cannot route to a real system: RFC 5737 documentation,
# RFC 1918 private, loopback, and link-local.
RESERVED_NETS = [
    ipaddress.ip_network(n)
    for n in (
        "192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24",  # RFC 5737
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",      # RFC 1918
        "127.0.0.0/8", "169.254.0.0/16", "0.0.0.0/8",
    )
]

# A host only matters here if a payload could *connect* to it, so match
# network contexts rather than bare dotted tokens -- otherwise every Python
# attribute access and every `foo.inp` filename reads as a hostname.
HOST_RE = re.compile(
    r"""(?x)
    (?: [a-z]+://                                   # scheme://host
      | /dev/tcp/                                   # bash reverse shell
      | stratum\+tcp://                             # mining pool
      | (?:curl|wget|nc|ncat|netcat|telnet|ssh|scp|sftp|rsync|ping|dig|
           nslookup|host|ftp)\s+(?:-[^\s]+\s+)*     # network tool + flags
      | @                                           # user@host
    )
    ((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})
    """
)
IPV4_RE = re.compile(
    r"""(?x)
    (?: [a-z]+:// | /dev/tcp/ | stratum\+tcp:// | @
      | (?:curl|wget|nc|ncat|netcat|telnet|ssh|scp|sftp|rsync|ping|dig|
           nslookup|host|ftp)\s+(?:-[^\s]+\s+)*
    )
    ((?:\d{1,3}\.){3}\d{1,3})
    """
)

# Hostnames that are documentation, standards or facility references rather
# than payload destinations.
# Facility and standards references that are not payload destinations. The
# `ornl.gov` entries appear in the *benign* control as realistic `--mail-user`
# and staging endpoints; the near-manifold control exists precisely to sit
# close to real scientific work, so replacing them with placeholders would
# degrade the thing it measures.
BENIGN_HOSTS = {
    "www.w3.org", "doi.org", "arxiv.org", "dx.doi.org",
    "ornl.gov", "olcf.ornl.gov", "docs.olcf.ornl.gov",
    "dtn.olcf.ornl.gov", "data.olcf.ornl.gov", "code.ornl.gov",
    "www.energy.gov", "genesis.energy.gov",
}

# Filename-like and version-like tokens the host regex over-matches.
NOT_A_HOST = re.compile(
    r"\.(py|ya?ml|json|md|txt|sh|lua|log|csv|tsv|out|err|so|gz|tar|cfg|ini|toml|"
    r"png|jpg|pdf|c|h|f90|cu|ipynb|db|bin|npy|dat|in|sub|slurm)$",
    re.IGNORECASE,
)


def _public_corpus_files() -> list[Path]:
    """Every file a public release would ship as corpus content.

    This includes the reward-ranked generated pools and the adaptive
    adversary's attack modules. They were once scoped for gated release; the
    whole artifact is public, so they are held to the same bar as the
    authored corpus rather than to a looser one.
    """
    files: list[Path] = []
    for root in (CORPUS_DIR, CONTROLS_DIR, SIEGE_DIR / "corpus_generated"):
        if root.is_dir():
            files += sorted(root.rglob("*.yaml"))
    files += sorted((SIEGE_DIR / "templates").glob("*.py"))
    files += sorted((SIEGE_DIR / "redteam" / "attacks").rglob("*.py"))
    return files


def _is_reserved_host(host: str) -> bool:
    h = host.lower().rstrip(".")
    return any(h == s.lstrip(".") or h.endswith(s) for s in RESERVED_SUFFIXES)


def _is_reserved_ip(text: str) -> bool:
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return True  # not an address (e.g. a version string); not our concern
    return any(addr in net for net in RESERVED_NETS)


def test_public_corpus_is_not_empty() -> None:
    """Guard the guard: an empty scan must not read as a pass."""
    files = _public_corpus_files()
    assert len(files) > 300, f"expected the full corpus, found {len(files)} files"


@pytest.mark.parametrize("path", _public_corpus_files(), ids=lambda p: p.name)
def test_no_routable_hosts_in_public_corpus(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")

    offenders = [
        h
        for h in HOST_RE.findall(text)
        if not _is_reserved_host(h)
        and h.lower() not in BENIGN_HOSTS
        and not NOT_A_HOST.search(h)
    ]
    assert not offenders, (
        f"{path.relative_to(SIEGE_DIR)} names non-reserved host(s) {sorted(set(offenders))}. "
        f"Public payloads must use RFC 2606 names (.example/.invalid/.test)."
    )

    bad_ips = [ip for ip in IPV4_RE.findall(text) if not _is_reserved_ip(ip)]
    assert not bad_ips, (
        f"{path.relative_to(SIEGE_DIR)} names routable address(es) {sorted(set(bad_ips))}. "
        f"Public payloads must use RFC 5737 documentation ranges."
    )


def test_mining_signatures_stay_on_the_defense_side() -> None:
    """Real mining-pool hostnames are detection signatures, not destinations.

    They must appear in the gate's denylist -- removing them silently weakens
    G5 -- and must not appear in the corpus, where they would be a payload
    pointed at live infrastructure.
    """
    from palisade.gates import denylists

    denylist_text = Path(denylists.__file__).read_text()
    assert "minexmr" in denylist_text, "the mining denylist lost its signatures"

    for path in _public_corpus_files():
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for sig in ("minexmr.com", "supportxmr.com", "nanopool.org", "hashvault.pro"):
            assert sig not in text, (
                f"{path.name} names the real pool {sig}; corpus payloads must "
                f"use a reserved placeholder even when the gate matches on the "
                f"real signature."
            )
