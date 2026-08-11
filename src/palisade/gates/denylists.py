"""
``gates/denylists.py`` -- shared deterministic IOC denylists for the
sandbox (G4) and HPC-job (G5) gates.

These indicator lists are consumed by *both* the merged G4 sandbox/code
gate (the ``run_bash`` execution-IOC scan) and the G5 HPC-job gate (the
SLURM body scan). They live in a neutral module that neither gate owns --
one source of truth, and no cross-gate import cycle (G5's slow tier
already imports G4's intent comparator, so G4 must not import G5).
``g5_hpc`` re-exports these names for backward compatibility.

Sources: public threat-intel feeds (Cado Security / Sysdig
cloud-cryptomining reports, the abuse.ch feodotracker miner-family list,
and the well-known XMRig/ethminer families). Operator policy augments
rather than replaces these sets.
"""

from __future__ import annotations

# Public mining-binary IOC list. >= 10 entries per the work-item
# acceptance criterion; the operator file augments rather than replaces
# this set.
DEFAULT_MINING_BINARY_DENYLIST: frozenset[str] = frozenset(
    {
        "xmrig",
        "xmr-stak",
        "xmr-stak-rx",
        "ethminer",
        "phoenixminer",
        "t-rex",
        "trex",
        "nbminer",
        "lolminer",
        "gminer",
        "teamredminer",
        "cpuminer",
        "cpuminer-multi",
        "minerd",
        "cgminer",
        "bfgminer",
        "ccminer",
        "srbminer",
        "srbminer-multi",
        "nanominer",
        "bminer",
        "miniz",
        "kawpowminer",
        "verthashminer",
        "wildrig",
        "xmrig-cuda",
    }
)

# Substring signatures (lower-cased) that betray a mining payload even
# when the binary itself has been renamed -- pool protocol URIs,
# proof-of-work algorithm markers, and the dominant public Monero pools.
# These catch the "library-call mining" pattern where no denylisted
# binary name appears.
DEFAULT_MINING_SIGNATURES: tuple[str, ...] = (
    "stratum+tcp://",
    "stratum+ssl://",
    "stratum2+tcp://",
    "randomx",
    "cryptonight",
    "argon2id_chukwa",
    "minexmr",
    "supportxmr.com",
    "nanopool.org",
    "pool.minexmr.com",
    "xmrpool",
    "nicehash.com",
    "ethermine.org",
    "f2pool.com",
    "2miners.com",
    "--donate-level",
    "--coin=monero",
)

# Known typo-squatted import names. The always-on G4 Tier-0 import check
# (``import_typo_squat_violation``) reads this set, and the Semgrep-tier rule
# ``contracts/semgrep/typo_squat.yml`` mirrors it -- one source of truth so the
# deterministic floor catches the same squats as the optional scanner. Stored
# lower-cased (the Tier-0 check lower-cases the parsed module name); the
# Semgrep rule keeps the original casing for the case-confusable ``jeIlyfish``.
# Stdlib-colliding names (e.g. ``crypt``) are deliberately left to the Semgrep
# rule, not the always-on floor, to keep the deterministic check FP-free.
DEFAULT_TYPO_SQUAT_DENYLIST: frozenset[str] = frozenset(
    {
        # general-PyPI squats
        "urllib3_secure",
        "urllib4",
        "requests3",
        "requestes",
        "python_dateutils",
        "dateutils",
        "jeilyfish",
        "python3_dateutil",
        "colourama",
        "beautifulsup4",
        "setup_tools",
        "pytorch",
        "tensorfllow",
        # scientific-stack squats (SIEGE b4_3)
        "mdanalysys",
        "scypi",
        "h5py3",
        "pymatgen3",
        "aes",
    }
)

# Legitimate scientific-Python top-level imports. The Tier-0 import check
# treats an unknown module that is a single-character (Damerau/OSA distance 1)
# near-miss of one of these as a typo-squat -- the generalization beyond the
# fixed denylist above (catches unseen single-edit squats like ``h5py3`` or
# ``mdanalysys`` without enumerating them). An exact match here is always
# allowed. Operator policy augments rather than replaces this set.
DEFAULT_SCIENTIFIC_IMPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "sympy",
        "h5py",
        "netcdf4",
        "xarray",
        "mdanalysis",
        "pymatgen",
        "ase",
        "sklearn",
        "scikit_learn",
        "statsmodels",
        "numba",
        "torch",
        "tensorflow",
        "jax",
        "plotly",
        "bokeh",
        "requests",
        "pint",
        "thermo",
        "cantera",
        "periodictable",
        "chempy",
        "rdkit",
        "networkx",
    }
)

# Sensitive filesystem locations for the path-scoping check. Matched as
# substrings against path tokens (and the raw script / command) so
# ``$HOME/.ssh/id_rsa`` and ``~/.ssh/id_rsa`` both fire.
DEFAULT_SENSITIVE_PATH_PATTERNS: tuple[str, ...] = (
    "~/.ssh",
    "$HOME/.ssh",
    "/.ssh/",
    "~/.aws",
    "$HOME/.aws",
    "/.aws/",
    "/.config/gcloud",
    "/etc/passwd",
    "/etc/shadow",
    "/etc/sudoers",
    "/root/",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
)


__all__ = [
    "DEFAULT_MINING_BINARY_DENYLIST",
    "DEFAULT_MINING_SIGNATURES",
    "DEFAULT_SENSITIVE_PATH_PATTERNS",
    "DEFAULT_TYPO_SQUAT_DENYLIST",
    "DEFAULT_SCIENTIFIC_IMPORT_ALLOWLIST",
]
