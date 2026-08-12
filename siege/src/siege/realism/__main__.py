"""
CLI: print the SIEGE corpus-realism report.

    uv run python -m siege.realism

Exit code is non-zero when any ``error``-severity finding is present, so it
doubles as a pre-commit / CI check alongside ``tests/test_corpus_realism``.
"""

from __future__ import annotations

import sys
from collections import Counter

from siege import load_instances
from siege.corpus_builder import CORPUS_DIR
from siege.realism.checks import check_corpus
from siege.realism.manifest import tool_catalog


def main(argv: list[str] | None = None) -> int:
    instances = load_instances(CORPUS_DIR)
    findings = check_corpus(instances)
    errors = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warn"]

    print(
        f"[realism] {len(instances)} instances vs {len(tool_catalog())} real tools "
        f"| {len(errors)} error(s), {len(warns)} warning(s)"
    )
    for f in errors:
        print(f"  {f}")
    for f in warns:
        print(f"  {f}")
    if findings:
        print(f"[realism] by check: {dict(Counter(f.check for f in findings))}")
    if not errors:
        print(
            "[realism] PASS -- every instance's tools, args, gates, uploads and "
            "config claims are consistent with the live implementation."
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
