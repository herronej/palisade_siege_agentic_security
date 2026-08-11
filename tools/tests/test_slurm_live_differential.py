"""
R5-C5: live slurmctld parser differential.

The controller integration is exercised by the live run (documented in
`slurm_live_differential_c5.md`); these lock the pure logic that decides what
counts as a bypass, so a future change can't silently reclassify an over-read as
a bypass or mis-parse the controller's time format.
"""

from __future__ import annotations

from tools.slurm_live_differential import LiveDiff, _parse_ctld_time


def test_ctld_time_parsing() -> None:
    assert _parse_ctld_time("1-23:00:00") == 86400 + 23 * 3600  # 47h, the split_directive time
    assert _parse_ctld_time("02:00:00") == 2 * 3600
    assert _parse_ctld_time("00:05:00") == 300
    assert _parse_ctld_time("UNLIMITED") is None
    assert _parse_ctld_time("(null)") is None


def test_bypass_is_g5_underreading_only() -> None:
    """A bypass is G5 reading a *smaller* job than the controller runs; G5 reading
    a *larger* value is the fail-safe over-read the manuscript's dagger describes."""
    # over-read: G5=512, controller=1 -> fail-safe, NOT a bypass (the live result).
    over = LiveDiff("leading_whitespace", "variant",
                    g5={"nodes": 512, "time_seconds": None, "partition": "batch"},
                    ctld={"nodes": 1, "time_seconds": None, "partition": "batch"})
    assert over.disagreements  # they disagree on nodes
    assert not over.is_bypass

    # bypass: G5 reads 4 nodes, controller runs 512 -> G5 would admit the big job.
    under = LiveDiff("hypothetical", "variant",
                     g5={"nodes": 4, "time_seconds": 3600, "partition": "batch"},
                     ctld={"nodes": 512, "time_seconds": 3600, "partition": "batch"})
    assert under.is_bypass


def test_agreement_and_rejection() -> None:
    agree = LiveDiff("split_directive", "variant",
                     g5={"nodes": 512, "time_seconds": 169200, "partition": "batch"},
                     ctld={"nodes": 512, "time_seconds": 169200, "partition": "batch"})
    assert not agree.disagreements
    assert not agree.is_bypass

    rej = LiveDiff("crlf_endings", "variant",
                   g5={"nodes": 4, "time_seconds": 3600, "partition": "batch"},
                   ctld={"rejected": "sbatch: error: ... UNIX line breaks"})
    assert rej.rejected
    assert not rej.disagreements
    assert not rej.is_bypass
