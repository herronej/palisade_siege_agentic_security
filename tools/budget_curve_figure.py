"""
Generate the paper's hard-win-ASR-vs-query-budget figure from its own artifact.

The figure was previously hand-drawn TikZ with the measured percentages typed
into ``\\draw`` coordinates. It was faithful, but a results figure that is not
generated from the data it reports is a reproducibility liability under the SC26
initiative (R1-8b / A44), and nothing mechanically checked that the coordinates
still matched the measurement.

This driver parses the committed measurement --- ``adaptive_budget_curve_w19.md``,
the output of the 20-seed adaptive sweep --- and emits the LaTeX ``figure``
environment. The artifact is the single source of truth: regenerate the figure
and any drift in the numbers moves the curve.

Plain TikZ rather than pgfplots, deliberately: the manuscript preamble already
loads ``tikz`` and adding a package is a compile risk we do not need to take for
four points per series.

Run: ``python -m tools.budget_curve_figure``
(``--check`` exits non-zero if the manuscript's figure is stale.)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from palisade.paths import REPO_ROOT

__all__ = ["Series", "parse_curves", "render_figure", "main"]

_DOCS = REPO_ROOT / "docs" / "palisade"
_ARTIFACT = _DOCS / "adaptive_budget_curve_w19.md"
_MANUSCRIPT = _DOCS / "manuscript.tex"

#: Query budgets, in artifact column order. Plotted at even x-spacing (the
#: budgets are geometric, so even spacing is a log axis).
BUDGETS: tuple[int, ...] = (16, 64, 256, 1000)

#: (artifact section heading fragment, short driver name, TikZ line style).
_DRIVERS: tuple[tuple[str, str, str], ...] = (
    ("Propagation-search", "propagation search", "thick"),
    ("Scheduler-injection", "scheduler injection", "thick,densely dashed"),
)

_TIERS: tuple[str, ...] = ("black_box", "grey_box", "white_box")

#: Band shading per tier. Two distinguishable greys, so a reader can tell which
#: band belongs to which curve; identical fills carried no information.
_BAND_FILL: dict[str, str] = {"black_box": "black!16", "grey_box": "black!7"}

# Plot geometry (TikZ cm). Kept here so the emitted figure is fully determined
# by this file plus the artifact.
_W, _H = 6.0, 3.0


@dataclass(frozen=True)
class Series:
    """One driver x tier curve: mean hard-win ASR at each budget, as percents."""

    driver: str
    tier: str
    style: str
    means: tuple[float, ...]
    los: tuple[float, ...]
    his: tuple[float, ...]

    @property
    def label(self) -> str:
        return self.tier.split("_")[0]


def _parse_pct(cell: str) -> float:
    """Leading percentage of a ``70% [66%,73%]`` cell."""
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*%", cell)
    if not m:
        raise ValueError(f"no leading percentage in cell {cell!r}")
    return float(m.group(1))


def _parse_ci(cell: str) -> tuple[float, float]:
    """Bracketed seed-bootstrap interval of a ``70% [66%,73%]`` cell.

    The caption claims the intervals narrow with budget, so the figure has to
    carry them: a caption asserting convergence over bands it does not draw is
    a promise the reader cannot check.
    """
    m = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*%?\s*,\s*(\d+(?:\.\d+)?)\s*%?\s*\]", cell)
    if not m:
        raise ValueError(f"no bracketed interval in cell {cell!r}")
    return float(m.group(1)), float(m.group(2))


def parse_curves(text: str) -> list[Series]:
    """Extract every driver x tier curve from the artifact markdown.

    Raises if a driver section, a tier row, or a budget column is missing, so a
    silently-truncated artifact fails the build rather than emitting a short
    curve.
    """
    out: list[Series] = []
    for heading, driver, style in _DRIVERS:
        block = re.search(
            rf"###\s+{re.escape(heading)}.*?\n(.*?)(?=\n###|\n##|\Z)", text, re.S
        )
        if block is None:
            raise ValueError(f"artifact has no '{heading}' section")
        for tier in _TIERS:
            row = re.search(rf"^\|\s*{tier}\s*\|(.+)$", block.group(1), re.M)
            if row is None:
                raise ValueError(f"'{heading}' has no {tier} row")
            cells = [c for c in row.group(1).split("|") if c.strip()]
            if len(cells) < len(BUDGETS):
                raise ValueError(
                    f"'{heading}' {tier}: {len(cells)} cells, need {len(BUDGETS)}"
                )
            cols = cells[: len(BUDGETS)]
            cis = [_parse_ci(c) for c in cols]
            out.append(
                Series(
                    driver=driver,
                    tier=tier,
                    style=style,
                    means=tuple(_parse_pct(c) for c in cols),
                    los=tuple(lo for lo, _ in cis),
                    his=tuple(hi for _, hi in cis),
                )
            )
    return out


def _xy(i: int, pct: float) -> str:
    x = _W * i / (len(BUDGETS) - 1)
    y = _H * pct / 100.0
    return f"({x:.2f},{y:.2f})"


def _path(s: Series) -> str:
    return " -- ".join(_xy(i, m) for i, m in enumerate(s.means))


def _band(s: Series) -> str:
    """Closed polygon of the seed-bootstrap interval, drawn under the mean."""
    up = [_xy(i, v) for i, v in enumerate(s.his)]
    down = [_xy(i, v) for i, v in reversed(list(enumerate(s.los)))]
    return " -- ".join(up + down) + " -- cycle"


def render_figure(series: list[Series]) -> str:
    """Emit the complete LaTeX ``figure`` environment."""
    by = {(s.driver, s.tier): s for s in series}

    # White-box saturates at 100% for both drivers, so the two coincide: draw
    # one line and say so, rather than overplotting and implying two results.
    white = [by[(d, "white_box")] for _, d, _ in _DRIVERS]
    if not all(all(m == 100.0 for m in w.means) for w in white):
        raise ValueError("white-box no longer saturates; the merged line is wrong")

    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "% GENERATED by tools.budget_curve_figure from",
        "% docs/palisade/adaptive_budget_curve_w19.md -- do not hand-edit.",
        "\\begin{tikzpicture}[font=\\footnotesize]",
        f"  \\draw[->] (0,0) -- ({_W + 0.6:.2f},0) node[right] {{\\scriptsize $Q$}};",
        f"  \\draw[->] (0,0) -- (0,{_H + 0.2:.2f}) "
        "node[above] {\\scriptsize hard-win ASR (\\%)};",
    ]
    ticks = "/".join("")  # placeholder to keep the loop literal readable
    del ticks
    xt = ", ".join(
        f"{_W * i / (len(BUDGETS) - 1):.2f}/{b}" for i, b in enumerate(BUDGETS)
    )
    lines += [
        f"  \\foreach \\x/\\l in {{{xt}}} "
        "\\draw (\\x,0) -- (\\x,-0.07) node[below] {\\scriptsize \\l};",
        f"  \\foreach \\y/\\l in {{0/0, {_H / 2:.2f}/50, {_H:.2f}/100}} "
        "\\draw (0,\\y) -- (-0.07,\\y) node[left] {\\scriptsize \\l};",
        f"  \\draw[thick,dotted] (0,{_H:.2f}) -- ({_W:.2f},{_H:.2f});",
        f"  \\node[right] at ({_W + 0.03:.2f},{_H:.2f}) {{\\scriptsize white}};",
    ]
    # Tier labels at the right edge (one short word, so they fit inside the
    # column); the driver is carried by line style and named in the legend
    # below, which sits in the empty lower band no curve enters.
    # Every band first, then every line. Interleaving them lets a later series'
    # fill paint over an earlier series' curve, which is what happened when the
    # grey bands were emitted after the black lines were drawn.
    plotted = [
        by[(driver, tier)]
        for _, driver, _ in _DRIVERS
        for tier in ("black_box", "grey_box")
    ]
    for s in plotted:
        lines.append(f"  \\fill[{_BAND_FILL[s.tier]}] {_band(s)};")
    for s in plotted:
        lines.append(f"  \\draw[{s.style}] {_path(s)};")
        lines.append(
            f"  \\node[right] at ({_W + 0.03:.2f},{_H * s.means[-1] / 100:.2f}) "
            f"{{\\scriptsize {s.label}}};"
        )
    # Anchor the legend below every plotted point, so it cannot collide with a
    # curve whatever the measurement moves to.
    floor = min(v for s in series if s.tier != "white_box" for v in s.los)
    legend_y = _H * floor / 100.0 - 0.40
    for k, (_, driver, style) in enumerate(_DRIVERS):
        y = legend_y - 0.35 * k
        lines += [
            f"  \\draw[{style}] (0.35,{y:.2f}) -- (1.05,{y:.2f});",
            f"  \\node[right] at (1.05,{y:.2f}) {{\\scriptsize {driver}}};",
        ]
    lines += [
        "\\end{tikzpicture}",
        _caption(by),
        "\\label{fig:budget}",
        "\\end{figure}",
    ]
    return "\n".join(lines)


def _caption(by: dict[tuple[str, str], Series]) -> str:
    p_b, p_g = by[("propagation search", "black_box")], by[
        ("propagation search", "grey_box")
    ]
    s_b, s_g = by[("scheduler injection", "black_box")], by[
        ("scheduler injection", "grey_box")
    ]
    return (
        "\\caption{Hard-win ASR against query budget for the two drivers that "
        "reach a sink tag-dropped, by access tier, deployed predicate. Curves "
        "are means over 20 seeds; shaded bands are seed-level percentile "
        "bootstraps rather than a pooled binomial, because episodes within a "
        "bandit run are correlated. Scheduler injection runs "
        f"{s_b.means[0]:.0f}\\% black-box and {s_g.means[0]:.0f}\\% grey-box at "
        f"$Q{{=}}16$, reaching {s_b.means[-1]:.0f}\\% and {s_g.means[-1]:.0f}\\% "
        f"at $Q{{=}}1000$; propagation search {p_b.means[0]:.0f}\\% to "
        f"{p_b.means[-1]:.0f}\\% black-box and {p_g.means[0]:.0f}\\% to "
        f"{p_g.means[-1]:.0f}\\% grey-box. The bands narrow sharply with "
        f"budget --- scheduler grey-box from "
        f"[{s_g.los[0]:.0f}, {s_g.his[0]:.0f}] at $Q{{=}}16$ to "
        f"[{s_g.los[-1]:.0f}, {s_g.his[-1]:.0f}] at $Q{{=}}1000$ --- and that is "
        "convergence, not low variance: every seed ends at the same "
        "characterized evading set, which no run exceeds even at 1000 queries. "
        "White-box saturates at 100\\% from $Q{=}16$ for both drivers, so the "
        "two coincide and carry no tier information. Under the oracle-assisted "
        "condition both drivers are 0\\% at every budget and tier. Generated "
        "from \\texttt{tools.budget\\_curve\\_figure}.}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the manuscript's figure differs from the artifact",
    )
    ap.add_argument("--write", action="store_true", help="splice into manuscript.tex")
    args = ap.parse_args(argv)

    fig = render_figure(parse_curves(_ARTIFACT.read_text()))

    if not (args.check or args.write):
        print(fig)
        return 0

    tex = _MANUSCRIPT.read_text()
    pat = re.compile(
        r"\\begin\{figure\}\[t\]\n(?:(?!\\end\{figure\}).)*?\\label\{fig:budget\}\n"
        r"\\end\{figure\}",
        re.S,
    )
    if pat.search(tex) is None:
        print("fig:budget not found in manuscript", file=sys.stderr)
        return 2
    if args.check:
        stale = pat.search(tex).group(0) != fig
        print("STALE" if stale else "up to date")
        return 1 if stale else 0
    _MANUSCRIPT.write_text(pat.sub(lambda _: fig, tex, count=1))
    print(f"spliced fig:budget into {_MANUSCRIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
