"""
Generate the evaluation section's two ablation figures from their own artifact.

Both figures were previously drawn with the percentages typed in by hand, and
both drifted: they carried a security partition two instances larger than the
measurement at five of nine configurations, and the per-boundary figure silently
plotted the *corpus* families (B1 45, B3 38, B4 45) rather than the security
partition its section defines (B1 5, B3 28, B4 40). The second is the worse
failure, because the section states that the misuse partition is excluded from
every security figure while the B1 panel was 40/45 misuse by count.

This driver parses ``full_ablation_production.md`` --- the deployed-bound
ablation, the same artifact ``tab:ablation`` is scored from --- and emits both
``figure`` environments. The artifact is the single source of truth: regenerate
and any drift in the numbers moves the curves.

Plain TikZ rather than pgfplots, matching ``budget_curve_figure`` and for the
same reason: the manuscript preamble already loads ``tikz`` and adding a package
is a compile risk not worth taking for nine points per series.

Run: ``python -m tools.eval_figures``
(``--check`` exits non-zero when a committed figure no longer matches.)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from siege.paths import REPO_ROOT

__all__ = [
    "CONFIGS",
    "SECURITY_FAMILIES",
    "Cell",
    "Ablation",
    "parse_ablation",
    "aggregate_figure",
    "boundary_figure",
]

#: The nine cumulative configurations, in the artifact's own naming, paired with
#: the short label the figures use.
CONFIGS: tuple[tuple[str, str], ...] = (
    ("baseline", "baseline"),
    ("+G4", "+G4"),
    ("+G4+G3", "+G3"),
    ("+G4+G3+G1", "+G1"),
    ("+G4+G3+G1+G5", "+G5"),
    ("full PALISADE", "full"),
    ("full +semgrep", "+semgrep"),
    ("full +slow", "+slow"),
    ("full +all", "+all"),
)

#: The security partition, which is what the evaluation section's rates are over.
#: These are *not* the corpus families: B1 excludes the 40 single-principal
#: misuse instances, and B3/B4 exclude the 15 science instances adjudicated by
#: the Contract Library. Both exclusions are what make the partitions disjoint.
SECURITY_FAMILIES: tuple[tuple[str, str], ...] = (
    ("B1", "B1 attached content"),
    ("B3", "B3 retrieval"),
    ("B4", "B4 code"),
    ("B5", "B5 submission"),
    ("XC", "XC cross-boundary"),
)

_SUITE_RE = re.compile(r"^### (Security|Science|Misuse|Benign)")
_CLASS_RE = re.compile(r"^#### \S+ -- `([^`]+)`")
_COUNT_RE = re.compile(r"^_(\d+) attack instance\(s\), (\d+) benign")
_PCT_RE = re.compile(r"^(\d+)%")


def _family_of(cls: str) -> str:
    for prefix, fam in (("b1_", "B1"), ("b3_", "B3"), ("b4_", "B4"),
                        ("b5_", "B5"), ("xc_", "XC")):
        if cls.startswith(prefix):
            return fam
    return "?"


@dataclass
class Cell:
    """One (class, configuration) measurement, as counts rather than rates."""

    n: int
    soft: int
    hard: int


@dataclass
class Ablation:
    """The parsed artifact: per-suite, per-class, per-configuration counts."""

    #: suite -> class -> config -> Cell
    cells: dict[str, dict[str, dict[str, Cell]]] = field(default_factory=dict)
    #: config -> benign utility percentage, pooled over the 181 controls
    benign_utility: dict[str, float] = field(default_factory=dict)

    def family_counts(self, suite: str, family: str, config: str) -> Cell:
        n = soft = hard = 0
        for cls, per_cfg in self.cells.get(suite, {}).items():
            if _family_of(cls) != family:
                continue
            c = per_cfg.get(config)
            if c is None:
                continue
            n += c.n
            soft += c.soft
            hard += c.hard
        return Cell(n=n, soft=soft, hard=hard)

    def suite_totals(self, suite: str, config: str) -> Cell:
        n = soft = hard = 0
        for per_cfg in self.cells.get(suite, {}).values():
            c = per_cfg.get(config)
            if c is None:
                continue
            n += c.n
            soft += c.soft
            hard += c.hard
        return Cell(n=n, soft=soft, hard=hard)


def parse_ablation(path: Path) -> Ablation:
    """Parse the ablation markdown into counts.

    The artifact prints integer percentages per (class, config); every class is
    small enough that ``round(pct/100 * n)`` recovers the count exactly. That is
    checked at the end against the artifact's own suite sizes, so a change in
    the report format fails loudly rather than producing a plausible figure.
    """
    out = Ablation()
    suite: str | None = None
    cls: str | None = None
    n_attack = 0
    n_benign = 0

    for line in path.read_text().splitlines():
        m = _SUITE_RE.match(line)
        if m:
            suite = {
                "Security": "security", "Science": "science",
                "Misuse": "misuse", "Benign": "benign",
            }[m.group(1)]
            cls = None
            continue
        m = _CLASS_RE.match(line)
        if m:
            cls = m.group(1)
            n_attack = n_benign = 0
            continue
        m = _COUNT_RE.match(line)
        if m and cls and suite:
            n_attack, n_benign = int(m.group(1)), int(m.group(2))
            out.cells.setdefault(suite, {}).setdefault(cls, {})
            continue
        if not (line.startswith("| ") and cls and suite):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5 or parts[0] in ("Configuration",) or parts[0].startswith("---"):
            continue
        cfg = parts[0]
        if suite == "benign":
            mb = _PCT_RE.match(parts[3])
            if mb is not None and n_benign:
                prev = out.benign_utility.get(cfg)
                # Pool the benign suites by instance count.
                acc = out.cells.setdefault("_benign_acc", {}).setdefault(cfg, {})
                acc_cell = acc.get("pool", Cell(0, 0, 0))
                acc["pool"] = Cell(
                    n=acc_cell.n + n_benign,
                    soft=acc_cell.soft + round(int(mb.group(1)) / 100 * n_benign),
                    hard=0,
                )
                del prev
            continue
        ma, mh = _PCT_RE.match(parts[1]), _PCT_RE.match(parts[4])
        if ma is None:
            continue
        soft = round(int(ma.group(1)) / 100 * n_attack)
        hard = round(int(mh.group(1)) / 100 * n_attack) if mh else 0
        out.cells[suite][cls][cfg] = Cell(n=n_attack, soft=soft, hard=hard)

    for cfg, acc in out.cells.pop("_benign_acc", {}).items():
        pool = acc["pool"]
        out.benign_utility[cfg] = 100.0 * pool.soft / pool.n if pool.n else 0.0

    sec = out.suite_totals("security", CONFIGS[0][0]).n
    mis = out.suite_totals("misuse", CONFIGS[0][0]).n
    sci = out.suite_totals("science", CONFIGS[0][0]).n
    if (sec, mis, sci) != (150, 40, 15):
        raise ValueError(
            f"partition sizes changed: security={sec} misuse={mis} science={sci}; "
            "expected 150/40/15. Re-check the corpus before regenerating figures."
        )
    return out


# -----------------------------------------------------------------
# TikZ emission
# -----------------------------------------------------------------

_W = 10.6   # cm, plot width for the aggregate figure
_H = 4.4    # cm, plot height


def _pts(series: list[float], *, w: float, h: float, ymax: float = 100.0) -> str:
    step = w / (len(series) - 1)
    return " ".join(
        f"({i * step:.3f},{v / ymax * h:.3f})" for i, v in enumerate(series)
    )


def _axis(w: float, h: float, ticks: list[int], ymax: float, label: str) -> list[str]:
    out = [
        f"\\draw[gray!45] (0,0) -- ({w:.2f},0);",
        f"\\draw[gray!45] (0,0) -- (0,{h:.2f});",
    ]
    for t in ticks:
        y = t / ymax * h
        out.append(f"\\draw[gray!20] (0,{y:.3f}) -- ({w:.2f},{y:.3f});")
        out.append(
            f"\\node[left,font=\\tiny,gray!70] at (0,{y:.3f}) {{{t}}};"
        )
    out.append(
        f"\\node[rotate=90,font=\\tiny,gray!70] at (-0.72,{h / 2:.2f}) {{{label}}};"
    )
    return out


def aggregate_figure(ab: Ablation) -> str:
    """The cumulative figure: security soft/hard, misuse, and benign utility."""
    sec_soft, sec_hard, mis, ben = [], [], [], []
    for cfg, _ in CONFIGS:
        s = ab.suite_totals("security", cfg)
        m = ab.suite_totals("misuse", cfg)
        sec_soft.append(100.0 * s.soft / s.n)
        sec_hard.append(100.0 * s.hard / s.n)
        mis.append(100.0 * m.soft / m.n)
        ben.append(ab.benign_utility.get(cfg, 100.0))

    step = _W / (len(CONFIGS) - 1)
    lines = [
        "% GENERATED by tools.eval_figures -- do not hand-edit.",
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[font=\\footnotesize]",
        "% --- attack-success panel ---",
    ]
    lines += _axis(_W, _H, [0, 25, 50, 75, 100], 100.0, "attack success (\\%)")
    for series, style, name in (
        (mis, "green!55!black, dashed", "Misuse ASR (single principal)"),
        (sec_soft, "blue!70!black", "Security soft-win ASR"),
        (sec_hard, "orange!85!black", "Security hard-win ASR"),
    ):
        lines.append(
            f"\\draw[{style}, thick] plot[mark=*,mark size=1pt] coordinates {{"
            f"{_pts(series, w=_W, h=_H)}}};"
        )
        del name
    for series, col in ((mis, "green!55!black"), (sec_soft, "blue!70!black"),
                        (sec_hard, "orange!85!black")):
        lines.append(
            f"\\node[right,font=\\tiny,{col}] at ({_W:.2f},"
            f"{series[-1] / 100 * _H:.3f}) {{{series[-1]:.0f}\\%}};"
        )
    # legend
    lines += [
        f"\\node[right,font=\\tiny] at (0.15,{_H - 0.25:.2f}) "
        "{\\textcolor{green!55!black}{--\\,--} misuse \\quad "
        "\\textcolor{blue!70!black}{---} security soft \\quad "
        "\\textcolor{orange!85!black}{---} security hard};",
        "% --- benign-utility panel ---",
        f"\\begin{{scope}}[yshift=-2.35cm]",
    ]
    bh = 1.5
    lines += [
        f"\\draw[gray!45] (0,0) -- ({_W:.2f},0);",
        f"\\draw[gray!45] (0,0) -- (0,{bh:.2f});",
    ]
    for t in (90, 95, 100):
        y = (t - 88) / 12 * bh
        lines.append(f"\\draw[gray!20] (0,{y:.3f}) -- ({_W:.2f},{y:.3f});")
        lines.append(f"\\node[left,font=\\tiny,gray!70] at (0,{y:.3f}) {{{t}}};")
    lines.append(
        "\\draw[yellow!60!black, thick] plot[mark=*,mark size=1pt] coordinates {"
        + " ".join(
            f"({i * step:.3f},{(v - 88) / 12 * bh:.3f})" for i, v in enumerate(ben)
        )
        + "};"
    )
    lines.append(
        f"\\node[rotate=90,font=\\tiny,gray!70] at (-0.72,{bh / 2:.2f}) "
        "{benign utility (\\%)};"
    )
    for i, (_, short) in enumerate(CONFIGS):
        lines.append(
            f"\\node[rotate=40,anchor=north east,font=\\tiny] at "
            f"({i * step:.3f},-0.08) {{\\texttt{{{short}}}}};"
        )
    lines += ["\\end{scope}", "\\end{tikzpicture}"]
    lines += [
        "\\caption{Cumulative ablation on the 150-instance security partition, "
        "production taint bound, seed 42. Misuse (40 single-principal instances) "
        "is plotted separately and is excluded from the security series; benign "
        "utility is pooled over the 181 controls. Generated by "
        "\\texttt{tools.eval\\_figures}.}",
        "\\label{fig:ablation-aggregate}",
        "\\end{figure}",
    ]
    return "\n".join(lines)


def boundary_figure(ab: Ablation) -> str:
    """Per-boundary small multiples, on the security partition.

    Drawn on the security partition rather than the corpus families, so the
    panels and ``tab:denominators`` carry the same denominators and the misuse
    exclusion the section states actually holds in the figure.
    """
    pw, ph = 3.1, 2.0
    lines = [
        "% GENERATED by tools.eval_figures -- do not hand-edit.",
        "\\begin{figure*}[t]",
        "\\centering",
        "\\begin{tikzpicture}[font=\\footnotesize]",
    ]
    for idx, (fam, title) in enumerate(SECURITY_FAMILIES):
        col, row = idx % 3, idx // 3
        x0, y0 = col * (pw + 1.15), -row * (ph + 1.35)
        soft, hard = [], []
        n = 0
        for cfg, _ in CONFIGS:
            c = ab.family_counts("security", fam, cfg)
            n = c.n
            soft.append(100.0 * c.soft / c.n if c.n else 0.0)
            hard.append(100.0 * c.hard / c.n if c.n else 0.0)
        lines.append(f"\\begin{{scope}}[xshift={x0:.2f}cm,yshift={y0:.2f}cm]")
        lines.append(
            f"\\node[anchor=south west,font=\\scriptsize] at (0,{ph + 0.12:.2f}) "
            f"{{\\textbf{{{title}}} ($n={n}$)}};"
        )
        lines += _axis(pw, ph, [0, 50, 100], 100.0, "rate (\\%)")
        lines.append(
            "\\draw[blue!70!black, thick] plot[mark=*,mark size=0.7pt] coordinates {"
            + _pts(soft, w=pw, h=ph) + "};"
        )
        lines.append(
            "\\draw[orange!85!black, thick] plot[mark=*,mark size=0.7pt] coordinates {"
            + _pts(hard, w=pw, h=ph) + "};"
        )
        step = pw / (len(CONFIGS) - 1)
        for i, (_, short) in enumerate(CONFIGS):
            lines.append(
                f"\\node[rotate=45,anchor=north east,font=\\tiny] at "
                f"({i * step:.3f},-0.06) {{\\texttt{{{short}}}}};"
            )
        lines.append("\\end{scope}")

    # residual panel
    x0, y0 = 2 * (pw + 1.15), -(ph + 1.35)
    lines.append(f"\\begin{{scope}}[xshift={x0:.2f}cm,yshift={y0:.2f}cm]")
    resid = [
        (fam, ab.family_counts("security", fam, "full +all").hard)
        for fam, _ in SECURITY_FAMILIES
    ]
    total = sum(h for _, h in resid)
    lines.append(
        f"\\node[anchor=south west,font=\\scriptsize] at (0,{ph + 0.12:.2f}) "
        f"{{\\textbf{{Residual at \\texttt{{+all}}}} ({total} of 150)}};"
    )
    lines += _axis(pw, ph, [0, 2, 4], 5.0, "hard wins")
    bw = pw / (len(resid) * 1.6)
    for i, (fam, h) in enumerate(resid):
        cx = (i + 0.5) * pw / len(resid)
        if h:
            lines.append(
                f"\\fill[orange!85!black] ({cx - bw / 2:.3f},0) rectangle "
                f"({cx + bw / 2:.3f},{h / 5 * ph:.3f});"
            )
            lines.append(
                f"\\node[above,font=\\tiny] at ({cx:.3f},{h / 5 * ph:.3f}) {{{h}}};"
            )
        lines.append(
            f"\\node[below,font=\\tiny] at ({cx:.3f},-0.05) {{{fam}}};"
        )
    lines.append("\\end{scope}")
    lines += ["\\end{tikzpicture}"]
    lines += [
        "\\caption{Soft-win and hard-win rate per boundary across the cumulative "
        "configurations, on the \\emph{security} partition of "
        "Table~\\ref{tab:denominators} ($n=150$); the misuse and science "
        "partitions are excluded, so these denominators are smaller than the "
        "corpus families of the same name. Blue is soft win, orange is hard win. "
        "\\texttt{+semgrep} and \\texttt{+slow} are independent additions to "
        "\\texttt{full} rather than successive steps, which is why B4's hard-win "
        "rate is non-monotone between them. Generated by "
        "\\texttt{tools.eval\\_figures}.}",
        "\\label{fig:ablation-boundary}",
        "\\end{figure*}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    repo = REPO_ROOT
    ap.add_argument(
        "--artifact",
        default=str(repo / "docs/palisade/full_ablation_production.md"),
        help="the deployed-bound ablation the figures are generated from",
    )
    ap.add_argument("--out", default=None, help="write the LaTeX here")
    ap.add_argument(
        "--check", action="store_true",
        help="exit non-zero if --out is missing or no longer matches the artifact",
    )
    ap.add_argument("--table", action="store_true", help="also print the counts table")
    args = ap.parse_args(argv)

    ab = parse_ablation(Path(args.artifact))
    tex = aggregate_figure(ab) + "\n\n" + boundary_figure(ab) + "\n"

    if args.table:
        print("config           soft/150   hard/150   misuse/40   benign%")
        for cfg, short in CONFIGS:
            s = ab.suite_totals("security", cfg)
            m = ab.suite_totals("misuse", cfg)
            print(
                f"{short:<15} {s.soft:>4}/{s.n:<4} {s.hard:>4}/{s.n:<4} "
                f"{m.soft:>4}/{m.n:<4}  {ab.benign_utility.get(cfg, 0):.1f}"
            )
        print()

    if args.check:
        if args.out is None:
            print("--check needs --out")
            return 2
        p = Path(args.out)
        if not p.exists():
            print(f"STALE: {p} does not exist")
            return 1
        if p.read_text() != tex:
            print(f"STALE: {p} no longer matches {args.artifact}")
            return 1
        print(f"ok: {p} matches {args.artifact}")
        return 0

    if args.out:
        Path(args.out).write_text(tex)
        print(f"wrote {args.out}")
    else:
        print(tex)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
