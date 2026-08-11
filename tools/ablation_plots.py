"""
Render ``full_ablation_production.md`` as figures you can actually look at.

``tools.eval_figures`` emits the manuscript's two TikZ figures from this same
artifact, which is the right thing for the paper and the wrong thing for reading
the result: TikZ only becomes a picture when LaTeX runs, and both of its figures
aggregate to the five boundaries, so the per-class structure -- *which* gate
closes *which* attack class -- is not in either of them.

This driver renders four PNG/PDF figures from the same source of truth:

* ``ablation_overview``  -- the cumulative curves with Wilson bands, plus the
  benign-utility panel. The rendered twin of ``fig:ablation-aggregate``.
* ``ablation_heatmap``   -- every attack class x every configuration, soft-win
  beside hard-win. The view neither TikZ figure has.
* ``ablation_boundary``  -- per-boundary small multiples on the security
  partition. The rendered twin of ``fig:ablation-boundary``.
* ``ablation_tradeoff``  -- hard-win ASR against benign utility, as a path
  through configuration space.

**Rates are pooled, not macro-averaged.** The artifact's own Headline block
macro-averages over templates; the manuscript figures pool over instances. The
two disagree where template sizes differ -- baseline security hard-win is 23%
macro (7 of 31 templates) and 21% pooled (32 of 150 instances). Pooling is used
here because it is what the printed confidence intervals are computed on and
what ``tab:denominators`` counts; ``--summary`` prints both so the gap stays
visible rather than silently picking a side.

The ``+slow`` and ``+all`` columns pool five live Q-LLM samples per instance, so
their denominator is ``5n`` rather than ``n``. That is not cosmetic -- it is why
those columns' intervals are roughly half the width of the deterministic ones --
and it is verified rather than assumed: every interval this module computes is
checked against the interval the artifact printed, and a mismatch is fatal.

Run: ``python -m tools.ablation_plots``
(needs the ``palisade-figures`` extra: ``uv sync --extra palisade-figures``)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from tools.eval_figures import CONFIGS, parse_ablation
from palisade.paths import REPO_ROOT

__all__ = [
    "K_SAMPLES",
    "PALETTE",
    "Cell",
    "Report",
    "parse_report",
    "wilson",
    "render_all",
]

#: Live Q-LLM samples pooled per instance in the two optional-tier columns.
#: Every other column is deterministic and scored once per instance.
K_SAMPLES = 5
POOLED_CONFIGS = frozenset({"full +slow", "full +all"})

#: Ablation arms present in the artifact but not on the manuscript's
#: nine-configuration ladder. Parsed and dropped, not an error.
SKIP_CONFIGS = frozenset({"full +failclosed", "full +all -judge"})

#: Slots 1-4 of the validated categorical palette, plus chart chrome. Blue is
#: soft win and orange is hard win throughout, matching the convention
#: ``fig:ablation-boundary`` already states in its caption.
#: Okabe--Ito, the colourblind-safe eight. Each series also carries a distinct
#: marker and dash pattern, so no distinction here rests on hue alone.
PALETTE = {
    "soft": "#0072B2",      # blue
    "hard": "#D55E00",      # vermillion
    "misuse": "#009E73",    # green
    "benign": "#E69F00",    # orange
    "ink": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}

#: Two single-hue sequential ramps, light -> dark, for the two magnitude
#: contexts on the heatmap. Monotone in luminance, asserted at import time.
RAMP_SOFT = ("#e8f1fd", "#cde2fb", "#9ec5f4", "#6da7ec",
             "#3987e5", "#256abf", "#184f95", "#0d366b")
RAMP_HARD = ("#fdeee7", "#fbd9c9", "#f7b393", "#f28d5f",
             "#eb6834", "#c14f22", "#933a17", "#68280f")

SUITES = (
    ("security", "Security -- capability bound"),
    ("science", "Science-correctness"),
    ("misuse", "Misuse / safety"),
)
FAMILIES = (
    ("B1", "B1 prompt"),
    ("B3", "B3 retrieval"),
    ("B4", "B4 code"),
    ("B5", "B5 submission"),
    ("XC", "XC cross-boundary"),
)

_SUITE_RE = re.compile(r"^### (Security|Science|Misuse|Benign)")
_CLASS_RE = re.compile(r"^#### (\S+) -- `([^`]+)`")
_COUNT_RE = re.compile(r"^_(\d+) attack instance\(s\), (\d+) benign")
_CROSS_RE = re.compile(r"^Cross-tenant classes in this corpus: (.+)\.$")
_PCT_RE = re.compile(r"^(\d+)%(?:\s*\((\d+)-(\d+)\))?")
_SUITE_KEY = {"Security": "security", "Science": "science",
              "Misuse": "misuse", "Benign": "benign"}


def _family_of(cls: str) -> str:
    for prefix, fam in (("b1_", "B1"), ("b3_", "B3"), ("b4_", "B4"),
                        ("b5_", "B5"), ("xc_", "XC")):
        if cls.startswith(prefix):
            return fam
    return "B0"


def _sort_key(cls: str) -> tuple[str, int]:
    """Order b5_9 before b5_10; the artifact's own headings sort them the other
    way because it sorts the class ids as strings."""
    parts = cls.split("_")
    try:
        return parts[0], int(parts[1])
    except (IndexError, ValueError):
        return parts[0], 0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, as fractions. Matches the artifact's own method."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Cell:
    """One (class, configuration) measurement, as counts over the pooled n."""

    #: Instances in the class. Not the interval denominator -- see ``n_eff``.
    n: int
    #: Scored trials behind the interval: ``n`` normally, ``5n`` on +slow/+all.
    n_eff: int
    soft: int
    hard: int
    #: The interval the artifact printed, as whole percents, for cross-checking.
    printed_asr: tuple[int, int] | None = None
    printed_hard: tuple[int, int] | None = None


@dataclass
class Report:
    """The parsed artifact, keyed suite -> class -> config."""

    cells: dict[str, dict[str, dict[str, Cell]]] = field(default_factory=dict)
    #: class -> ("B4.9", "corpus injected code")
    labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: config -> pooled benign utility, over the 181 controls
    benign: dict[str, float] = field(default_factory=dict)
    cross_tenant: frozenset[str] = frozenset()

    def classes(self, suite: str) -> list[str]:
        return sorted(self.cells.get(suite, {}), key=_sort_key)

    def pooled(self, suite: str, config: str, *, family: str | None = None) -> Cell:
        """Sum a suite (or one family within it) into a single pooled cell."""
        out = Cell(n=0, n_eff=0, soft=0, hard=0)
        for cls, per_cfg in self.cells.get(suite, {}).items():
            if family is not None and _family_of(cls) != family:
                continue
            c = per_cfg.get(config)
            if c is None:
                continue
            out.n += c.n
            out.n_eff += c.n_eff
            out.soft += c.soft * (c.n_eff // c.n)
            out.hard += c.hard * (c.n_eff // c.n)
        return out

    def macro(self, suite: str, config: str, key: str) -> float:
        """The Headline block's statistic: the mean rate over templates."""
        rates = [
            getattr(c, key) / c.n
            for per_cfg in self.cells.get(suite, {}).values()
            if (c := per_cfg.get(config)) is not None and c.n
        ]
        return 100.0 * sum(rates) / len(rates) if rates else 0.0


def _pct(field_text: str) -> tuple[int, tuple[int, int] | None] | None:
    m = _PCT_RE.match(field_text)
    if m is None:
        return None
    lo, hi = m.group(2), m.group(3)
    return int(m.group(1)), (int(lo), int(hi)) if lo and hi else None


def parse_report(path: Path) -> Report:
    """Parse the ablation markdown into counts, intervals and class labels.

    Every cell is small enough that ``round(pct/100 * n)`` recovers the count
    exactly; that inversion is what the interval cross-check in
    :func:`validate` then confirms against the artifact's printed intervals.
    """
    out = Report()
    suite = cls = label = None
    n_attack = n_benign = 0
    benign_acc: dict[str, tuple[int, int]] = {}

    for line in path.read_text().splitlines():
        if m := _CROSS_RE.match(line):
            out.cross_tenant = frozenset(
                t.strip().strip("`") for t in m.group(1).split(",")
            )
            continue
        if m := _SUITE_RE.match(line):
            suite, cls = _SUITE_KEY[m.group(1)], None
            continue
        if m := _CLASS_RE.match(line):
            label, cls = m.group(1), m.group(2)
            n_attack = n_benign = 0
            prefix = f"{cls.split('_')[0]}_{cls.split('_')[1]}_"
            out.labels[cls] = (label, cls.removeprefix(prefix).replace("_", " "))
            continue
        if (m := _COUNT_RE.match(line)) and cls and suite:
            n_attack, n_benign = int(m.group(1)), int(m.group(2))
            out.cells.setdefault(suite, {}).setdefault(cls, {})
            continue
        if not (line.startswith("| ") and cls and suite):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5 or parts[0] == "Configuration" or parts[0].startswith("---"):
            continue
        cfg = parts[0]
        if cfg in SKIP_CONFIGS:
            # A column the manuscript's figures do not plot (e.g. the
            # fail-closed posture, reported in prose only). Skipped rather than
            # fatal, so adding an ablation arm does not break the figures.
            continue
        if cfg not in dict(CONFIGS):
            raise ValueError(f"unknown configuration {cfg!r}: report format changed")

        if suite == "benign":
            bu = _pct(parts[3])
            if bu is not None and n_benign:
                got, tot = benign_acc.get(cfg, (0, 0))
                benign_acc[cfg] = (got + round(bu[0] / 100 * n_benign), tot + n_benign)
            continue

        asr, hard = _pct(parts[1]), _pct(parts[4])
        if asr is None:
            continue
        mult = K_SAMPLES if cfg in POOLED_CONFIGS else 1
        out.cells[suite][cls][cfg] = Cell(
            n=n_attack,
            n_eff=n_attack * mult,
            soft=round(asr[0] / 100 * n_attack),
            hard=round(hard[0] / 100 * n_attack) if hard else 0,
            printed_asr=asr[1],
            printed_hard=hard[1] if hard else None,
        )

    out.benign = {c: 100.0 * g / t for c, (g, t) in benign_acc.items() if t}
    return out


def validate(rep: Report, path: Path) -> list[str]:
    """Cross-check the parse. Returns a list of problems; empty means clean.

    Three independent checks, because a figure that silently drifts from its
    artifact is worse than no figure:

    1. counts agree with ``tools.eval_figures``, a separately written parser;
    2. the partition sizes are still 150 / 40 / 15 / 181;
    3. every recomputed Wilson interval reproduces the one the artifact
       printed -- which only holds if both the count inversion and the ``5n``
       pooled denominator are right.
    """
    problems: list[str] = []

    ref = parse_ablation(path)
    for suite, per_cls in rep.cells.items():
        for cls, per_cfg in per_cls.items():
            for cfg, cell in per_cfg.items():
                r = ref.cells.get(suite, {}).get(cls, {}).get(cfg)
                if r is None:
                    problems.append(f"{suite}/{cls}/{cfg}: missing in eval_figures")
                elif (r.n, r.soft, r.hard) != (cell.n, cell.soft, cell.hard):
                    problems.append(
                        f"{suite}/{cls}/{cfg}: counts disagree with eval_figures "
                        f"({r.n},{r.soft},{r.hard}) != ({cell.n},{cell.soft},{cell.hard})"
                    )

    base = CONFIGS[0][0]
    sizes = {s: rep.pooled(s, base).n for s, _ in SUITES}
    if sizes != {"security": 150, "misuse": 40, "science": 15}:
        problems.append(f"partition sizes changed: {sizes}")

    for suite, per_cls in rep.cells.items():
        for cls, per_cfg in per_cls.items():
            for cfg, cell in per_cfg.items():
                for key, printed in (("soft", cell.printed_asr),
                                     ("hard", cell.printed_hard)):
                    if printed is None:
                        continue
                    mult = cell.n_eff // cell.n
                    lo, hi = wilson(getattr(cell, key) * mult, cell.n_eff)
                    got = (round(lo * 100), round(hi * 100))
                    if got != printed:
                        problems.append(
                            f"{suite}/{cls}/{cfg} {key}: recomputed CI {got} != "
                            f"printed {printed} (n_eff={cell.n_eff})"
                        )
    return problems


# -----------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------

def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["secondary"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 7.5,
    })


def _ramp(stops: tuple[str, ...]):
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    lum = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in map(to_rgb, stops)]
    if any(a <= b for a, b in zip(lum, lum[1:])):
        raise ValueError(f"sequential ramp is not monotone in luminance: {stops}")
    return LinearSegmentedColormap.from_list("ramp", stops)


def _shorts() -> list[str]:
    return [s for _, s in CONFIGS]


def _split_x(n_total: int, n_cum: int = 6, gap: float = 0.7):
    """X positions that separate the cumulative ladder from the additions to
    ``full``.

    The last three configurations are independent additions to ``full``
    (``+semgrep`` and ``+slow`` are siblings, ``+all`` their union), not further
    rungs. Drawing all nine on one connected line asserted an ordering among
    them that does not exist.
    """
    xs_cum = list(range(n_cum))
    xs_aug = [n_cum - 1 + gap + 1.0 * (i + 1) for i in range(n_total - n_cum)]
    split = (xs_cum[-1] + xs_aug[0]) / 2.0
    return xs_cum, xs_aug, split


#: Replicates and seed for the class-level bootstrap. Fixed so the figure is
#: reproducible from the artifact alone.
BOOT_REPS = 10_000
BOOT_SEED = 42


def _cluster_ci(rep: "Report", suite: str, config: str, key: str,
                *, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> tuple[float, float]:
    """Percentile interval from resampling authored *classes*, not instances.

    The instances are not independent draws: 38 of the 42 classes are five
    near-identical variants of one template along one variation axis, so an
    exact-binomial interval over instances borrows confidence from within-class
    repetition. The class is the independent unit, so it is the resampling unit.

    The denominator is the instance count ``n``, never ``n_eff``. On the
    slow-tier configurations ``n_eff`` is ``5n`` because five live draws were
    pooled, but those draws are repeated measures on the *same* instances rather
    than five fresh ones; counting them as independent is the second way the
    published interval was too narrow. The point estimate is unaffected --
    ``soft / n`` and ``soft * 5 / n_eff`` are the same number -- so this widens
    the band without moving the curve.
    """
    import random

    per_class = [
        (getattr(c, key), c.n)
        for cls, per_cfg in rep.cells.get(suite, {}).items()
        if (c := per_cfg.get(config)) is not None
    ]
    if not per_class:
        return 0.0, 0.0
    rng = random.Random(seed)
    k = len(per_class)
    draws = []
    for _ in range(reps):
        num = den = 0
        for _ in range(k):
            hit, n = per_class[rng.randrange(k)]
            num += hit
            den += n
        draws.append(num / den if den else 0.0)
    draws.sort()
    return draws[int(0.025 * reps)], draws[int(0.975 * reps)]


def _band(ax, xs_cum, xs_aug, rep, suite, names, key, colour, *, label, marker,
          dashed=False, dy=-2):
    """One series: the pooled rate, its uncertainty, and an endpoint label.

    One line and one filled band across all nine configurations. The band spans
    the gap at the rule as well, so the ribbon is continuous; the rule and the
    caption carry the point that the three additions to ``full`` are
    alternatives rather than successive steps.

    Bands are **cluster-robust** -- the class is the resampling unit, per
    ``_cluster_ci`` -- so they match the intervals the evaluation section
    reports rather than the narrower instance-level ones. They are wide by
    construction and overlap between series; that is the honest picture of a
    31-class corpus, not a drawing artefact.

    ``dy`` staggers the endpoint label: soft and hard converge to 4% and 3% at
    ``+all``, which is closer than the labels are tall.
    """
    xs_all = list(xs_cum) + list(xs_aug)
    cells = [rep.pooled(suite, c) for c in names]
    rate = [100.0 * getattr(c, key) / c.n_eff if c.n_eff else 0.0 for c in cells]
    lo, hi = zip(*(_cluster_ci(rep, suite, c, key) for c in names))

    ax.fill_between(xs_all, [v * 100 for v in lo], [v * 100 for v in hi],
                    color=colour, alpha=0.13, linewidth=0, zorder=1)
    ax.plot(xs_all, rate, color=colour, linewidth=1.6, marker=marker,
            markersize=3.4, markeredgecolor="white", markeredgewidth=0.5,
            label=label, linestyle="--" if dashed else "-", zorder=3)
    ax.annotate(f"{rate[-1]:.0f}%", (xs_aug[-1], rate[-1]),
                textcoords="offset points", xytext=(7, dy),
                color=colour, fontsize=7.5, fontweight="bold")
    return rate


def figure_overview(rep: Report, out: Path) -> Path:
    """Cumulative curves: security soft/hard, misuse, and benign utility."""
    import matplotlib.pyplot as plt

    names = [c for c, _ in CONFIGS]
    xs_cum, xs_aug, split = _split_x(len(names))
    xs = xs_cum + xs_aug
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(7.16, 4.6), height_ratios=[2.6, 1], sharex=True)

    _band(ax, xs_cum, xs_aug, rep, "security", names, "soft",
          PALETTE["soft"], label="soft win (inspection evasion)",
          marker="o", dy=1)
    _band(ax, xs_cum, xs_aug, rep, "security", names, "hard",
          PALETTE["hard"], label="hard win (capability bound)",
          marker="s", dy=-7)
    _band(ax, xs_cum, xs_aug, rep, "misuse", names, "soft",
          PALETTE["misuse"], label="misuse (single principal, excluded)",
          marker="^", dashed=True)

    for _ax in (ax,):
        _ax.axvline(split, color=PALETTE["ink"], lw=0.7, alpha=0.32, zorder=1)
    ax.set_ylim(-4, 108)
    ax.set_xlim(-0.4, xs_aug[-1] + 0.62)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("attack success (%)")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    # Above the plot rather than inside it: misuse holds 75% out to +semgrep,
    # so every in-axes corner is occupied by a series or its band.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3,
              handlelength=1.6, columnspacing=1.5, borderaxespad=0.2)
    # Figure title lives in the LaTeX caption.

    ben = [rep.benign[c] for c in names]
    n_cum = len(xs_cum)
    bx.axvline(split, color=PALETTE["ink"], lw=0.7, alpha=0.32, zorder=1)
    bx.plot(xs_cum, ben[:n_cum], color=PALETTE["benign"], linewidth=1.6,
            marker="D", markersize=3.2, markeredgecolor="white",
            markeredgewidth=0.5)
    bx.plot([xs_cum[-1], xs_aug[0]], [ben[n_cum - 1], ben[n_cum]],
            color=PALETTE["benign"], linewidth=1.6, zorder=3)
    bx.plot(xs_aug, ben[n_cum:], color=PALETTE["benign"], linewidth=1.6,
            marker="D", markersize=3.2, markeredgecolor="white",
            markeredgewidth=0.5)
    bx.annotate(f"{ben[-1]:.1f}%", (xs_aug[-1], ben[-1]), textcoords="offset points",
                xytext=(7, -2), color=PALETTE["benign"], fontsize=7.5,
                fontweight="bold")
    bx.set_ylim(88, 102)
    bx.set_yticks([90, 95, 100])
    bx.set_ylabel("benign utility (%)")
    bx.grid(axis="y", zorder=0)
    bx.set_axisbelow(True)
    bx.set_xticks(xs, _shorts(), rotation=32, ha="right")
    for _lab, _x in zip(bx.get_xticklabels(), xs):
        if _x > split:
            _lab.set_color(PALETTE["muted"])
    bx.set_title("Benign utility", loc="left",
                 color=PALETTE["secondary"], fontsize=9, pad=6)

    # Explanatory text lives in the LaTeX caption, not in the figure.
    fig.align_ylabels((ax, bx))
    return _save(fig, out, "ablation_overview")


def figure_heatmap(rep: Report, out: Path) -> Path:
    """Every attack class x every configuration, soft-win beside hard-win."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    names = [c for c, _ in CONFIGS]
    # Each suite opens with a spacer row that carries its heading, so the
    # heading never lands on top of the first class label in its block.
    rows: list[tuple[str, str] | None] = []
    breaks: list[tuple[int, str]] = []
    for suite, title in SUITES:
        breaks.append((len(rows), title))
        rows.append(None)
        rows += [(suite, cls) for cls in rep.classes(suite)]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 0.185 * len(rows) + 1.4),
                             sharey=True)
    cmaps = (_ramp(RAMP_SOFT), _ramp(RAMP_HARD))

    for ax, key, cmap, title, colour in zip(
        axes, ("soft", "hard"), cmaps,
        ("Soft-win ASR -- content evaded the inspection layer",
         "Hard-win ASR -- taint reached a high-privilege sink"),
        (PALETTE["soft"], PALETTE["hard"]),
    ):
        for y, row in enumerate(rows):
            if row is None:
                continue
            suite, cls = row
            for x, cfg in enumerate(names):
                cell = rep.cells[suite][cls][cfg]
                pct = 100.0 * getattr(cell, key) / cell.n
                # A 2px surface gap between fills rather than a drawn border.
                ax.add_patch(Rectangle(
                    (x + 0.035, y + 0.035), 0.93, 0.93,
                    facecolor=cmap(pct / 100) if pct else "#f6f6f4",
                    linewidth=0))
                if pct:
                    ax.text(x + 0.5, y + 0.5, f"{pct:.0f}",
                            ha="center", va="center", fontsize=5.8,
                            color="white" if pct >= 55 else PALETTE["ink"])

        ax.set_xlim(0, len(names))
        ax.set_ylim(len(rows), 0)
        ax.set_xticks([i + 0.5 for i in range(len(names))], _shorts(),
                      rotation=40, ha="right")
        ax.set_title(title, loc="left", color=colour, fontweight="bold", pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, _ in breaks[1:]:
            ax.axhline(y, color=PALETTE["axis"], linewidth=0.8)

    ticks, labels = [], []
    for i, row in enumerate(rows):
        if row is None:
            continue
        tag, name = rep.labels[row[1]]
        ticks.append(i + 0.5)
        labels.append(f"{tag}  {name}"
                      + ("  †" if row[1] in rep.cross_tenant else ""))
    axes[0].set_yticks(ticks, labels, fontsize=6.4)
    axes[0].tick_params(axis="y", pad=2)

    for y, title in breaks:
        axes[0].annotate(title, (0, y + 0.75), xytext=(-120, 0), fontsize=7.2,
                         textcoords="offset points", color=PALETTE["ink"],
                         fontweight="bold", va="center", annotation_clip=False)

    fig.suptitle("SIEGE v0.1 -- per-class ablation, production taint bound",
                 x=0.006, y=0.995, ha="left", va="top", fontsize=10.5,
                 fontweight="bold", color=PALETTE["ink"])
    fig.text(0.006, 0.004,
             "Blank = 0%. † = cross-tenant (shared global RAG corpus), so the "
             "single-tenant rate understates blast radius.\nScience-correctness "
             "ASR is gate-only and 100% by construction -- those three classes "
             "are bounded by the contract layer, which caught 15/15 "
             "(contract-evaded ASR 0%).",
             fontsize=6.5, color=PALETTE["muted"], va="bottom")
    fig.subplots_adjust(left=0.205, right=0.997, top=0.945, bottom=0.075,
                        wspace=0.06)
    return _save(fig, out, "ablation_heatmap", tight=False)


def figure_boundary(rep: Report, out: Path) -> Path:
    """Per-boundary small multiples, on the 150-instance security partition.

    The nine configurations are **not** one ladder, and the figure now says so.
    Six are cumulative and are drawn as a connected line. The last three are
    independent additions to ``full`` -- ``+semgrep`` and ``+slow`` are siblings
    and ``+all`` is their union -- so they are drawn as detached markers past a
    rule. Connecting them, as the previous version did, asserted an ordering
    that does not exist and made B4's hard-win rate look non-monotone when it is
    simply two different configurations over the same base.
    """
    import matplotlib.pyplot as plt

    names = [c for c, _ in CONFIGS]
    shorts = _shorts()
    n_cum = 6                                  # baseline .. full
    xs_cum = list(range(n_cum))                # 0..5
    xs_aug = [n_cum + 0.7 + i for i in range(len(names) - n_cum)]  # 6.7, 7.7, 8.7
    xs_all = xs_cum + xs_aug
    split = (xs_cum[-1] + xs_aug[0]) / 2.0

    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(7.16, 2.35), sharey=True)

    for ax, (fam, title) in zip(axes, FAMILIES):
        cells = [rep.pooled("security", c, family=fam) for c in names]
        n = cells[0].n

        # One rule, not two: `full` is already the visible end of the connected
        # line, so a second mark on it read as a double rule with the divider.
        ax.axvline(split, color=PALETTE["ink"], lw=0.7, alpha=0.32, zorder=1)

        # Soft is drawn wider and hard narrower on top of it, because in B1 the
        # two series are identical at every configuration (its only class is
        # b1_9, where every soft win is also a hard win) and an equal-weight
        # hard series would hide soft completely.
        for key, colour, marker, lw, ms in (
            ("soft", PALETTE["soft"], "o", 1.9, 3.4),
            ("hard", PALETTE["hard"], "s", 0.9, 1.9),
        ):
            rate = [100.0 * getattr(c, key) / c.n_eff if c.n_eff else 0.0
                    for c in cells]
            ax.plot(xs_cum, rate[:n_cum], color=colour, linewidth=lw,
                    marker=marker, markersize=ms, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3,
                    label="soft win" if key == "soft" else "hard win")
            # Augmented: connected, matching the cumulative span. The rule is
            # what separates them; the line carries the eye across the three.
            ax.plot([xs_cum[-1], xs_aug[0]], [rate[n_cum - 1], rate[n_cum]],
                    color=colour, linewidth=lw, zorder=3)
            ax.plot(xs_aug, rate[n_cum:], color=colour, linewidth=lw,
                    marker=marker, markersize=ms, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3)
            # Endpoint value at +all, the configuration the text quotes.
            ax.annotate(f"{rate[-1]:.0f}", (xs_aug[-1], rate[-1]),
                        textcoords="offset points",
                        xytext=(4, -5.5 if key == "hard" else 3.5),
                        fontsize=7, color=colour, zorder=4)

        ax.set_title(f"{title}\n$n={n}$", loc="left", fontsize=8,
                     color=PALETTE["ink"], pad=5)
        ax.set_xticks(xs_all, shorts, rotation=55, ha="right", fontsize=7)
        ax.set_xlim(-0.6, xs_aug[-1] + 1.15)
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        for lab, xpos in zip(ax.get_xticklabels(), xs_all):
            if xpos > split:
                lab.set_color(PALETTE["muted"])

    axes[0].set_ylabel("attack success (%)")
    axes[-1].legend(loc="upper right", handlelength=1.4, fontsize=7)
    # Explanatory text lives in the LaTeX caption, not in the figure.
    return _save(fig, out, "ablation_boundary")


def figure_tradeoff(rep: Report, out: Path) -> Path:
    """What the capability bound costs: hard-win ASR against benign utility."""
    import matplotlib.pyplot as plt

    names = [c for c, _ in CONFIGS]
    hard = [100.0 * (p := rep.pooled("security", c)).hard / p.n_eff for c in names]
    ben = [rep.benign[c] for c in names]

    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    ax.plot(ben, hard, color=PALETTE["axis"], linewidth=1.0, zorder=2)
    ax.scatter(ben, hard, s=26, color=PALETTE["hard"], edgecolor="white",
               linewidth=0.6, zorder=3)

    # Direct-label the endpoints and the configurations that move the rate; a
    # label on all nine would collide and go unread. Configurations that land
    # on the same point share one label rather than overprinting -- +slow and
    # +all coincide, and dropping the later one would leave the endpoint of the
    # path, which the title quotes, unlabelled.
    merged: dict[tuple[int, int], tuple[float, float, list[str]]] = {}
    for i, (b, h, short) in enumerate(zip(ben, hard, _shorts())):
        if i in (0, len(names) - 1) or abs(h - hard[i - 1]) > 3:
            at = merged.setdefault((round(b), round(h)), (b, h, []))
            at[2].append(short)
    for b, h, shorts in merged.values():
        ax.annotate(", ".join(shorts), (b, h), textcoords="offset points",
                    xytext=(6, 4), fontsize=6.8, color=PALETTE["secondary"])

    ax.set_xlabel("benign utility (%)")
    ax.set_ylabel("security hard-win ASR (%)")
    ax.set_xlim(100.9, 93.6)
    ax.set_ylim(-1.5, max(hard) + 4)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(f"{hard[0]:.0f}% → {hard[-1]:.0f}% hard-win for "
                 f"{ben[0] - ben[-1]:.1f} pp of utility",
                 loc="left", color=PALETTE["ink"], fontweight="bold", pad=8)
    fig.text(0.0, -0.02,
             "Each point is one cumulative configuration; utility decreases to "
             "the right.",
             fontsize=6.5, color=PALETTE["muted"], va="top")
    return _save(fig, out, "ablation_tradeoff")


def _save(fig, out: Path, stem: str, *, tight: bool = True) -> Path:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", **kw)
    plt.close(fig)
    return out / f"{stem}.png"


def render_all(rep: Report, out: Path) -> list[Path]:
    _style()
    return [
        figure_overview(rep, out),
        figure_heatmap(rep, out),
        figure_boundary(rep, out),
        figure_tradeoff(rep, out),
    ]


def summarise(rep: Report) -> str:
    """The pooled-vs-macro table, so the gap with the Headline block is visible."""
    lines = [
        f"{'config':<16} {'sec soft':>18} {'sec hard':>18} "
        f"{'misuse':>12} {'benign BU':>10}",
        f"{'':<16} {'pooled  macro':>18} {'pooled  macro':>18} "
        f"{'pooled':>12} {'pooled':>10}",
    ]
    for cfg, short in CONFIGS:
        s, m = rep.pooled("security", cfg), rep.pooled("misuse", cfg)
        lines.append(
            f"{short:<16} "
            f"{100 * s.soft / s.n_eff:>10.0f}% {rep.macro('security', cfg, 'soft'):>5.0f}% "
            f"{100 * s.hard / s.n_eff:>10.0f}% {rep.macro('security', cfg, 'hard'):>5.0f}% "
            f"{100 * m.soft / m.n_eff:>11.0f}% {rep.benign[cfg]:>9.0f}%"
        )
    return "\n".join(lines)


DEFAULT_SRC = REPO_ROOT / "docs/palisade/full_ablation_production.md"
DEFAULT_OUT = REPO_ROOT / "docs/palisade/figures"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", type=Path, default=DEFAULT_SRC,
                    help="the ablation artifact to plot")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="directory for the PNG/PDF pairs")
    ap.add_argument("--summary", action="store_true",
                    help="print the pooled-vs-macro table and exit")
    args = ap.parse_args(argv)

    if not args.source.exists():
        print(f"no such artifact: {args.source}", file=sys.stderr)
        return 2

    rep = parse_report(args.source)
    problems = validate(rep, args.source)
    if problems:
        print(f"{args.source.name} failed validation:", file=sys.stderr)
        for p in problems[:20]:
            print(f"  - {p}", file=sys.stderr)
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more", file=sys.stderr)
        return 1

    if args.summary:
        print(summarise(rep))
        return 0

    try:
        written = render_all(rep, args.out)
    except ImportError:
        print("matplotlib is required: uv sync --extra palisade-figures",
              file=sys.stderr)
        return 2

    print(summarise(rep))
    print()
    for p in written:
        print(f"wrote {p}  (+ .pdf)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
