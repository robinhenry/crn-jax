"""Render benchmark plots from saved correctness/throughput data.

Outputs (all to figures/):
  correctness_<model>.png      — marginal-distribution histograms (crn-jax /
                                 GillesPy2 overlaid) at T=20.
  trajectories_<model>.png     — overlaid sample trajectories per library.
  throughput_speedup.png       — bar plot at fixed N: GillesPy2 CPU →
                                 crn-jax CPU → crn-jax GPU. Headline figure.
  throughput_scaling.png       — log-log throughput vs N, showing GillesPy2's
                                 ceiling vs crn-jax GPU's reach to 1M+.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Series style: GillesPy2 baseline = blue; crn-jax = red (pop) for CPU,
# darker red for GPU.
COLORS = {
    ("gillespy2", "cpu"): "#1f77b4",
    ("crn_jax", "cpu"):   "#ff7f0e",
    ("crn_jax", "gpu"):   "#d62728",
}
LABELS = {
    ("gillespy2", "cpu"): "GillesPy2 (CPU, C++)",
    ("crn_jax", "cpu"):   "crn-jax (CPU)",
    ("crn_jax", "gpu"):   "crn-jax (GPU)",
}
SHORT_LABELS = {
    ("gillespy2", "cpu"): "GillesPy2\nCPU",
    ("crn_jax", "cpu"):   "crn-jax\nCPU",
    ("crn_jax", "gpu"):   "crn-jax\nGPU",
}
SPECIES_LABELS = {
    "birth_death": ["X"],
    "linear_cascade": [f"A{i+1}" for i in range(10)],
}
MODEL_TITLES = {
    "birth_death": "Birth-death (1 species, 2 reactions)",
    "linear_cascade": "Linear cascade (10 species, 20 reactions)",
}
MODELS = ["birth_death", "linear_cascade"]


# ---------- correctness ----------------------------------------------------
def plot_correctness(model_name: str):
    path = RESULTS_DIR / f"correctness_{model_name}.npz"
    if not path.exists():
        print(f"  skip {model_name}: {path.name} not found")
        return
    data = np.load(path)
    species_names = SPECIES_LABELS[model_name]
    n_species = data["crn_jax"].shape[1]

    n_cols = min(n_species, 5)
    n_rows = (n_species + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)
    axes = axes.flatten()

    crn_color = COLORS[("crn_jax", "gpu")]
    ref_color = COLORS[("gillespy2", "cpu")]

    for i in range(n_species):
        ax = axes[i]
        all_vals = np.concatenate([data["crn_jax"][:, i], data["gillespy2"][:, i]])

        hi = all_vals.max()
        lo = all_vals.min()

        span = max(1, int(hi - lo))
        if span <= 50:
            bins = np.arange(int(lo) - 0.5, int(hi) + 1.5, 1.0)
        else:
            bins = np.linspace(lo, hi, 50)

        for lib, color, label in [
            ("gillespy2", ref_color, "GillesPy2"),
            ("crn_jax", crn_color, "crn-jax"),
        ]:
            v = data[lib][:, i]
            v = v[(v >= bins[0]) & (v <= bins[-1])]
            ax.hist(v, bins=bins, alpha=0.5, color=color, label=label, density=True)
        ax.set_title(f"{species_names[i]} at T=20")
        ax.set_xlabel("count")
        ax.set_ylabel("density")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    for j in range(n_species, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Marginal distributions at T=20 — {model_name} (N=10 000)", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / f"correctness_{model_name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- trajectory comparison ------------------------------------------
def plot_trajectories_comparison(model_name: str, n_traj: int = 5):
    from benchmarks.models import get
    model = get(model_name)
    species_names = SPECIES_LABELS[model_name]

    print(f"  generating {n_traj} trajectories for {model_name}…", flush=True)
    runs = {
        "crn_jax":   np.asarray(model.run_crn_jax(jax.random.PRNGKey(7), n_traj, return_full_trajectory=True)),
        "gillespy2": np.asarray(model.run_gillespy2(42, n_traj, return_full_trajectory=True)),
    }

    n_species = len(species_names)

    def _normalise(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if arr.ndim == 2:
            arr = arr[..., None]
        n_steps = arr.shape[1]
        if n_steps == 200:  # crn-jax (no t=0)
            times = np.linspace(0.1, 20.0, 200)
        else:
            times = np.linspace(0.0, 20.0, n_steps)
        return times, arr

    if model_name == "linear_cascade":
        idxs = [0, 4, 9]
        labels = [species_names[i] for i in idxs]
    else:
        idxs = list(range(n_species))
        labels = species_names

    n_cols = len(idxs)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.2), squeeze=False)
    axes = axes[0]

    for col_i, sp_idx in enumerate(idxs):
        ax = axes[col_i]
        for lib, color, label in [
            ("gillespy2", COLORS[("gillespy2", "cpu")], "GillesPy2"),
            ("crn_jax",   COLORS[("crn_jax", "gpu")],   "crn-jax"),
        ]:
            times, vals = _normalise(runs[lib])
            for t in range(n_traj):
                ax.step(
                    times, vals[t, :, sp_idx],
                    color=color, alpha=0.6, lw=1.0, where="post",
                    label=label if t == 0 else None,
                )
        ax.set_title(f"{labels[col_i]} — {n_traj} trajectories per library")
        ax.set_xlabel("time")
        ax.set_ylabel("count")
        if col_i == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Sample trajectories — {model_name.replace('_', ' ').title()}", fontsize=11)
    fig.tight_layout()
    out = FIGURES_DIR / f"trajectories_{model_name}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- throughput ------------------------------------------------------
def _load_throughput() -> dict[tuple[str, str], list[dict]]:
    """Returns {(lib, device): [rows...]} from results/throughput_<device>.json."""
    out = {}
    for device in ("cpu", "gpu"):
        path = RESULTS_DIR / f"throughput_{device}.json"
        if not path.exists():
            print(f"  skip {device}: {path.name} not found")
            continue
        rows = [r for r in json.loads(path.read_text()) if "error" not in r]
        for lib in ("crn_jax", "gillespy2"):
            sub = sorted([r for r in rows if r["lib"] == lib], key=lambda r: r["n"])
            if sub:
                out[(lib, device)] = sub
    return out


def _style_axes(ax):
    """Shared axis styling: drop top/right spines, softer ticks, grid below."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.set_axisbelow(True)


def plot_speedup_bars(rows_by_series):
    """Ruff-README-style horizontal bar chart, exported as SVG.

    Matches the visual style of
    https://github.com/astral-sh/ruff/blob/main/scripts/benchmarks/graph-spec.json:
    single panel, one row per (tool, model), uniform ``#E15759`` bars,
    bold hero rows, very thin bars, light vertical gridlines, transparent
    background, Apple-system font stack.
    """
    import re
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    series_order = [("crn_jax", "gpu"), ("gillespy2", "cpu")]
    BAR_COLOR = "#E15759"
    TEXT_LIGHT = "#333333"
    TEXT_DARK = "#c9d1d9"
    GRID_COLOR = (127 / 255, 127 / 255, 127 / 255, 0.25)

    # Build ordered rows: (label, seconds, is_hero).
    rows_data: list[tuple[str, float, bool]] = []
    for model in MODELS:
        peaks = {}
        for s in series_order:
            all_r = [r for r in rows_by_series.get(s, []) if r["model"] == model]
            if all_r:
                peaks[s] = max(all_r, key=lambda r: r["traj_per_s"])
        if len(peaks) != len(series_order):
            continue
        model_friendly = model.replace("_", " ")
        for s in series_order:
            seconds = 1_000_000 / peaks[s]["traj_per_s"]
            tool_short = "crn-jax (GPU)" if s[0] == "crn_jax" else "GillesPy2 (CPU)"
            rows_data.append((
                f"{tool_short}  ·  {model_friendly}",
                seconds,
                s[0] == "crn_jax",
            ))

    if not rows_data:
        return

    values = [r[1] for r in rows_data]
    labels = [r[0] for r in rows_data]
    heroes = [r[2] for r in rows_data]
    max_v = max(values)

    def _fmt(v: float) -> str:
        if v < 1:
            return f"{v:.2f}s"
        if v < 10:
            return f"{v:.1f}s"
        return f"{v:.0f}s"

    # Force matplotlib to emit <text> elements (not paths) in the SVG so we
    # can swap the font-family afterwards.
    with plt.rc_context({
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    }):
        fig, ax = plt.subplots(figsize=(9.0, 0.45 * len(rows_data) + 0.6))
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        ys = np.arange(len(rows_data))
        ax.barh(
            ys, values,
            height=0.42,
            color=BAR_COLOR,
            edgecolor="none",
            zorder=3,
        )

        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        for lbl, hero in zip(ax.get_yticklabels(), heroes):
            lbl.set_fontsize(12)
            lbl.set_color(TEXT_LIGHT)
            lbl.set_fontweight("bold" if hero else "normal")
        ax.invert_yaxis()

        for y_pos, v, hero in zip(ys, values, heroes):
            ax.text(
                v, y_pos, f"  {_fmt(v)}",
                ha="left", va="center",
                fontsize=12, color=TEXT_LIGHT,
                fontweight="bold" if hero else "normal",
            )

        ax.set_xlim(0, max_v * 1.12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}s"))

        ax.grid(True, axis="x", which="major", color=GRID_COLOR, lw=1.0, zorder=0)
        ax.grid(False, axis="y")

        for spine in ("top", "right", "left", "bottom"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=10)
        ax.tick_params(axis="x", length=0, pad=6, labelsize=12, colors=TEXT_LIGHT)
        ax.set_axisbelow(True)

        fig.tight_layout()

        out = FIGURES_DIR / "throughput_speedup.svg"
        fig.savefig(out, format="svg", bbox_inches="tight", transparent=True)
        plt.close(fig)

    # Post-process: (1) GitHub's markdown sanitiser rejects SVGs that declare
    # an external DTD, so drop the DOCTYPE and RDF metadata block; (2) drop
    # the ``pt`` unit from width/height (VS Code's preview mis-sizes them);
    # (3) swap matplotlib's font-family for the ruff Apple-system stack.
    # Then emit a dark-mode sibling by switching the text fill colour.
    ruff_font = (
        '-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,'
        'sans-serif,"Apple Color Emoji","Segoe UI Emoji"'
    )
    svg = out.read_text()
    svg = re.sub(r'<!DOCTYPE[^>]*>\s*', "", svg)
    svg = re.sub(r"<metadata>.*?</metadata>\s*", "", svg, flags=re.DOTALL)
    svg = re.sub(r'(width|height)="([\d.]+)pt"', r'\1="\2"', svg)
    svg = re.sub(r'font-family:[^;"\']+', f"font-family:{ruff_font}", svg)
    svg = re.sub(
        r'font-family="[^"]*"', f'font-family="{ruff_font}"', svg,
    )
    out.write_text(svg)
    print(f"  wrote {out.name}")

    dark_out = FIGURES_DIR / "throughput_speedup_dark.svg"
    dark_out.write_text(svg.replace(TEXT_LIGHT, TEXT_DARK))
    print(f"  wrote {dark_out.name}")

    # Remove the stale PNG if it exists so the README only has one source.
    old_png = FIGURES_DIR / "throughput_speedup.png"
    if old_png.exists():
        old_png.unlink()


def plot_throughput_scaling(rows_by_series):
    """Log-log throughput vs N, one panel per model.

    Shows that GillesPy2 saturates near a per-trajectory cost while
    crn-jax GPU keeps climbing until vmap saturates the device or memory runs
    out (≥1M trajectories on the simpler models).
    """
    series_order = [("gillespy2", "cpu"), ("crn_jax", "cpu"), ("crn_jax", "gpu")]

    fig, axes = plt.subplots(
        1, len(MODELS), figsize=(4.6 * len(MODELS), 3.8), squeeze=False,
    )
    axes = axes[0]

    for ax, model in zip(axes, MODELS):
        for s in series_order:
            rows = [r for r in rows_by_series.get(s, []) if r["model"] == model]
            rows.sort(key=lambda r: r["n"])
            if not rows:
                continue
            ns = [r["n"] for r in rows]
            tps = [r["traj_per_s"] for r in rows]
            ax.plot(
                ns, tps, "o-",
                color=COLORS[s], label=LABELS[s],
                lw=2.2, markersize=6,
                markeredgecolor="white", markeredgewidth=1.0,
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(MODEL_TITLES[model], fontsize=11, pad=10)
        ax.set_xlabel("# parallel trajectories", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("trajectories / sec  (T=20)", fontsize=10)
            ax.legend(
                loc="upper left", fontsize=9,
                frameon=True, framealpha=0.9, edgecolor="#cccccc",
            )
        ax.grid(
            True, which="major",
            linestyle=":", color="#bbbbbb", alpha=0.7, zorder=0,
        )
        ax.grid(False, which="minor")
        _style_axes(ax)

    fig.suptitle(
        "Throughput scaling — crn-jax (GPU) reaches batch sizes that are infeasible on CPU",
        fontsize=13, y=0.99,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "throughput_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- main ------------------------------------------------------------
def main():
    print("Correctness plots:")
    for m in MODELS:
        plot_correctness(m)
    print("Trajectory comparison plots:")
    for m in MODELS:
        plot_trajectories_comparison(m, n_traj=5)
    print("Throughput plots:")
    rows = _load_throughput()
    if rows:
        plot_speedup_bars(rows)
        plot_throughput_scaling(rows)


if __name__ == "__main__":
    main()
