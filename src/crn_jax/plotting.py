"""Plotting helpers for stochastic reaction-network trajectories.

``matplotlib`` is an *optional* dependency of ``crn_jax`` (it lives under the
``examples`` extra), so it is imported lazily inside each function. Installing
with ``pip install crn-jax[examples]`` pulls it in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from .types import Dataset

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_trajectories(
    times: Array,
    xs: Array,
    *,
    ax: "Axes | None" = None,
    xlabel: str = "time",
    ylabel: str = "x",
    title: str | None = None,
    color: str = "tab:blue",
    alpha: float | None = None,
) -> "tuple[Figure, Axes]":
    """Plot one or more Gillespie trajectories sharing a time axis.

    Uses step plotting (``where="post"``) to faithfully reflect the
    piecewise-constant nature of discrete molecule counts between reactions.

    Args:
        times: ``(T,)`` shared time axis.
        xs: ``(T,)`` single trajectory or ``(N, T)`` ensemble of ``N``
            trajectories.
        ax: Existing matplotlib axes to draw into. If ``None``, a new figure
            and axes are created.
        xlabel, ylabel, title: Axis labels.
        color: Line colour.
        alpha: Per-trajectory alpha. Defaults to ``1.0`` for a single
            trajectory and ``max(0.2, 1/N)`` for an ensemble.

    Returns:
        ``(fig, ax)`` — the matplotlib ``Figure`` and ``Axes`` the
        trajectories were drawn into. When ``ax`` is supplied, ``fig`` is
        ``ax.figure``.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 — lazy optional dep

    t = jnp.asarray(times)
    x = jnp.asarray(xs)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"xs must be 1D or 2D, got shape {x.shape}")
    n, _ = x.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    if alpha is None:
        alpha = 1.0 if n == 1 else max(0.2, 1.0 / n)

    for trj in x:
        ax.step(t, trj, where="post", color=color, alpha=alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)  # species counts are non-negative
    if title is not None:
        ax.set_title(title)
    return fig, ax


def plot_species_trajectories(
    dataset: Dataset,
    *,
    axes: "Sequence[Axes] | None" = None,
    title: str | None = None,
    colors: Sequence[str] | None = None,
    alpha: float | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot a multi-species ensemble as one step-plot subplot per species.

    Each subplot shows every replicate's trajectory for a single species,
    sharing the time axis vertically. For an ``S``-species dataset this
    produces ``S`` stacked subplots, labelled by ``dataset.species``.

    Args:
        dataset: Output of :func:`sample_trajectories`. Uses ``dataset.times``
            (``(T,)``), ``dataset.xs`` (``(N, T, S)``), and ``dataset.species``.
        axes: Existing length-``S`` sequence of axes to draw into. If
            ``None``, a new figure with ``S`` vertically stacked subplots is
            created.
        title: Optional figure-level title (``fig.suptitle``).
        colors: Per-species colours, length ``S``. Defaults to the matplotlib
            ``tab:`` cycle.
        alpha: Per-trajectory alpha (forwarded to :func:`plot_trajectories`).
        figsize: Figure size when creating new axes. Defaults to
            ``(7, 1.8 * S)``.

    Returns:
        ``(fig, axes)`` — the figure and a length-``S`` ``np.ndarray`` of
        axes (one per species, in ``dataset.species`` order).
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 — lazy optional dep

    xs = dataset.xs
    if xs.ndim != 3:
        raise ValueError(f"dataset.xs must be 3D (n_replicates, n_steps, n_species), got shape {xs.shape}")
    n_species = xs.shape[-1]
    if len(dataset.species) != n_species:
        raise ValueError(f"dataset.species has {len(dataset.species)} names but dataset.xs has {n_species} species")

    if axes is None:
        if figsize is None:
            figsize = (7, 1.8 * n_species)
        fig, axes_arr = plt.subplots(n_species, 1, figsize=figsize, sharex=True, squeeze=False)
        axes_arr = axes_arr[:, 0]
    else:
        if len(axes) != n_species:
            raise ValueError(f"axes has length {len(axes)} but dataset has {n_species} species")
        axes_arr = np.asarray(axes)
        fig = axes_arr[0].figure

    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
        colors = [cycle[i % len(cycle)] for i in range(n_species)]
    elif len(colors) != n_species:
        raise ValueError(f"colors has length {len(colors)} but dataset has {n_species} species")

    for i, (name, ax, color) in enumerate(zip(dataset.species, axes_arr, colors)):
        plot_trajectories(
            dataset.times,
            xs[:, :, i],
            ax=ax,
            xlabel="time" if i == n_species - 1 else "",
            ylabel=name,
            color=color,
            alpha=alpha,
        )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes_arr


def plot_species_distributions(
    dataset: Dataset,
    *,
    axes: "Sequence[Axes] | None" = None,
    title: str | None = None,
    n_bins: int = 50,
    cmap: str = "viridis",
    colorbar: bool = True,
    figsize: tuple[float, float] | None = None,
) -> "tuple[Figure, np.ndarray]":
    """Plot the marginal distribution of each species over time, as a heatmap.

    For each species, time is on the x-axis and species count on the y-axis;
    colour encodes the marginal-distribution density at each time slice,
    normalised independently per time step (so each column's most-occupied
    bin saturates the colormap). This keeps the shape of the marginal
    visible at every t even when one column is much more concentrated than
    the others (e.g. a delta initial condition vs. a relaxed stationary
    distribution).

    Args:
        dataset: Output of :func:`sample_trajectories`. Uses ``dataset.times``,
            ``dataset.xs`` (``(N, T, S)``), and ``dataset.species``.
        axes: Existing length-``S`` sequence of axes to draw into. If ``None``,
            a new figure with ``S`` vertically stacked subplots is created.
        title: Optional figure-level title (``fig.suptitle``).
        n_bins: Number of count bins per species. Default 50.
        cmap: Matplotlib colormap name.
        colorbar: If ``True``, attach a colorbar to each species subplot.
            Colour values are fractions of each column's max density (∈ [0, 1]).
        figsize: Figure size when creating new axes. Defaults to
            ``(7, 1.8 * S)``.

    Returns:
        ``(fig, axes)`` — the figure and a length-``S`` ``np.ndarray`` of
        axes (one per species, in ``dataset.species`` order).
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 — lazy optional dep

    xs = dataset.xs
    if xs.ndim != 3:
        raise ValueError(f"dataset.xs must be 3D (n_replicates, n_steps, n_species), got shape {xs.shape}")
    n_species = xs.shape[-1]
    if len(dataset.species) != n_species:
        raise ValueError(f"dataset.species has {len(dataset.species)} names but dataset.xs has {n_species} species")
    n_replicates, n_steps, _ = xs.shape

    if axes is None:
        if figsize is None:
            figsize = (7, 1.8 * n_species)
        fig, axes_arr = plt.subplots(n_species, 1, figsize=figsize, sharex=True, squeeze=False)
        axes_arr = axes_arr[:, 0]
    else:
        if len(axes) != n_species:
            raise ValueError(f"axes has length {len(axes)} but dataset has {n_species} species")
        axes_arr = np.asarray(axes)
        fig = axes_arr[0].figure

    times = np.asarray(dataset.times)
    t0, t1 = float(times[0]), float(times[-1])
    for i, (name, ax) in enumerate(zip(dataset.species, axes_arr)):
        data = np.asarray(xs[:, :, i])  # (n_replicates, n_steps)
        x_min = float(data.min())
        x_max = float(data.max())
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5

        # If the species is integer-valued and the range is small, snap the
        # bins to integers so the heatmap rows align with discrete counts;
        # otherwise spread `n_bins` linearly across the observed range.
        span = x_max - x_min
        integer_valued = np.allclose(data, np.round(data))
        if integer_valued and span + 1 <= n_bins:
            lo, hi = int(round(x_min)), int(round(x_max))
            bin_edges = np.arange(lo - 0.5, hi + 1.0)
        else:
            bin_edges = np.linspace(x_min, x_max, n_bins + 1)
        y_lo, y_hi = float(bin_edges[0]), float(bin_edges[-1])
        n_bins_eff = len(bin_edges) - 1

        # Per-time-step histogram → (n_bins, n_steps), then normalise each
        # column by its own max so the marginal's shape is visible at every
        # t (otherwise a single tightly-concentrated column — e.g. a delta
        # initial condition — sets the global colormap range and washes the
        # rest of the plot out).
        hist = np.empty((n_bins_eff, n_steps), dtype=np.float64)
        for t in range(n_steps):
            counts, _ = np.histogram(data[:, t], bins=bin_edges)
            hist[:, t] = counts
        col_max = hist.max(axis=0, keepdims=True)
        hist /= np.where(col_max > 0, col_max, 1.0)

        im = ax.imshow(
            hist,
            aspect="auto",
            origin="lower",
            extent=[t0, t1, y_lo, y_hi],
            cmap=cmap,
            interpolation="nearest",
        )
        ax.set_ylabel(name)
        if i == n_species - 1:
            ax.set_xlabel("time")
        if colorbar:
            fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes_arr
