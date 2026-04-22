"""Plotting helpers for stochastic reaction-network trajectories.

``matplotlib`` is an *optional* dependency of ``crn_jax`` (it lives under the
``examples`` extra), so it is imported lazily inside each function. Installing
with ``pip install crn-jax[examples]`` pulls it in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

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
    if title is not None:
        ax.set_title(title)
    return fig, ax
