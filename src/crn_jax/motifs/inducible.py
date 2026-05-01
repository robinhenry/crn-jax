"""Inducible expression — Hill-modulated birth-death (e.g. PLtetO-1 + aTc).

State: one species ``X`` (e.g. GFP molecule count).
Input: scalar ``u`` (e.g. aTc concentration), held constant per trajectory.

Reactions
---------
    R0:  ∅ → X     at rate  β · uⁿ / (Kⁿ + uⁿ)     (Hill-modulated production)
    R1:  X → ∅     at rate  γ · X                   (linear degradation)

The stationary distribution at constant ``u`` is Poisson with mean
``⟨X⟩(u) = (β/γ) · uⁿ / (Kⁿ + uⁿ)`` — Fano factor 1.

Defaults match the canonical "inducible" benchmark used across the
neural-crn experiment series (β = 30 molec/min, K = 6 ng/mL, n = 2,
γ = 0.023 /min for stable GFP).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..kinetics import hill_function
from ..types import PRNGKey
from ._common import (
    State,
    flatten_species,
    make_vmap_simulator,
    repeat_input_per_triple,
    sample_initial_state,
)

# --- Parameters --------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Params:
    """Truth parameters for the inducible motif.

    Defaults match a stable-GFP reporter under the PLtetO-1 promoter.
    Switch ``gamma`` to 0.07 for ssrA-tagged GFP (faster turnover, lower
    steady state but same Hill shape).
    """

    beta: float = 30.0
    K: float = 6.0
    n: float = 2.0
    gamma: float = 0.023


# --- Propensities + reactions -----------------------------------------------


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    """Return a ``(state, u) -> Array[2]`` closure with ``params`` baked in."""

    def f(state: State, u: Array) -> Array:
        a_plus = params.beta * hill_function(u, params.K, params.n)
        a_minus = params.gamma * state.x
        return jnp.array([a_plus, a_minus])

    return f


def apply_reaction(state: State, j: Array) -> State:
    """R0 increments ``x``, R1 decrements (floored at 0)."""
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


# --- One-call dataset --------------------------------------------------------


class Dataset(NamedTuple):
    """Output of :func:`simulate_dataset`.

    All arrays are NumPy (host-side) since downstream usage is bin/analyse,
    not further JAX work.
    """

    times: np.ndarray  # (n_steps,) — sample times in minutes
    x0: np.ndarray  # (n_envs,) — sampled initial X
    u: np.ndarray  # (n_envs,) — per-trajectory constant input
    Xs: np.ndarray  # (n_envs, n_steps) — full trajectories
    X_t: np.ndarray  # (n_envs * n_steps,) — flat triples
    dX: np.ndarray  # (n_envs * n_steps,)
    u_per_triple: np.ndarray  # (n_envs * n_steps,) — u broadcast to triple level


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_envs: int = 1800,
    n_steps: int = 1440,
    dt: float = 1.0,
    x0_dist: tuple = ("uniform", 0.0, 1500.0),
    u_dist: tuple = ("uniform", 0.0, 35.0),
) -> Dataset:
    """Simulate ``n_envs`` independent inducible-motif trajectories.

    Each trajectory draws its own initial X and its own constant u. Returns
    both the raw trajectories and the flat ``(X_t, dX, u_per_triple)``
    one-step transitions used by moment-matching pipelines.
    """
    k_x0, k_u, k_sim = jax.random.split(key, 3)
    x0 = sample_initial_state(k_x0, (n_envs,), x0_dist)
    u_arr = sample_initial_state(k_u, (n_envs,), u_dist)
    keys = jax.random.split(k_sim, n_envs)

    run = make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction, State)
    states = run(keys, x0, dt, u_arr)

    times = jnp.arange(1, n_steps + 1) * dt
    Xs = np.asarray(states.x)
    x0_np = np.asarray(x0)
    u_np = np.asarray(u_arr)
    X_t, dX = flatten_species(x0_np, Xs)
    u_per_triple = repeat_input_per_triple(u_np, n_steps)

    return Dataset(
        times=np.asarray(times),
        x0=x0_np,
        u=u_np,
        Xs=Xs,
        X_t=X_t,
        dX=dX,
        u_per_triple=u_per_triple,
    )
