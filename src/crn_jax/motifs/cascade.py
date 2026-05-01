"""Two-stage cascade — input → X → Y (e.g. aTc → LacI → EYFP).

State: two species ``[X, Y]``.
Input: scalar ``u``, held constant per trajectory.

Reactions
---------
    R0:  ∅ → X     at rate  β_X · u^n_u / (K_u^n_u + u^n_u)     ν = (+1, 0)
    R1:  X → ∅     at rate  γ_X · X                              ν = (-1, 0)
    R2:  ∅ → Y     at rate  β_Y / (1 + (X/K_X)^n_X)             ν = ( 0,+1)
    R3:  Y → ∅     at rate  γ_Y · Y                              ν = ( 0,-1)

The cascade *inverts* the input: at u → ∞, X is high (~1740) and Y is
suppressed (~0); at u = 0, X = 0 and Y is high (~1300).
"""

import dataclasses
import functools
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
    """Hooshangi 2005 / Cox 2007 cascade defaults."""

    # Stage 1 — u → X
    beta_X: float = 40.0  # molec / min
    K_u: float = 6.0  # ng / mL  (same units as input u)
    n_u: float = 2.0  # Hill coefficient (dimensionless)
    gamma_X: float = 0.023  # 1 / min
    # Stage 2 — X → Y (Y is repressed by X)
    beta_Y: float = 30.0  # molec / min
    K_X: float = 60.0  # molecules  (same units as state X)
    n_X: float = 2.7  # Hill coefficient (dimensionless)
    gamma_Y: float = 0.023  # 1 / min


# --- Propensities + reactions -----------------------------------------------


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, u: Array) -> Array:
        X = state.x[0]
        Y = state.x[1]
        a1 = params.beta_X * hill_function(u, params.K_u, params.n_u)
        a2 = params.gamma_X * X
        a3 = params.beta_Y / (1.0 + (X / params.K_X) ** params.n_X)
        a4 = params.gamma_Y * Y
        return jnp.array([a1, a2, a3, a4])

    return f


def apply_reaction(state: State, j: Array) -> State:
    """Stoichiometries: ν₁=(+1,0), ν₂=(-1,0), ν₃=(0,+1), ν₄=(0,-1)."""
    dX = jnp.where(j == 0, 1.0, jnp.where(j == 1, -1.0, 0.0))
    dY = jnp.where(j == 2, 1.0, jnp.where(j == 3, -1.0, 0.0))
    new_x = jnp.maximum(0.0, state.x + jnp.array([dX, dY]))
    return state._replace(x=new_x)


@functools.lru_cache(maxsize=None)
def _build_simulator(n_steps: int, params: Params):
    """Cache the JIT'd batch simulator per (n_steps, params) — see
    :func:`crn_jax.motifs.inducible._build_simulator` for the rationale.
    """
    return make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction)


# --- One-call dataset --------------------------------------------------------


class Dataset(NamedTuple):
    times: np.ndarray  # (n_steps,) — sample times (in `dt` units)
    x0: np.ndarray  # (n_replicates,) — sampled initial X
    y0: np.ndarray  # (n_replicates,) — sampled initial Y
    u: np.ndarray  # (n_replicates,) — per-trajectory constant input
    Xs: np.ndarray  # (n_replicates, n_steps) — full X trajectories
    Ys: np.ndarray  # (n_replicates, n_steps) — full Y trajectories
    X_t: np.ndarray  # (n_replicates * n_steps,) — flat X[k] for each (replicate, step)
    Y_t: np.ndarray  # (n_replicates * n_steps,) — flat Y[k] for each (replicate, step)
    dX: np.ndarray  # (n_replicates * n_steps,) — flat ΔX = X[k+1] − X[k]
    dY: np.ndarray  # (n_replicates * n_steps,) — flat ΔY = Y[k+1] − Y[k]
    u_per_triple: np.ndarray  # (n_replicates * n_steps,) — u broadcast to triple level


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_replicates: int = 1800,
    n_steps: int = 1000,
    dt: float = 0.1,
    x0_dist: tuple = ("uniform", 0.0, 1800.0),
    y0_dist: tuple = ("uniform", 0.0, 1400.0),
    u_dist: tuple = ("uniform", 0.0, 35.0),
) -> Dataset:
    """Simulate ``n_replicates`` independent cascade trajectories.

    Each trajectory draws its own (X(0), Y(0), u). Default ``n_steps=1000``
    × ``dt=0.1`` gives 100-min trajectories — about one cascade response
    time. For partial-observability or late-time work, bump ``n_steps``
    so the cascade has equilibrated under each input.
    """
    k_x0, k_y0, k_u, k_sim = jax.random.split(key, 4)
    x0 = sample_initial_state(k_x0, (n_replicates,), x0_dist)
    y0 = sample_initial_state(k_y0, (n_replicates,), y0_dist)
    u_arr = sample_initial_state(k_u, (n_replicates,), u_dist)
    keys = jax.random.split(k_sim, n_replicates)

    # Stack into (n_replicates, 2) — the per-replicate State.x shape.
    x0_state = jnp.stack([x0, y0], axis=-1)

    run = _build_simulator(n_steps, params)
    states = run(keys, x0_state, dt, u_arr)

    # states.x shape (n_replicates, n_steps, 2) — split into per-species trajectories.
    xs_full = np.asarray(states.x)
    Xs = xs_full[:, :, 0]
    Ys = xs_full[:, :, 1]

    times = jnp.arange(1, n_steps + 1) * dt
    x0_np = np.asarray(x0)
    y0_np = np.asarray(y0)
    u_np = np.asarray(u_arr)
    X_t, dX = flatten_species(x0_np, Xs)
    Y_t, dY = flatten_species(y0_np, Ys)
    u_per_triple = repeat_input_per_triple(u_np, n_steps)

    return Dataset(
        times=np.asarray(times),
        x0=x0_np,
        y0=y0_np,
        u=u_np,
        Xs=Xs,
        Ys=Ys,
        X_t=X_t,
        Y_t=Y_t,
        dX=dX,
        dY=dY,
        u_per_triple=u_per_triple,
    )
