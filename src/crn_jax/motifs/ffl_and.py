"""C1-FFL with AND logic gate at the output (Mangan & Alon 2003).

State: three species ``[X, Y, Z]``.
Input: scalar ``u``, held constant per trajectory.

Reactions
---------
    R0:  ∅ → X     at rate  β_X · u^n_u / (K_u^n_u + u^n_u)        ν=(+1, 0, 0)
    R1:  X → ∅     at rate  γ_X · X                                 ν=(-1, 0, 0)
    R2:  ∅ → Y     at rate  β_Y · X^n_xy / (K_xy^n_xy + X^n_xy)    ν=( 0,+1, 0)
    R3:  Y → ∅     at rate  γ_Y · Y                                 ν=( 0,-1, 0)
    R4:  ∅ → Z     at rate  β_Z · Hill(X) · Hill(Y)                 ν=( 0, 0,+1)   (AND gate)
    R5:  Z → ∅     at rate  γ_Z · Z                                 ν=( 0, 0,-1)

The output reaction R4 is multiplicatively gated: Z is only produced
when *both* X and Y are above their respective half-max thresholds.

Defaults from Alon's textbook (2nd ed., Ch. 4) / Kaplan et al. 2008.
Steady states at saturating u: X_ss ≈ 1740, Y_ss ≈ 2167, Z_ss ≈ 2606.

dt = 0.1 is the recommended sampling interval — same reasoning as the
cascade motif (R2 and R4 are state-dependent and need fine sampling).
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
    Vec3State as State,
    flatten_species,
    make_vmap_simulator,
    repeat_input_per_triple,
    sample_initial_state,
)

# --- Parameters --------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Params:
    """Alon textbook 2nd ed., Ch. 4 / Kaplan 2008 / Mangan-Alon 2003 defaults."""

    # Stage 1 — u → X
    beta_X: float = 40.0
    K_u: float = 6.0
    n_u: float = 2.0
    gamma_X: float = 0.023
    # Stage 2 — X → Y (Y activated by X)
    beta_Y: float = 50.0
    K_xy: float = 100.0
    n_xy: float = 2.0
    gamma_Y: float = 0.023
    # Output — (X, Y) → Z (multiplicative AND gate)
    beta_Z: float = 60.0
    K_xz: float = 100.0
    n_xz: float = 2.0
    K_yz: float = 80.0
    n_yz: float = 2.0
    gamma_Z: float = 0.023


# --- Propensities + reactions -----------------------------------------------


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, u: Array) -> Array:
        X = state.x[0]
        Y = state.x[1]
        Z = state.x[2]
        a1 = params.beta_X * hill_function(u, params.K_u, params.n_u)
        a2 = params.gamma_X * X
        a3 = params.beta_Y * hill_function(X, params.K_xy, params.n_xy)
        a4 = params.gamma_Y * Y
        h_xz = hill_function(X, params.K_xz, params.n_xz)
        h_yz = hill_function(Y, params.K_yz, params.n_yz)
        a5 = params.beta_Z * h_xz * h_yz
        a6 = params.gamma_Z * Z
        return jnp.array([a1, a2, a3, a4, a5, a6])

    return f


def apply_reaction(state: State, j: Array) -> State:
    dX = jnp.where(j == 0, 1.0, jnp.where(j == 1, -1.0, 0.0))
    dY = jnp.where(j == 2, 1.0, jnp.where(j == 3, -1.0, 0.0))
    dZ = jnp.where(j == 4, 1.0, jnp.where(j == 5, -1.0, 0.0))
    new_x = jnp.maximum(0.0, state.x + jnp.array([dX, dY, dZ]))
    return state._replace(x=new_x)


# --- One-call dataset --------------------------------------------------------


class Dataset(NamedTuple):
    times: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    z0: np.ndarray
    u: np.ndarray
    Xs: np.ndarray
    Ys: np.ndarray
    Zs: np.ndarray
    X_t: np.ndarray
    Y_t: np.ndarray
    Z_t: np.ndarray
    dX: np.ndarray
    dY: np.ndarray
    dZ: np.ndarray
    u_per_triple: np.ndarray


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_envs: int = 1500,
    n_steps: int = 2000,
    dt: float = 0.1,
    x0_dist: tuple = ("uniform", 0.0, 1800.0),
    y0_dist: tuple = ("uniform", 0.0, 2200.0),
    z0_dist: tuple = ("uniform", 0.0, 2700.0),
    u_dist: tuple = ("uniform", 0.0, 35.0),
) -> Dataset:
    """Simulate ``n_envs`` independent FFL trajectories with AND-gate output.

    Default ``n_steps=2000`` × ``dt=0.1`` = 200 min/traj ≈ 4.6 cascade
    response times — long enough to populate the (X, Y) joint distribution
    but biased toward the saturating regime. For exponent identifiability
    (n_xz, n_yz) consider step responses or wider u sampling — see the
    ``experiments/19_ffl_and_gate`` README.
    """
    k_x0, k_y0, k_z0, k_u, k_sim = jax.random.split(key, 5)
    x0 = sample_initial_state(k_x0, (n_envs,), x0_dist)
    y0 = sample_initial_state(k_y0, (n_envs,), y0_dist)
    z0 = sample_initial_state(k_z0, (n_envs,), z0_dist)
    u_arr = sample_initial_state(k_u, (n_envs,), u_dist)
    keys = jax.random.split(k_sim, n_envs)

    x0_state = jnp.stack([x0, y0, z0], axis=-1)

    run = make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction, State)
    states = run(keys, x0_state, dt, u_arr)

    xs_full = np.asarray(states.x)
    Xs = xs_full[:, :, 0]
    Ys = xs_full[:, :, 1]
    Zs = xs_full[:, :, 2]

    times = jnp.arange(1, n_steps + 1) * dt
    x0_np = np.asarray(x0)
    y0_np = np.asarray(y0)
    z0_np = np.asarray(z0)
    u_np = np.asarray(u_arr)
    X_t, dX = flatten_species(x0_np, Xs)
    Y_t, dY = flatten_species(y0_np, Ys)
    Z_t, dZ = flatten_species(z0_np, Zs)
    u_per_triple = repeat_input_per_triple(u_np, n_steps)

    return Dataset(
        times=np.asarray(times),
        x0=x0_np,
        y0=y0_np,
        z0=z0_np,
        u=u_np,
        Xs=Xs,
        Ys=Ys,
        Zs=Zs,
        X_t=X_t,
        Y_t=Y_t,
        Z_t=Z_t,
        dX=dX,
        dY=dY,
        dZ=dZ,
        u_per_triple=u_per_triple,
    )
