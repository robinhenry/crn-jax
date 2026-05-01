"""Negative autoregulation — Hill repressor on a single species (e.g. TetR–GFP).

State: one species ``X`` (no exogenous input — the production rate
depends on ``X`` itself, making this the canonical feedback motif).

Reactions
---------
    R0:  ∅ → X     at rate  β / (1 + (X/K)ⁿ)         (Hill-repressed production)
    R1:  X → ∅     at rate  γ · X                      (linear degradation)

The deterministic equilibrium is the root of ``β / (1 + (X/K)ⁿ) = γ X``;
for the default parameters (β=40, K=40, n=1.4, γ=0.023) it sits around
X ≈ 150–185.

The non-integer Hill exponent (n=1.4) is intentional — it's the test
case for symbolic regression's ability to recover non-canonical
exponents.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..types import PRNGKey
from ._common import (
    State,
    flatten_species,
    make_vmap_simulator,
    sample_initial_state,
)

# --- Parameters --------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Params:
    beta: float = 40.0
    K: float = 40.0
    n: float = 1.4
    gamma: float = 0.023


# --- Propensities + reactions -----------------------------------------------


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    """Return a ``(state, _u) -> Array[2]`` closure with ``params`` baked in.

    The motif has no exogenous input; the second argument is accepted but
    ignored, kept in the signature for compatibility with
    :func:`crn_jax.simulate_trajectory`.
    """

    def f(state: State, _u: Array) -> Array:
        a_prod = params.beta / (1.0 + (state.x / params.K) ** params.n)
        a_deg = params.gamma * state.x
        return jnp.array([a_prod, a_deg])

    return f


def apply_reaction(state: State, j: Array) -> State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


# --- One-call dataset --------------------------------------------------------


class Dataset(NamedTuple):
    """Output of :func:`simulate_dataset`. No ``u`` field — autoreg has no input."""

    times: np.ndarray
    x0: np.ndarray
    Xs: np.ndarray
    X_t: np.ndarray
    dX: np.ndarray


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_envs: int = 1800,
    n_steps: int = 1440,
    dt: float = 1.0,
    x0_dist: tuple = ("uniform", 0.0, 400.0),
) -> Dataset:
    """Simulate ``n_envs`` independent autoregulation trajectories.

    No input is sampled; the motif's production rate is fully determined by
    ``X`` itself.
    """
    k_x0, k_sim = jax.random.split(key, 2)
    x0 = sample_initial_state(k_x0, (n_envs,), x0_dist)
    keys = jax.random.split(k_sim, n_envs)

    run = make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction, State)
    # No exogenous input — pass zero per replicate. simulate_trajectory will
    # still thread the (constant) input through, but it's ignored by the
    # propensity.
    states = run(keys, x0, dt, jnp.zeros((n_envs,)))

    times = jnp.arange(1, n_steps + 1) * dt
    Xs = np.asarray(states.x)
    x0_np = np.asarray(x0)
    X_t, dX = flatten_species(x0_np, Xs)

    return Dataset(
        times=np.asarray(times),
        x0=x0_np,
        Xs=Xs,
        X_t=X_t,
        dX=dX,
    )
