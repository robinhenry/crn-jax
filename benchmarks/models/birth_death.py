"""Birth-death process: ∅ -- λ --> X, X -- μ·x --> ∅.

Steady state of X(T): Poisson(λ/μ). With λ=3.0, μ=0.1 → mean 30, var 30.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from crn_jax import simulate_trajectory

PARAMS = dict(
    birth_rate=3.0,
    death_rate=0.1,
    x0=0,
    t_final=20.0,
    dt=0.1,
)
N_STEPS = int(PARAMS["t_final"] / PARAMS["dt"])  # 200


# --- crn-jax encoding --------------------------------------------------------
class _State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


def _propensities(state: _State, _input: jax.Array) -> jax.Array:
    return jnp.array([PARAMS["birth_rate"], PARAMS["death_rate"] * state.x])


def _apply_reaction(state: _State, j: jax.Array) -> _State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


def _build_crn_jax_runner():
    state0 = _State(
        time=jnp.array(0.0),
        x=jnp.array(float(PARAMS["x0"])),
        next_reaction_time=jnp.array(jnp.inf),
    )

    @jax.jit
    @jax.vmap
    def run(key):
        return simulate_trajectory(
            key=key,
            initial_state=state0,
            timestep=PARAMS["dt"],
            n_steps=N_STEPS,
            compute_propensities_fn=_propensities,
            apply_reaction_fn=_apply_reaction,
        )

    return run


_crn_jax_runner = None


def run_crn_jax(key: jax.Array, n_trajectories: int, *, return_full_trajectory: bool = False) -> jax.Array:
    """Returns an on-device jax.Array. Caller is responsible for block_until_ready / np.asarray.
    Full: (n_trajectories, N_STEPS); final: (n_trajectories,)."""
    global _crn_jax_runner
    if _crn_jax_runner is None:
        _crn_jax_runner = _build_crn_jax_runner()
    keys = jax.random.split(key, n_trajectories)
    states = _crn_jax_runner(keys)
    return states.x if return_full_trajectory else states.x[:, -1]


# --- gillespy2 encoding ------------------------------------------------------
def _build_gillespy2_model():
    import gillespy2

    model = gillespy2.Model(name="BirthDeath")
    birth = gillespy2.Parameter(name="lam", expression=str(PARAMS["birth_rate"]))
    death = gillespy2.Parameter(name="mu", expression=str(PARAMS["death_rate"]))
    model.add_parameter([birth, death])
    x = gillespy2.Species(name="X", initial_value=PARAMS["x0"])
    model.add_species([x])
    r_birth = gillespy2.Reaction(name="birth", rate=birth, reactants={}, products={x: 1})
    r_death = gillespy2.Reaction(name="death", rate=death, reactants={x: 1}, products={})
    model.add_reaction([r_birth, r_death])
    model.timespan(gillespy2.TimeSpan.linspace(t=PARAMS["t_final"], num_points=N_STEPS + 1))
    return model


_gillespy2_cache: tuple | None = None


def _get_gillespy2():
    global _gillespy2_cache
    if _gillespy2_cache is None:
        from gillespy2 import SSACSolver

        model = _build_gillespy2_model()
        # SSACSolver compiles a C++ binary on construction — the equivalent of
        # JIT for the JAX runner — so we cache it for fair per-call timing.
        _gillespy2_cache = (model, SSACSolver(model=model))
    return _gillespy2_cache


def run_gillespy2(seed: int, n_trajectories: int, *, return_full_trajectory: bool = False):
    """Full: (n_trajectories, N_STEPS+1); final: (n_trajectories,)."""
    model, solver = _get_gillespy2()
    # GillesPy2's C++ SSA solver requires a strictly positive seed.
    results = model.run(
        solver=solver,
        number_of_trajectories=n_trajectories,
        seed=int(seed) + 1,
    )
    arr = np.stack([np.asarray(r["X"]) for r in results])
    if return_full_trajectory:
        return arr
    return arr[:, -1]
