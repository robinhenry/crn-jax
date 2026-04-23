"""Lotka-Volterra (prey-predator).

Species: Prey (X), Predator (Y).
Reactions:
    Prey reproduction:    X --c1---->  2X       at c1·X
    Interinput:          X + Y -c2->  2Y       at c2·X·Y
    Predator death:       Y --c3---->  ∅        at c3·Y

Default rates: c1=1.0, c2=0.005, c3=0.6 (the classic SMfSB textbook setting).
Initial state: X=50, Y=100.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from crn_jax import simulate_trajectory

PARAMS = dict(
    c=(1.0, 0.005, 0.6),
    x0=50,
    y0=100,
    t_final=20.0,
    dt=0.1,
)
N_STEPS = int(PARAMS["t_final"] / PARAMS["dt"])  # 200


# --- crn-jax encoding --------------------------------------------------------
class _State(NamedTuple):
    time: jax.Array
    x: jax.Array  # prey
    y: jax.Array  # predator
    next_reaction_time: jax.Array


def _propensities(state: _State, _input: jax.Array) -> jax.Array:
    c1, c2, c3 = PARAMS["c"]
    return jnp.array(
        [
            c1 * state.x,
            c2 * state.x * state.y,
            c3 * state.y,
        ]
    )


def _apply_reaction(state: _State, j: jax.Array) -> _State:
    # Reaction 0: X -> 2X (Δx=+1, Δy=0)
    # Reaction 1: X+Y -> 2Y (Δx=-1, Δy=+1)
    # Reaction 2: Y -> ∅ (Δx=0, Δy=-1)
    dx = jnp.array([1.0, -1.0, 0.0])[j]
    dy = jnp.array([0.0, 1.0, -1.0])[j]
    return state._replace(
        x=jnp.maximum(0.0, state.x + dx),
        y=jnp.maximum(0.0, state.y + dy),
    )


_crn_jax_runner = None


def _build_crn_jax_runner():
    state0 = _State(
        time=jnp.array(0.0),
        x=jnp.array(float(PARAMS["x0"])),
        y=jnp.array(float(PARAMS["y0"])),
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
            # LV has runaway-prey tail events when the predator goes extinct.
            # Cap high enough that we don't truncate them artificially.
            max_steps=5_000_000,
        )

    return run


def run_crn_jax(key: jax.Array, n_trajectories: int, *, return_full_trajectory: bool = False):
    """Full: (n_trajectories, N_STEPS, 2); final: (n_trajectories, 2)."""
    global _crn_jax_runner
    if _crn_jax_runner is None:
        _crn_jax_runner = _build_crn_jax_runner()
    keys = jax.random.split(key, n_trajectories)
    states = _crn_jax_runner(keys)
    states.x.block_until_ready()
    out = np.stack([np.asarray(states.x), np.asarray(states.y)], axis=-1)
    if return_full_trajectory:
        return out
    return out[:, -1, :]


# --- gillespy2 encoding ------------------------------------------------------
def _build_gillespy2_model():
    import gillespy2

    c1, c2, c3 = PARAMS["c"]
    model = gillespy2.Model(name="LotkaVolterra")
    p1 = gillespy2.Parameter(name="c1", expression=str(c1))
    p2 = gillespy2.Parameter(name="c2", expression=str(c2))
    p3 = gillespy2.Parameter(name="c3", expression=str(c3))
    model.add_parameter([p1, p2, p3])
    prey = gillespy2.Species(name="Prey", initial_value=PARAMS["x0"])
    pred = gillespy2.Species(name="Predator", initial_value=PARAMS["y0"])
    model.add_species([prey, pred])
    r_repro = gillespy2.Reaction(
        name="prey_repro", rate=p1, reactants={prey: 1}, products={prey: 2}
    )
    r_inter = gillespy2.Reaction(
        name="interinput", rate=p2, reactants={prey: 1, pred: 1}, products={pred: 2}
    )
    r_death = gillespy2.Reaction(
        name="pred_death", rate=p3, reactants={pred: 1}, products={}
    )
    model.add_reaction([r_repro, r_inter, r_death])
    model.timespan(gillespy2.TimeSpan.linspace(t=PARAMS["t_final"], num_points=N_STEPS + 1))
    return model


_gillespy2_cache: tuple | None = None


def _get_gillespy2():
    global _gillespy2_cache
    if _gillespy2_cache is None:
        from gillespy2 import SSACSolver
        model = _build_gillespy2_model()
        _gillespy2_cache = (model, SSACSolver(model=model))
    return _gillespy2_cache


def run_gillespy2(seed: int, n_trajectories: int, *, return_full_trajectory: bool = False):
    model, solver = _get_gillespy2()
    results = model.run(
        solver=solver, number_of_trajectories=n_trajectories, seed=int(seed) + 1
    )
    arr = np.stack(
        [np.stack([np.asarray(r["Prey"]), np.asarray(r["Predator"])], axis=-1) for r in results]
    )
    if return_full_trajectory:
        return arr  # (n, n_steps+1, 2)
    return arr[:, -1, :]  # (n, 2)
