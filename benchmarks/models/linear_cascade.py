"""Linear cascade: ∅ -- λ --> A1 -- k --> A2 -- k --> ... -- k --> A10 -- d --> ∅.

10 species (A1..A10), 20 reactions:
  - 1 source production:  ∅ → A1 at rate λ
  - 9 conversions:        Ai → A(i+1) at rate k·Ai (i=1..9)
  - 10 first-order decays: Ai → ∅ at rate d·Ai (i=1..10)

Steady-state mean (deterministic ODE) for A1 = λ / (k + d) = 10/(1+0.2) ≈ 8.33;
for downstream species: E[Ai] = (k/(k+d)) * E[A(i-1)] (until A10 which has no
outgoing conversion → E[A10] = (k/d) * E[A9] = 5 * E[A9] ≈ 81.4 in steady
state). The cascade hasn't fully relaxed at T=20, but both libraries should
agree.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from crn_jax import simulate_trajectory

N_SPECIES = 10

PARAMS = dict(
    source_rate=10.0,  # λ
    conv_rate=1.0,  # k (A_i -> A_(i+1))
    decay_rate=0.2,  # d (A_i -> ∅)
    n_species=N_SPECIES,
    t_final=20.0,
    dt=0.1,
)
N_STEPS = int(PARAMS["t_final"] / PARAMS["dt"])  # 200


def _expected_steady_state() -> np.ndarray:
    """Closed-form ODE steady state for the symmetric cascade."""
    lam, k, d = PARAMS["source_rate"], PARAMS["conv_rate"], PARAMS["decay_rate"]
    means = np.zeros(N_SPECIES)
    means[0] = lam / (k + d)
    for i in range(1, N_SPECIES - 1):
        means[i] = (k / (k + d)) * means[i - 1]
    means[N_SPECIES - 1] = (k / d) * means[N_SPECIES - 2]
    return means


# --- crn-jax encoding --------------------------------------------------------
class _State(NamedTuple):
    time: jax.Array
    x: jax.Array  # shape (N_SPECIES,)
    next_reaction_time: jax.Array


def _propensities(state: _State, _input: jax.Array) -> jax.Array:
    lam, k, d = PARAMS["source_rate"], PARAMS["conv_rate"], PARAMS["decay_rate"]
    # Order: [source, conv_1->2, conv_2->3, ..., conv_9->10, decay_1, ..., decay_10]
    source = jnp.array([lam])
    convs = k * state.x[:-1]  # 9 conversions (uses A1..A9)
    decays = d * state.x  # 10 decays
    return jnp.concatenate([source, convs, decays])


def _apply_reaction(state: _State, j: jax.Array) -> _State:
    # Stoichiometry matrix (n_reactions x n_species).
    # Reaction 0: source ∅ -> A1
    # Reactions 1..9: A_i -> A_(i+1) for i=1..9 (1-indexed); zero-indexed i=0..8
    # Reactions 10..19: decay A_i -> ∅ for i=1..10; zero-indexed i=0..9
    n = N_SPECIES
    sto = jnp.zeros((1 + (n - 1) + n, n))  # 20 x 10
    sto = sto.at[0, 0].set(1.0)
    for i in range(n - 1):
        sto = sto.at[1 + i, i].set(-1.0)
        sto = sto.at[1 + i, i + 1].set(1.0)
    for i in range(n):
        sto = sto.at[1 + (n - 1) + i, i].set(-1.0)
    delta = sto[j]
    return state._replace(x=jnp.maximum(0.0, state.x + delta))


_crn_jax_runner = None


def _build_crn_jax_runner():
    state0 = _State(
        time=jnp.array(0.0),
        x=jnp.zeros(N_SPECIES),
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
            max_steps=200_000,
        )

    return run


def run_crn_jax(key: jax.Array, n_trajectories: int, *, return_full_trajectory: bool = False) -> jax.Array:
    """Returns an on-device jax.Array. Caller is responsible for block_until_ready / np.asarray.
    Full: (n_trajectories, N_STEPS, N_SPECIES); final: (n_trajectories, N_SPECIES)."""
    global _crn_jax_runner
    if _crn_jax_runner is None:
        _crn_jax_runner = _build_crn_jax_runner()
    keys = jax.random.split(key, n_trajectories)
    states = _crn_jax_runner(keys)
    return states.x if return_full_trajectory else states.x[:, -1, :]


# --- gillespy2 encoding ------------------------------------------------------
def _build_gillespy2_model():
    import gillespy2

    n = N_SPECIES
    lam, k, d = PARAMS["source_rate"], PARAMS["conv_rate"], PARAMS["decay_rate"]
    model = gillespy2.Model(name="LinearCascade")
    p_lam = gillespy2.Parameter(name="lam", expression=str(lam))
    p_k = gillespy2.Parameter(name="k", expression=str(k))
    p_d = gillespy2.Parameter(name="d", expression=str(d))
    model.add_parameter([p_lam, p_k, p_d])
    species = [gillespy2.Species(name=f"A{i + 1}", initial_value=0) for i in range(n)]
    model.add_species(species)
    rxns = []
    rxns.append(gillespy2.Reaction(name="source", rate=p_lam, reactants={}, products={species[0]: 1}))
    for i in range(n - 1):
        rxns.append(
            gillespy2.Reaction(
                name=f"conv_{i + 1}_{i + 2}",
                rate=p_k,
                reactants={species[i]: 1},
                products={species[i + 1]: 1},
            )
        )
    for i in range(n):
        rxns.append(
            gillespy2.Reaction(
                name=f"decay_{i + 1}",
                rate=p_d,
                reactants={species[i]: 1},
                products={},
            )
        )
    model.add_reaction(rxns)
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
    results = model.run(solver=solver, number_of_trajectories=n_trajectories, seed=int(seed) + 1)
    arr = np.stack([np.stack([np.asarray(r[f"A{i + 1}"]) for i in range(N_SPECIES)], axis=-1) for r in results])
    if return_full_trajectory:
        return arr
    return arr[:, -1, :]
