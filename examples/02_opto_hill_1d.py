"""Example 02 — fully-independent replicates of optogenetic-driven Hill gene expression.

Reactions:
    ∅  -- k_prod · hill(U, K, n)  -->  X  (production, Hill-driven by light U ∈ [0, 1])
    X  -- k_deg · X               -->  ∅  (linear degradation)

Every replicate trajectory gets its own (key, initial state, step-on time), so
trajectories differ in three independent ways: randomness, starting point, and
when the light flips from 0 to 1. A single ``jax.jit(jax.vmap(...))`` call
dispatches all replicates in parallel — this is the canonical "batch of
independent experiments" pattern.
"""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from crn_jax import simulate_trajectory
from crn_jax.kinetics import hill_function

# --- Model parameters --------------------------------------------------------
K_PROD = 5.0  # max production rate (molecules / min)
K_HILL = 0.5  # half-max light input
N_HILL = 2.0  # Hill coefficient
K_DEG = 0.05  # first-order degradation rate
# Steady states: U=0 → ⟨X⟩ = 0; U=1 → ⟨X⟩ = K_PROD·hill(1)/K_DEG ≈ 80.

DT = 5.0  # sampling interval (min)
N_STEPS = 120  # 120 × 5 min = 10 hours
N_REPLICATES = 6  # laid out as a 2 × 3 grid of subplots

# Per-replicate priors: initial molecule count and light step-on time.
X0_RANGE = (0.0, 20.0)  # uniform (molecules)
STEP_TIME_RANGE = (30.0, 300.0)  # uniform (min) — when each replicate's light turns on


# --- State and reactions -----------------------------------------------------
class State(NamedTuple):
    # Scalar simulation time.
    time: jax.Array
    # Molecule count of species X.
    x: jax.Array
    # Time of the next scheduled reaction (preserved across intervals).
    next_reaction_time: jax.Array


def propensities(state: State, input: jax.Array) -> jax.Array:
    # `input` here is the (scalar) light input U ∈ [0, 1].
    a_plus = K_PROD * hill_function(input, K_HILL, N_HILL)
    a_minus = K_DEG * state.x
    return jnp.array([a_plus, a_minus])


def apply_reaction(state: State, j: jax.Array) -> State:
    # j == 0 is production (+1), j == 1 is degradation (-1).
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


# --- Per-replicate configuration ---------------------------------------------
def _sample_configs(
    key: jax.Array,
    n_replicates: int,
) -> tuple[jax.Array, State, jax.Array]:
    """Draw ``n`` independent (sim_key, initial_state, input_schedule) triples.

    Each leaf of the returned PyTree has a leading batch dim of size ``n``,
    which is what ``jax.vmap(..., in_axes=0)`` consumes.
    """
    k_x0, k_step, k_sim = jax.random.split(key, 3)

    # Random initial molecule counts → batched `State` (each leaf shape (n,)).
    x0s = jax.random.uniform(k_x0, (n_replicates,), minval=X0_RANGE[0], maxval=X0_RANGE[1])
    initial_states = State(
        time=jnp.zeros(n_replicates),
        x=x0s,
        next_reaction_time=jnp.full((n_replicates,), jnp.inf),
    )

    # Random per-replicate step-on times. Each replicate's light schedule is
    # simply U=0 before its step-time and U=1 afterwards (one transition).
    step_times = jax.random.uniform(
        k_step,
        (n_replicates,),
        minval=STEP_TIME_RANGE[0],
        maxval=STEP_TIME_RANGE[1],
    )
    interval_starts = jnp.arange(N_STEPS) * DT  # (N_STEPS,)
    us = jnp.where(
        interval_starts[None, :] >= step_times[:, None],  # (n, N_STEPS)
        1.0,
        0.0,
    )

    # One independent simulation key per replicate.
    sim_keys = jax.random.split(k_sim, n_replicates)

    return sim_keys, initial_states, us


# --- Simulation --------------------------------------------------------------
def _simulate(
    keys: jax.Array,
    initial_states: State,
    inputs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run one trajectory per replicate, vmapping over ALL per-replicate inputs."""

    # vmap's default in_axes=0 maps over axis 0 of every argument, including
    # every leaf of the `initial_states` PyTree. Each replicate therefore sees
    # a scalar `State` and a (N_STEPS,) `inputs` array — exactly what a
    # single-replicate call to `simulate_trajectory` expects.
    @jax.jit
    @jax.vmap
    def run_one(key, state, acts):
        return simulate_trajectory(
            key=key,
            initial_state=state,
            timestep=DT,
            n_steps=N_STEPS,
            compute_propensities_fn=propensities,
            apply_reaction_fn=apply_reaction,
            inputs=acts,
        )

    states = run_one(keys, initial_states, inputs)

    # Prepend each replicate's initial state so the starting-point variation is
    # visible at t=0 in the plot.
    xs = jnp.concatenate([initial_states.x[:, None], states.x], axis=1)
    times = jnp.arange(N_STEPS + 1) * DT
    return times, xs


# --- Plotting ----------------------------------------------------------------
def _plot(times: jax.Array, us: jax.Array, xs: jax.Array, path: str | Path) -> None:
    n = xs.shape[0]
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 5),
        sharex=True,
        sharey=True,
    )

    # Extend `us` by one step (repeat last value) so it aligns with the
    # length-(N_STEPS+1) `times` axis from `_simulate`.
    us_padded = jnp.concatenate([us, us[:, -1:]], axis=1)
    y_max = float(xs.max()) * 1.1

    # Per-replicate step-on time = start of the first interval where U = 1.
    # `argmax` returns the first True index (all-zero rows would map to 0, but
    # our STEP_TIME_RANGE guarantees a transition within every trajectory).
    first_on_idx = jnp.argmax(us, axis=1)  # (n,)
    step_times = jnp.asarray(times[:-1])[first_on_idx]  # (n,)

    for i, ax in enumerate(axes.flat):
        # Orange shading wherever that replicate's light was on (U = 1).
        ax.fill_between(
            times,
            0,
            y_max,
            where=(us_padded[i] > 0.5),
            step="post",
            color="tab:orange",
            alpha=0.25,
        )
        # The X(t) trajectory on top.
        ax.step(times, xs[i], where="post", color="tab:blue", lw=1.3)
        ax.set_title(f"step at {float(step_times[i]):.0f} min", fontsize=10)
        if i % n_cols == 0:
            ax.set_ylabel("X (molecules)")
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time (min)")

    # Single legend across the figure explaining the orange shading.
    orange_patch = mpatches.Patch(
        color="tab:orange",
        alpha=0.25,
        label="light on (U = 1)",
    )
    fig.legend(handles=[orange_patch], loc="upper right", frameon=False)
    fig.suptitle(
        f"Opto-Hill: {n} replicates with independent (key, x₀, step-on time)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


# --- Entry point -------------------------------------------------------------
if __name__ == "__main__":
    keys, initial_states, us = _sample_configs(jax.random.PRNGKey(0), N_REPLICATES)
    times, xs = _simulate(keys, initial_states, us)
    _plot(times, us, xs, Path(__file__).parent / "example_02.png")
