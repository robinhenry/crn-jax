"""Example 01 — a minimal 1-species birth-death process.

Reactions:
    ∅ -- λ   -->  X  (birth)
    X -- μ·x -->  ∅  (death)

This example simulates 10 Gillespie trajectories and plots them against the
steady-state mean λ/μ.
"""

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax import plot_trajectories, simulate_trajectory

# --- Model parameters --------------------------------------------------------
BIRTH_RATE = 3.0  # λ
DEATH_RATE = 0.1  # μ
# Steady-state distribution is Poisson(λ/μ) with mean = 30.

DT = 1.0  # sampling interval (observations are recorded every DT)
N_STEPS = 200  # number of sampling intervals per trajectory
N_REPLICATES = 10  # number of independent trajectories to simulate


# --- State and reactions -----------------------------------------------------
class State(NamedTuple):
    # Scalar simulation time.
    time: jax.Array
    # Molecule count of species X.
    x: jax.Array
    # Time of the next scheduled reaction (carried across intervals so the
    # trajectory is independent of the sampling grid — see crn_jax.gillespie).
    next_reaction_time: jax.Array


def propensities(state: State, _input: jax.Array) -> jax.Array:
    # Constant-rate birth + first-order degradation. The `input` argument is
    # required by the `compute_propensities_fn` contract but unused here.
    return jnp.array([BIRTH_RATE, DEATH_RATE * state.x])


def apply_reaction(state: State, j: jax.Array) -> State:
    # j == 0 is a birth (+1 molecule), j == 1 is a death (-1 molecule).
    # The max(0, ...) guards against ever dipping negative under floating
    # point; the propensity a_death = μ·x vanishes at x=0 so it's belt-and-braces.
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


# --- Simulation --------------------------------------------------------------
def _simulate(key: jax.Array, n_replicates: int) -> tuple[jax.Array, jax.Array]:
    """Return ``(times, xs)`` where ``xs`` has shape ``(n_replicates, N_STEPS)``."""
    # Initial state: zero molecules at t=0, no pending reaction scheduled.
    state0 = State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )

    # `simulate_trajectory` scans simulate_interval N_STEPS times and stacks the
    # per-step states. `vmap` parallelises across replicates; `jit` compiles
    # the whole batched trajectory once.
    @jax.jit
    @jax.vmap
    def run_one(k):
        return simulate_trajectory(
            key=k,
            initial_state=state0,
            timestep=DT,
            n_steps=N_STEPS,
            compute_propensities_fn=propensities,
            apply_reaction_fn=apply_reaction,
        )

    # Split the master key into one per replicate so each trajectory is independent.
    keys = jax.random.split(key, n_replicates)
    states = run_one(keys)

    # `states.x` has shape (n_replicates, N_STEPS); state i along the time axis
    # corresponds to wall-clock time (i + 1) * DT.
    times = jnp.arange(1, N_STEPS + 1) * DT
    return times, states.x


# --- Plotting ----------------------------------------------------------------
def _plot(times: jax.Array, xs: jax.Array, path: str | Path) -> None:
    # `plot_trajectories` accepts a (N, T) ensemble and step-plots each replicate.
    fig, ax = plot_trajectories(
        times,
        xs,
        ylabel="X (molecules)",
        title=f"Birth-death: {xs.shape[0]} trajectories (λ={BIRTH_RATE}, μ={DEATH_RATE})",
    )
    # Overlay the analytic steady-state mean λ/μ for reference.
    ax.axhline(BIRTH_RATE / DEATH_RATE, color="k", ls="--", lw=1, label="λ/μ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


# --- Entry point -------------------------------------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    times, xs = _simulate(key, N_REPLICATES)
    _plot(times, xs, Path(__file__).parent / "example_01.png")
