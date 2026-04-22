"""Example 01 — a minimal 1-species birth-death process.

Reactions:
    ∅ -- λ   -->  X  (birth)
    X -- μ·x -->  ∅  (death)

We simulate 10 trajectories and plot them against the steady-state mean λ/μ.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax import plot_trajectories
from crn_jax.templates import step_interval


class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


BIRTH_RATE = 3.0   # λ
DEATH_RATE = 0.1   # μ
# Steady-state mean = λ / μ = 30.


def propensities(state: State, _action: jax.Array) -> jax.Array:
    return jnp.array([BIRTH_RATE, DEATH_RATE * state.x])


def apply_reaction(state: State, j: jax.Array) -> State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


DT = 1.0
N_STEPS = 200


def simulate_trajectory(key):
    """Return ``xs`` of shape ``(N_STEPS,)`` sampled every ``DT``."""
    state0 = State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )

    def body(state, k):
        new_state = step_interval(
            key=k,
            state=state,
            action=jnp.array(0.0),
            timestep=DT,
            max_steps=10_000,
            compute_propensities_fn=propensities,
            apply_reaction_fn=apply_reaction,
        )
        return new_state, new_state.x

    _, xs = jax.lax.scan(body, state0, jax.random.split(key, N_STEPS))
    return xs


def main():
    n_replicates = 10

    times = jnp.arange(N_STEPS) * DT
    keys = jax.random.split(jax.random.PRNGKey(0), n_replicates)
    xs = jax.jit(jax.vmap(simulate_trajectory))(keys)

    import matplotlib.pyplot as plt

    ax = plot_trajectories(
        times, xs,
        ylabel="X (molecules)",
        title=f"Birth-death: {n_replicates} trajectories (λ={BIRTH_RATE}, μ={DEATH_RATE})",
    )
    ax.axhline(BIRTH_RATE / DEATH_RATE, color="k", ls="--", lw=1, label="λ/μ")
    ax.legend()
    plt.tight_layout()
    plt.savefig("example_01.png", dpi=120)
    print("Saved plot to example_01.png")


if __name__ == "__main__":
    main()
