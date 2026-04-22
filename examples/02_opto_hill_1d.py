"""Example 02 — opto-Hill gene expression: light-driven Hill production + linear degradation.

Reactions:
    ∅      -- k_prod · hill(U, K, n)  -->  X       (production, Hill-driven by light U ∈ [0, 1])
    X      -- k_deg · X               -->  ∅       (linear degradation)

This is the model used for NSDE / symbolic-regression experiments in the
parent project. Running this script simulates the response to a step input
U: 0 → 1 and plots the resulting trajectory bundle.

Run:

    python examples/02_opto_hill_1d.py
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax import run_gillespie_loop
from crn_jax.extras import hill_function


K_PROD = 5.0
K_HILL = 0.5
N_HILL = 2.0
K_DEG = 0.05


class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


def propensities(state: State, action: jax.Array) -> jax.Array:
    """Action is the (scalar) light input U ∈ [0, 1]."""
    a_plus = K_PROD * hill_function(action, K_HILL, N_HILL)
    a_minus = K_DEG * state.x
    return jnp.array([a_plus, a_minus])


def apply_reaction(state: State, j: jax.Array) -> State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


def run_interval(key, state, action, dt, previous_action):
    final, next_rxn = run_gillespie_loop(
        key=key,
        initial_state=state,
        action=action,
        target_time=state.time + dt,
        max_steps=50_000,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=state.next_reaction_time,
        previous_action=previous_action,
    )
    return final._replace(next_reaction_time=next_rxn)


def main():
    dt = 5.0              # minutes between observations
    n_steps = 120         # 10 hours total
    n_replicates = 64

    # Step-input schedule: U=0 for first 30 min, then U=1 for the rest.
    steps = jnp.arange(n_steps) * dt
    us = jnp.where(steps < 30.0, 0.0, 1.0)

    def one_trajectory(key):
        keys = jax.random.split(key, n_steps)
        state = State(
            time=jnp.array(0.0),
            x=jnp.array(0.0),
            next_reaction_time=jnp.array(jnp.inf),
        )

        def body(carry, inputs):
            state, prev_u = carry
            key, u = inputs
            new_state = run_interval(key, state, u, dt, prev_u)
            return (new_state, u), new_state.x

        (_, _), xs = jax.lax.scan(body, (state, jnp.array(0.0)), (keys, us))
        return xs

    print(f"Running {n_replicates} opto-Hill trajectories over {n_steps * dt:.0f} min...")
    batch = jax.jit(jax.vmap(one_trajectory))(
        jax.random.split(jax.random.PRNGKey(0), n_replicates)
    )
    print(f"  trajectory batch shape:  {batch.shape}  (replicates × timesteps)")
    print(f"  pre-step mean X  (t<30): {float(jnp.mean(batch[:, :6])):.2f}")
    print(f"  post-step mean X (t>200): {float(jnp.mean(batch[:, 40:])):.2f}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        t = np.asarray(steps)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

        for trj in np.asarray(batch[:20]):
            ax1.plot(t, trj, color="tab:blue", alpha=0.2)
        ax1.plot(t, np.asarray(jnp.mean(batch, axis=0)), color="tab:blue", lw=2, label="mean")
        ax1.set_ylabel("X (molecules)")
        ax1.legend()
        ax1.set_title("Opto-Hill gene expression — step input U: 0 → 1 at t = 30 min")

        ax2.step(t, np.asarray(us), color="tab:orange", where="post")
        ax2.set_xlabel("time (min)")
        ax2.set_ylabel("U (light)")
        ax2.set_ylim(-0.1, 1.1)

        plt.tight_layout()
        plt.savefig("example_02.png", dpi=120)
        print("Saved plot to example_02.png")
    except ImportError:
        print("(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
