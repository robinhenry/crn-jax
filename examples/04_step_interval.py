"""Example 04 — stepping through a state one fixed interval at a time.

Most RL / control / data-collection pipelines want to advance the simulation
by a fixed ``dt`` at each step, recording observations in between. The
``crn_jax.templates.step_interval`` helper does exactly that and threads the
pending-reaction time for you.

Run:

    python examples/04_step_interval.py
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax.templates import step_interval


class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


BIRTH_RATE = 2.0
DEATH_RATE = 0.1


def propensities(state: State, action: jax.Array) -> jax.Array:
    return jnp.array([BIRTH_RATE, DEATH_RATE * state.x])


def apply_reaction(state: State, j: jax.Array) -> State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


def main():
    dt = 1.0
    n_steps = 60
    print(f"Stepping the state forward in {n_steps} intervals of {dt} min.")

    @jax.jit
    def run(key):
        keys = jax.random.split(key, n_steps)
        state = State(
            time=jnp.array(0.0),
            x=jnp.array(0.0),
            next_reaction_time=jnp.array(jnp.inf),
        )

        def body(carry, k):
            state = carry
            new_state = step_interval(
                key=k,
                state=state,
                action=jnp.array(0.0),
                timestep=dt,
                max_steps=1_000,
                compute_propensities_fn=propensities,
                apply_reaction_fn=apply_reaction,
            )
            return new_state, new_state.x

        _, xs = jax.lax.scan(body, state, keys)
        return xs

    xs = run(jax.random.PRNGKey(0))
    for i in [0, 10, 20, 30, 40, 50, 59]:
        print(f"  t = {i * dt:5.1f} min   x = {int(xs[i]):3d}")
    print(f"\n  steady-state prediction λ/μ = {BIRTH_RATE / DEATH_RATE:.1f}")


if __name__ == "__main__":
    main()
