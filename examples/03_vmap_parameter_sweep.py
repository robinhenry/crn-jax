"""Example 03 — parameter sweep via ``jax.vmap`` (the GPU payoff demo).

Runs 1 024 independent birth-death trajectories with *different* λ/μ kinetic
constants in a single ``jax.jit(jax.vmap(...))`` call. This is the use case
that motivates the library: the same SSA routine executes in parallel across
a population of parameter settings.

Run:

    python examples/03_vmap_parameter_sweep.py

On a CPU this example still completes in seconds. On a GPU it completes
essentially as fast as a single run.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax import run_gillespie_loop


N_REPLICATES = 1024
T_FINAL = 100.0


class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


def propensities(state: State, action: jax.Array) -> jax.Array:
    """``action`` carries (λ, μ) — the per-replicate kinetic parameters."""
    birth, death = action[0], action[1]
    return jnp.array([birth, death * state.x])


def apply_reaction(state: State, j: jax.Array) -> State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


def single_run(key, action):
    state0 = State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )
    final, _ = run_gillespie_loop(
        key=key,
        initial_state=state0,
        action=action,
        target_time=T_FINAL,
        max_steps=50_000,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=state0.next_reaction_time,
    )
    return final.x


def main():
    # Log-uniform grid over λ ∈ [0.5, 10], μ ∈ [0.02, 0.5].
    key_params = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key_params)
    lams = jnp.exp(jax.random.uniform(k1, (N_REPLICATES,), minval=jnp.log(0.5), maxval=jnp.log(10.0)))
    mus = jnp.exp(jax.random.uniform(k2, (N_REPLICATES,), minval=jnp.log(0.02), maxval=jnp.log(0.5)))
    actions = jnp.stack([lams, mus], axis=1)            # (N, 2)
    keys = jax.random.split(jax.random.PRNGKey(0), N_REPLICATES)

    batched_run = jax.jit(jax.vmap(single_run))

    # Warm-up to amortise JIT compilation out of the timing.
    batched_run(keys[:4], actions[:4]).block_until_ready()

    t0 = time.time()
    xs = batched_run(keys, actions).block_until_ready()
    dt = time.time() - t0

    print(f"Ran {N_REPLICATES} independent birth-death trajectories to t={T_FINAL}")
    print(f"  on device: {jax.default_backend()}")
    print(f"  wallclock: {dt:.3f} s  ({dt / N_REPLICATES * 1000:.3f} ms / trajectory)")

    # Compare empirical means to analytical steady-state λ/μ.
    expected = lams / mus
    empirical = xs
    relerr = jnp.abs(empirical - expected) / jnp.maximum(expected, 1.0)
    print(
        f"  median |emp - λ/μ| / (λ/μ) = {float(jnp.median(relerr)):.3f}   "
        f"(stochastic — ~O(1/√(λ·T)) at T={T_FINAL})"
    )


if __name__ == "__main__":
    main()
