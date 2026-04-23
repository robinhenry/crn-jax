"""Benchmark crn-jax on the birth-death reference model.

Sweeps batch size (number of parallel trajectories under vmap) and reports:
  - compile time (first call, includes JIT tracing + XLA compilation)
  - run time    (median over ``N_REPEATS`` warmed-up calls)
  - throughput  (trajectory-steps per second)

Run with:
    python benchmarks/bench_crn_jax.py
"""

from __future__ import annotations

import statistics
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax import simulate_trajectory

from models import BIRTH_DEATH, BirthDeath

BATCH_SIZES = (1, 10, 100, 1_000, 10_000)
N_REPEATS = 5


class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


def _make_run_fn(model: BirthDeath):
    """Build a jitted, vmapped trajectory simulator for ``model``."""

    def propensities(state: State, _input: jax.Array) -> jax.Array:
        return jnp.array([model.birth_rate, model.death_rate * state.x])

    def apply_reaction(state: State, j: jax.Array) -> State:
        dx = jnp.where(j == 0, 1.0, -1.0)
        return state._replace(x=jnp.maximum(0.0, state.x + dx))

    state0 = State(
        time=jnp.array(0.0),
        x=jnp.array(model.initial_x),
        next_reaction_time=jnp.array(jnp.inf),
    )

    @jax.jit
    @jax.vmap
    def run(key: jax.Array) -> State:
        return simulate_trajectory(
            key=key,
            initial_state=state0,
            timestep=model.dt,
            n_steps=model.n_steps,
            compute_propensities_fn=propensities,
            apply_reaction_fn=apply_reaction,
        )

    return run


def _time_once(run_fn, keys) -> float:
    start = time.perf_counter()
    out = run_fn(keys)
    # Block until the GPU/CPU async dispatch finishes — otherwise we'd time the
    # launch, not the work.
    jax.block_until_ready(out)
    return time.perf_counter() - start


def bench(model: BirthDeath, batch_sizes=BATCH_SIZES, n_repeats: int = N_REPEATS) -> None:
    run = _make_run_fn(model)
    master_key = jax.random.PRNGKey(0)

    print(f"Device: {jax.devices()[0]}  (backend={jax.default_backend()})")
    print(f"Model:  birth-death  (λ={model.birth_rate}, μ={model.death_rate}, "
          f"n_steps={model.n_steps}, dt={model.dt})")
    print()
    header = f"{'batch':>8} {'compile (s)':>12} {'run (s)':>10} {'traj/s':>12} {'steps/s':>14}"
    print(header)
    print("-" * len(header))

    for n in batch_sizes:
        keys = jax.random.split(master_key, n)

        # First call = trace + compile + run.
        compile_s = _time_once(run, keys)

        # Subsequent calls = run only. Median to dampen jitter.
        run_samples = [_time_once(run, keys) for _ in range(n_repeats)]
        run_s = statistics.median(run_samples)

        traj_per_s = n / run_s
        steps_per_s = n * model.n_steps / run_s
        print(f"{n:>8d} {compile_s:>12.3f} {run_s:>10.4f} {traj_per_s:>12.1f} {steps_per_s:>14.0f}")


if __name__ == "__main__":
    bench(BIRTH_DEATH)
