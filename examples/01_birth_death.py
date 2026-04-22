"""Example 01 — minimal 1-species birth-death process via Gillespie SSA.

Reactions:
    ∅      -- λ   -->  X       (birth)
    X      -- μ·x -->  ∅       (death)

Analytical steady state is Poisson(λ/μ). We simulate a single long trajectory
and then 1 024 shorter replicates to compare the empirical steady-state
distribution against the Poisson prediction.

Run:

    python examples/01_birth_death.py
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from crn_jax import run_gillespie_loop


class State(NamedTuple):
    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


BIRTH_RATE = 3.0   # λ
DEATH_RATE = 0.1   # μ
# Steady-state mean = λ / μ = 30 (Poisson(30)).


def propensities(state: State, action: jax.Array) -> jax.Array:
    # action is ignored here — constant-rate birth-death.
    return jnp.array([BIRTH_RATE, DEATH_RATE * state.x])


def apply_reaction(state: State, j: jax.Array) -> State:
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


def simulate_until(key, initial, target_time):
    final, next_rxn = run_gillespie_loop(
        key=key,
        initial_state=initial,
        action=jnp.array(0.0),            # unused
        target_time=target_time,
        max_steps=50_000,
        compute_propensities_fn=propensities,
        apply_reaction_fn=apply_reaction,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=initial.next_reaction_time,
    )
    return final._replace(next_reaction_time=next_rxn)


def main():
    # --- Single long trajectory --------------------------------------------
    print(f"Simulating birth-death (λ={BIRTH_RATE}, μ={DEATH_RATE}) for T=200...")
    state0 = State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )
    state_final = simulate_until(jax.random.PRNGKey(0), state0, target_time=200.0)
    print(f"  final x  = {int(state_final.x)}")
    print(f"  final t  = {float(state_final.time):.2f}")

    # --- 1024 replicates sampled at T=100 ----------------------------------
    print("\nSampling steady-state distribution from 1 024 independent replicates...")

    @jax.jit
    @jax.vmap
    def replicate(key):
        final = simulate_until(key, state0, target_time=100.0)
        return final.x

    keys = jax.random.split(jax.random.PRNGKey(1), 1024)
    samples = replicate(keys)

    empirical_mean = float(jnp.mean(samples))
    empirical_var = float(jnp.var(samples))
    expected_mean = BIRTH_RATE / DEATH_RATE
    print(f"  empirical E[X] = {empirical_mean:.2f}   (Poisson predicts {expected_mean:.2f})")
    print(f"  empirical Var  = {empirical_var:.2f}   (Poisson predicts {expected_mean:.2f})")
    print(
        "  mean/var ratio =",
        f"{empirical_mean / empirical_var:.3f}",
        "(Poisson predicts 1.000)",
    )

    # --- Optional: plot ----------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy.stats import poisson

        plt.figure(figsize=(6, 4))
        counts, bins, _ = plt.hist(
            np.asarray(samples),
            bins=np.arange(0, 70) - 0.5,
            density=True,
            alpha=0.6,
            label="Gillespie samples",
        )
        ks = np.arange(0, 70)
        plt.plot(ks, poisson.pmf(ks, expected_mean), "r-", lw=2, label=f"Poisson({expected_mean:.0f})")
        plt.xlabel("X at t=100")
        plt.ylabel("density")
        plt.title("Birth-death steady-state — Gillespie vs Poisson prediction")
        plt.legend()
        plt.tight_layout()
        plt.savefig("example_01.png", dpi=120)
        print("\nSaved plot to example_01.png")
    except ImportError:
        print("\n(matplotlib / scipy not installed — skipping plot)")


if __name__ == "__main__":
    main()
