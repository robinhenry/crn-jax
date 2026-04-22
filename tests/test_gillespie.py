"""Tests for ``crn_jax.gillespie`` — the SSA driver and its wrappers.

These tests intentionally avoid any domain-specific scaffolding — they use a
plain NamedTuple birth-death process defined in ``conftest.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crn_jax.gillespie import simulate_interval, simulate_until

from .conftest import (
    BirthDeathState,
    birth_death_apply,
    birth_death_propensities,
    initial_state,
)


def _run_to(state: BirthDeathState, key, action, target_time, max_steps=10_000):
    return simulate_until(
        key=key,
        initial_state=state,
        action=action,
        target_time=target_time,
        max_steps=max_steps,
        compute_propensities_fn=birth_death_propensities,
        apply_reaction_fn=birth_death_apply,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=state.next_reaction_time,
    )


def test_determinism_same_key_same_trajectory():
    """Same PRNG key ⇒ identical final state and pending time."""
    key = jax.random.PRNGKey(42)
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    s1, t1 = _run_to(s0, key, action, target_time=20.0)
    s2, t2 = _run_to(s0, key, action, target_time=20.0)

    assert jnp.array_equal(s1.x, s2.x)
    assert jnp.array_equal(s1.time, s2.time)
    assert jnp.array_equal(t1, t2)


def test_different_keys_diverge():
    """Different keys ⇒ different trajectories (with overwhelming probability)."""
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    trajectories = []
    for seed in range(5):
        key = jax.random.PRNGKey(seed)
        s, _ = _run_to(s0, key, action, target_time=50.0)
        trajectories.append(float(s.x))

    # With 5 independent seeds over a 50-minute window it is essentially
    # impossible for all five to land on identical X.
    assert len(set(trajectories)) > 1


def test_final_time_at_or_past_target():
    """After returning, the state time should equal the last reaction time
    that fired within the interval (≤ target_time), and next_reaction_time
    should be ≥ target_time."""
    key = jax.random.PRNGKey(0)
    action = jnp.array([2.0, 0.05])
    s0 = initial_state(x0=10.0)

    s, next_rxn_time = _run_to(s0, key, action, target_time=15.0)
    assert float(s.time) <= 15.0
    assert float(next_rxn_time) >= 15.0


def test_zero_propensities_is_frozen():
    """If all propensities are 0 the system should not advance."""
    key = jax.random.PRNGKey(7)
    action = jnp.array([0.0, 0.0])
    s0 = initial_state(x0=5.0)

    s, _ = _run_to(s0, key, action, target_time=100.0)
    assert float(s.x) == 5.0
    assert float(s.time) == 0.0


def test_pure_birth_mean_matches_poisson():
    """With birth rate λ and no death, E[X(T)] ≈ λ·T at large T (empirical
    check over many replicates)."""
    lam, T = 2.0, 50.0
    action = jnp.array([lam, 0.0])
    s0 = initial_state()

    def one_run(key):
        s, _ = _run_to(s0, key, action, target_time=T, max_steps=5_000)
        return s.x

    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    xs = jax.vmap(one_run)(keys)
    mean = float(jnp.mean(xs))
    # Expected mean = λ·T = 100; allow generous band for 200 samples.
    assert 85.0 < mean < 115.0


def test_pending_time_carries_across_calls():
    """Chaining two calls over [0, T/2] and [T/2, T] should produce the same
    *distribution* over X(T) as a single call over [0, T] — this is the whole
    point of preserving the pending reaction time."""
    lam, T = 3.0, 10.0
    action = jnp.array([lam, 0.2])
    s0 = initial_state(x0=5.0)

    def single(k):
        s, _ = _run_to(s0, k, action, target_time=T)
        return s.x

    def split(k):
        k1, k2 = jax.random.split(k)
        s_a, next_rxn_a = _run_to(s0, k1, action, target_time=T / 2)
        s_a = s_a._replace(next_reaction_time=next_rxn_a)
        s_b, _ = _run_to(s_a, k2, action, target_time=T)
        return s_b.x

    keys = jax.random.split(jax.random.PRNGKey(0), 500)
    xs_single = jax.vmap(single)(keys)
    xs_split = jax.vmap(split)(keys)
    # Distributions should have near-equal means.
    assert abs(float(jnp.mean(xs_single)) - float(jnp.mean(xs_split))) < 2.0


def test_max_steps_is_respected():
    """With ``max_steps=0`` the loop must not execute any reactions even when
    propensities are large."""
    key = jax.random.PRNGKey(0)
    action = jnp.array([1000.0, 1000.0])
    s0 = initial_state(x0=50.0)

    result = simulate_until(
        key=key,
        initial_state=s0,
        action=action,
        target_time=1000.0,
        max_steps=0,
        compute_propensities_fn=birth_death_propensities,
        apply_reaction_fn=birth_death_apply,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=s0.next_reaction_time,
    )
    s, _ = result
    # No reaction applied → x unchanged, time unchanged.
    assert float(s.x) == 50.0
    assert float(s.time) == 0.0


def test_action_change_invalidates_pending_time():
    """When ``previous_action`` differs from ``action``, the pending reaction
    time is reset to infinity so a fresh tau is sampled with the new
    propensities. We verify by contrasting fixed-key runs that should differ
    only because of the invalidation path."""
    key = jax.random.PRNGKey(99)
    action = jnp.array([1.0, 0.2])
    prev_action_same = jnp.array([1.0, 0.2])  # no change
    prev_action_diff = jnp.array([5.0, 0.2])  # change

    s0 = initial_state(x0=20.0)
    # Give s0 a pending time in the past of infinity — otherwise both paths
    # identical. Pre-seed a finite pending time so the "same action" path
    # actually reuses it.
    s_seeded = s0._replace(next_reaction_time=jnp.array(0.5))

    def run(prev):
        return simulate_until(
            key=key,
            initial_state=s_seeded,
            action=action,
            target_time=5.0,
            max_steps=10_000,
            compute_propensities_fn=birth_death_propensities,
            apply_reaction_fn=birth_death_apply,
            get_time_fn=lambda s: s.time,
            update_time_fn=lambda s, t: s._replace(time=t),
            pending_reaction_time=s_seeded.next_reaction_time,
            previous_action=prev,
        )

    s_same, _ = run(prev_action_same)
    s_diff, _ = run(prev_action_diff)
    # With the action unchanged the seeded pending time is honoured; with it
    # changed the loop must resample. The resulting state time is typically
    # different between the two paths.
    assert float(s_same.time) != float(s_diff.time) or float(s_same.x) != float(s_diff.x)


def test_jit_compiles_and_runs():
    """``simulate_until`` must compile under ``jax.jit`` when ``max_steps``
    is treated as a static int."""
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    @jax.jit
    def run(key, state, action):
        return simulate_until(
            key=key,
            initial_state=state,
            action=action,
            target_time=10.0,
            max_steps=1_000,
            compute_propensities_fn=birth_death_propensities,
            apply_reaction_fn=birth_death_apply,
            get_time_fn=lambda s: s.time,
            update_time_fn=lambda s, t: s._replace(time=t),
            pending_reaction_time=state.next_reaction_time,
        )

    s, _ = run(jax.random.PRNGKey(0), s0, action)
    # Just check it runs and produces a finite state.
    assert jnp.isfinite(s.x)
    assert jnp.isfinite(s.time)


def test_vmap_parallel_trajectories():
    """vmap over keys should produce a batched trajectory without errors."""
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    def one(key):
        s, _ = _run_to(s0, key, action, target_time=10.0)
        return s.x

    keys = jax.random.split(jax.random.PRNGKey(0), 64)
    xs = jax.vmap(one)(keys)
    assert xs.shape == (64,)
    assert jnp.all(jnp.isfinite(xs))


def test_vmap_parallel_different_parameters():
    """vmap over different birth/death rates should yield N independent runs."""
    n = 32
    birth_rates = jnp.linspace(0.5, 5.0, n)
    death_rates = jnp.full(n, 0.1)
    actions = jnp.stack([birth_rates, death_rates], axis=1)  # (n, 2)

    s0 = initial_state()

    def one(key, action):
        s, _ = _run_to(s0, key, action, target_time=50.0)
        return s.x

    keys = jax.random.split(jax.random.PRNGKey(0), n)
    xs = jax.vmap(one)(keys, actions)
    # Higher birth rate ⇒ higher mean X (monotone trend at 50 min).
    # Not a sharp test (stochastic) but the extremes should differ.
    assert float(jnp.mean(xs[-4:])) > float(jnp.mean(xs[:4]))


# --- simulate_interval wrapper ---------------------------------------------------


def test_simulate_interval_advances_time():
    """After one interval of length 5, state.time should be ≤ 5 and
    next_reaction_time should be ≥ 5."""
    key = jax.random.PRNGKey(0)
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    s1 = simulate_interval(
        key=key,
        state=s0,
        action=action,
        timestep=5.0,
        max_steps=1_000,
        compute_propensities_fn=birth_death_propensities,
        apply_reaction_fn=birth_death_apply,
    )
    assert float(s1.time) <= 5.0
    assert float(s1.next_reaction_time) >= 5.0


def test_simulate_interval_chain_matches_distribution():
    """Chaining ``simulate_interval`` calls should reproduce the marginal at T."""
    action = jnp.array([1.5, 0.1])
    T = 20.0

    def chained(key):
        state = initial_state()
        for i in range(4):
            key, sub = jax.random.split(key)
            state = simulate_interval(
                key=sub,
                state=state,
                action=action,
                timestep=T / 4,
                max_steps=2_000,
                compute_propensities_fn=birth_death_propensities,
                apply_reaction_fn=birth_death_apply,
                interval_start=jnp.array(i * T / 4),
            )
        return state.x

    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    xs = jax.vmap(chained)(keys)
    # Steady state mean ≈ 1.5 / 0.1 = 15; at T=20 we haven't fully equilibrated,
    # but should be in the plausible band.
    mean = float(jnp.mean(xs))
    assert 5.0 < mean < 25.0


def test_simulate_interval_jits():
    """The wrapper should compile under jit just like the core function."""
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    @jax.jit
    def one_step(key, state):
        return simulate_interval(
            key=key,
            state=state,
            action=action,
            timestep=2.0,
            max_steps=500,
            compute_propensities_fn=birth_death_propensities,
            apply_reaction_fn=birth_death_apply,
        )

    s = one_step(jax.random.PRNGKey(0), s0)
    assert jnp.isfinite(s.x)
