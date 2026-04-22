"""Tests for the ``step_interval`` template wrapper."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from crn_jax.templates import step_interval

from .conftest import (
    BirthDeathState,
    birth_death_apply,
    birth_death_propensities,
    initial_state,
)


def test_step_interval_advances_time():
    """After one interval of length 5, state.time should be ≤ 5 and
    next_reaction_time should be ≥ 5."""
    key = jax.random.PRNGKey(0)
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    s1 = step_interval(
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


def test_step_interval_chain_matches_distribution():
    """Chaining ``step_interval`` calls should reproduce the marginal at T."""
    action = jnp.array([1.5, 0.1])
    T = 20.0

    def chained(key):
        state = initial_state()
        for i in range(4):
            key, sub = jax.random.split(key)
            state = step_interval(
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


def test_step_interval_jits():
    """The template should compile under jit just like the core function."""
    action = jnp.array([1.0, 0.1])
    s0 = initial_state()

    @jax.jit
    def one_step(key, state):
        return step_interval(
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
