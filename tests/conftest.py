"""Shared fixtures for crn-jax tests.

Tests use tiny NamedTuple states to avoid depending on any specific dataclass
framework (flax, pydantic, etc.).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BirthDeathState(NamedTuple):
    """Minimal state: one species X, plus bookkeeping fields."""

    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


def birth_death_propensities(state: BirthDeathState, input: jax.Array) -> jax.Array:
    """Birth at constant rate ``input[0]``, death at rate ``input[1] * x``.

    The ``input`` is a length-2 array of (birth_rate, death_rate) so the tests
    can exercise the input-change invalidation path without needing a full
    control signal.
    """
    birth_rate, death_rate = input[0], input[1]
    return jnp.array([birth_rate, death_rate * state.x])


def birth_death_apply(state: BirthDeathState, j: jax.Array) -> BirthDeathState:
    """Reaction 0: x += 1. Reaction 1: x -= 1 (floored at 0)."""
    dx = jnp.where(j == 0, 1.0, -1.0)
    return state._replace(x=jnp.maximum(0.0, state.x + dx))


def initial_state(x0: float = 0.0) -> BirthDeathState:
    return BirthDeathState(
        time=jnp.array(0.0),
        x=jnp.array(float(x0)),
        next_reaction_time=jnp.array(jnp.inf),
    )
