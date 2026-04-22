"""Small helpers that are useful for stochastic reaction-network modelling.

These are *optional*: :func:`crn_jax.run_gillespie_loop` does not import them.
They live here so that users writing bio / systems-biology models don't have
to re-derive common primitives.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .types import PRNGKey


def sample_lognormal(
    key: PRNGKey,
    loc: float | Array,
    scale: float | Array,
) -> Array:
    """Draw a single log-normal sample: ``exp(loc + scale * N(0, 1))``.

    When ``scale == 0`` the return value is exactly ``exp(loc)`` regardless of
    ``key``. This lets callers default priors to point masses (deterministic
    behaviour) and opt into spread by raising ``scale``.

    Args:
        key: PRNG key.
        loc: Location parameter of the underlying normal (log-space mean).
        scale: Scale parameter of the underlying normal (log-space stdev).

    Returns:
        Scalar JAX array sampled from the log-normal distribution.
    """
    return jnp.exp(loc + scale * jax.random.normal(key))


def hill_function(
    x: Array,
    K: float | Array,
    n: float | Array,
) -> Array:
    """Hill function for cooperative binding / regulation.

    Formula: ``x**n / (K**n + x**n)``.

    The Hill equation models sigmoidal responses in biological systems —
    commonly gene regulation, enzyme kinetics, and receptor binding. It is
    frequently used inside propensity functions passed to
    :func:`crn_jax.run_gillespie_loop`.

    Args:
        x: Input concentration (molecules, arbitrary units).
        K: Half-maximal concentration (``EC50`` / ``IC50``). At ``x == K`` the
            output is ``0.5``.
        n: Hill coefficient.

            * ``n > 1``: positive cooperativity (sigmoidal, ultrasensitive).
            * ``n == 1``: non-cooperative (hyperbolic, Michaelis-Menten).
            * ``n < 1``: negative cooperativity (gradual response).

    Returns:
        Hill-function value in ``[0, 1]``.

    Examples:
        >>> from crn_jax.extras import hill_function
        >>> hill_function(x=100.0, K=90.0, n=3.6)
        Array(0.57..., dtype=float32)
    """
    x_powered = jnp.power(x, n)
    K_powered = jnp.power(K, n)
    return x_powered / (K_powered + x_powered)
