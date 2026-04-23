"""Small helpers that are useful for stochastic reaction-network modelling"""

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

    Args:
        x: Input concentration (molecules, arbitrary units).
        K: Half-maximal concentration. At ``x == K`` the
            output is ``0.5``.
        n: Hill coefficient.

    Returns:
        Hill-function value in ``[0, 1]``.
    """
    x_powered = jnp.power(x, n)
    K_powered = jnp.power(K, n)
    return x_powered / (K_powered + x_powered)
