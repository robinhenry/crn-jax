"""Tests for the kinetics module (``hill_function``, ``sample_lognormal``)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from crn_jax.kinetics import hill_function, sample_lognormal


def test_hill_at_K_is_half():
    """x == K ⇒ hill = 1/2 for any n."""
    for n in [1.0, 2.0, 3.5]:
        val = float(hill_function(jnp.array(10.0), K=10.0, n=n))
        assert abs(val - 0.5) < 1e-5


def test_hill_monotone_in_x():
    """For fixed K, n > 0, the Hill function is monotone non-decreasing in x."""
    xs = jnp.linspace(0.0, 100.0, 201)
    vals = hill_function(xs, K=20.0, n=2.5)
    diffs = jnp.diff(vals)
    assert jnp.all(diffs >= -1e-6)


def test_hill_range_is_0_1():
    xs = jnp.array([0.0, 1e-3, 1.0, 10.0, 1e4])
    vals = hill_function(xs, K=5.0, n=3.0)
    assert float(jnp.min(vals)) >= 0.0
    assert float(jnp.max(vals)) <= 1.0


def test_hill_jits():
    @jax.jit
    def f(x):
        return hill_function(x, K=2.0, n=2.0)

    assert abs(float(f(jnp.array(2.0))) - 0.5) < 1e-5


def test_lognormal_deterministic_when_scale_zero():
    """scale == 0 ⇒ exp(loc) regardless of key."""
    k1 = jax.random.PRNGKey(0)
    k2 = jax.random.PRNGKey(1)
    v1 = float(sample_lognormal(k1, loc=jnp.log(3.0), scale=0.0))
    v2 = float(sample_lognormal(k2, loc=jnp.log(3.0), scale=0.0))
    assert abs(v1 - 3.0) < 1e-5
    assert abs(v2 - 3.0) < 1e-5


def test_lognormal_mean_approx():
    """E[exp(μ + σN)] = exp(μ + σ²/2). Empirical check at 10k samples."""
    mu, sigma = 0.5, 0.3
    keys = jax.random.split(jax.random.PRNGKey(0), 10_000)
    samples = jax.vmap(lambda k: sample_lognormal(k, mu, sigma))(keys)
    empirical_mean = float(jnp.mean(samples))
    expected = float(jnp.exp(mu + sigma ** 2 / 2))
    # Within 3% for 10k samples.
    assert abs(empirical_mean - expected) / expected < 0.03


def test_lognormal_positive():
    """All samples must be strictly positive."""
    keys = jax.random.split(jax.random.PRNGKey(0), 1000)
    samples = jax.vmap(lambda k: sample_lognormal(k, 0.0, 1.0))(keys)
    assert bool(jnp.all(samples > 0))
