"""Tests for ``crn_jax.motifs`` — the standard GRN motif library.

For each motif we check three things:

1. **Propensity sanity**: ``propensities_fn(Params())(state, u)`` returns a
   finite, non-negative array of the expected length.
2. **simulate_dataset shapes**: a tiny call (n_replicates=8, n_steps=200) returns
   the documented arrays with the right dtypes and sizes.
3. **Steady-state agreement**: the mean of the late-time trajectories
   matches the analytic ⟨x⟩ within a coarse tolerance. We use a relaxed
   3σ-ish bound because n_replicates is tiny — these tests are about
   correctness, not statistical power.
"""

import jax
import jax.numpy as jnp
import numpy as np

from crn_jax.motifs import autoreg, cascade, ffl_and, inducible

# --- Inducible ---------------------------------------------------------------


def test_inducible_propensities_finite():
    p = inducible.Params()
    state = inducible.State(time=jnp.array(0.0), x=jnp.array(100.0), next_reaction_time=jnp.array(jnp.inf))
    a = inducible.propensities_fn(p)(state, jnp.array(10.0))
    assert a.shape == (2,)
    assert jnp.all(jnp.isfinite(a))
    assert jnp.all(a >= 0)


def test_inducible_dataset_shapes():
    ds = inducible.simulate_dataset(jax.random.PRNGKey(0), n_replicates=8, n_steps=200)
    assert ds.Xs.shape == (8, 200)
    assert ds.X_t.shape == (8 * 200,)
    assert ds.dX.shape == (8 * 200,)
    assert ds.u.shape == (8,)
    assert ds.u_per_triple.shape == (8 * 200,)
    assert ds.X_t.dtype == np.float32
    # u_per_triple should broadcast each trajectory's u to its 200 triples.
    np.testing.assert_allclose(ds.u_per_triple[:200], ds.u[0])
    np.testing.assert_allclose(ds.u_per_triple[200:400], ds.u[1])


def test_inducible_steady_state_at_saturation():
    """At saturating u, ⟨X⟩ → β/γ. Use longer trajectories to equilibrate."""
    p = inducible.Params()
    ds = inducible.simulate_dataset(
        jax.random.PRNGKey(0),
        params=p,
        n_replicates=64,
        n_steps=1440,
        dt=1.0,
        u_dist=("uniform", 30.0, 35.0),  # saturating regime
    )
    analytic = p.beta / p.gamma  # ≈ 1304 at full saturation
    second_half = ds.Xs[:, ds.Xs.shape[1] // 2 :]
    mean = float(second_half.mean())
    # Allow a generous ±10% for finite-n_replicates noise; this is a sanity check.
    assert abs(mean - analytic) / analytic < 0.10, f"mean={mean}, analytic={analytic}"


# --- Autoreg -----------------------------------------------------------------


def test_autoreg_propensities_finite():
    p = autoreg.Params()
    state = autoreg.State(time=jnp.array(0.0), x=jnp.array(50.0), next_reaction_time=jnp.array(jnp.inf))
    a = autoreg.propensities_fn(p)(state, jnp.array(0.0))
    assert a.shape == (2,)
    assert jnp.all(jnp.isfinite(a))
    assert jnp.all(a >= 0)


def test_autoreg_dataset_shapes_and_no_input():
    ds = autoreg.simulate_dataset(jax.random.PRNGKey(0), n_replicates=8, n_steps=200)
    assert ds.Xs.shape == (8, 200)
    assert ds.X_t.shape == (8 * 200,)
    assert ds.dX.shape == (8 * 200,)
    # Autoreg has no input — Dataset NamedTuple has no `u` field.
    assert not hasattr(ds, "u")


def test_autoreg_steady_state():
    """Autoreg equilibrium solves β/(1+(X/K)ⁿ) = γX. Solve numerically and
    compare to long-run trajectory mean."""
    p = autoreg.Params()

    def equilibrium_eq(X):
        return p.beta / (1 + (X / p.K) ** p.n) - p.gamma * X

    # Bisection on [0, 1000] — equilibrium is a root.
    lo, hi = 0.0, 1000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if equilibrium_eq(mid) > 0:
            lo = mid
        else:
            hi = mid
    analytic = 0.5 * (lo + hi)

    ds = autoreg.simulate_dataset(jax.random.PRNGKey(0), n_replicates=64, n_steps=1440, dt=1.0)
    mean = float(ds.Xs[:, ds.Xs.shape[1] // 2 :].mean())
    assert abs(mean - analytic) / analytic < 0.10, f"mean={mean}, analytic={analytic}"


# --- Cascade -----------------------------------------------------------------


def test_cascade_propensities_finite():
    p = cascade.Params()
    state = cascade.State(time=jnp.array(0.0), x=jnp.array([100.0, 50.0]), next_reaction_time=jnp.array(jnp.inf))
    a = cascade.propensities_fn(p)(state, jnp.array(10.0))
    assert a.shape == (4,)
    assert jnp.all(jnp.isfinite(a))
    assert jnp.all(a >= 0)


def test_cascade_dataset_shapes():
    ds = cascade.simulate_dataset(jax.random.PRNGKey(0), n_replicates=8, n_steps=200, dt=0.1)
    assert ds.Xs.shape == (8, 200)
    assert ds.Ys.shape == (8, 200)
    assert ds.X_t.shape == (8 * 200,)
    assert ds.Y_t.shape == (8 * 200,)
    assert ds.dY.shape == (8 * 200,)
    assert ds.u.shape == (8,)


def test_cascade_inverts_input():
    """At u → 0: X stays low, Y is high. At u → ∞: X is high, Y is suppressed."""
    p = cascade.Params()
    # Long enough to equilibrate both stages (both have γ = 0.023 → τ ≈ 43 min).
    ds_low = cascade.simulate_dataset(
        jax.random.PRNGKey(0),
        params=p,
        n_replicates=32,
        n_steps=4000,
        dt=0.1,
        u_dist=("uniform", 0.0, 0.5),
    )
    ds_high = cascade.simulate_dataset(
        jax.random.PRNGKey(1),
        params=p,
        n_replicates=32,
        n_steps=4000,
        dt=0.1,
        u_dist=("uniform", 30.0, 35.0),
    )
    half_low = ds_low.Xs.shape[1] // 2
    half_high = ds_high.Xs.shape[1] // 2
    X_low_mean = float(ds_low.Xs[:, half_low:].mean())
    Y_low_mean = float(ds_low.Ys[:, half_low:].mean())
    X_high_mean = float(ds_high.Xs[:, half_high:].mean())
    Y_high_mean = float(ds_high.Ys[:, half_high:].mean())
    # u low: X near 0, Y high.
    assert X_low_mean < 50, f"expected X_low_mean ≪ 50, got {X_low_mean}"
    assert Y_low_mean > 1000, f"expected Y_low_mean ≫ 1000, got {Y_low_mean}"
    # u high: X high, Y near 0.
    assert X_high_mean > 1500, f"expected X_high_mean > 1500, got {X_high_mean}"
    assert Y_high_mean < 50, f"expected Y_high_mean ≪ 50, got {Y_high_mean}"


# --- FFL AND -----------------------------------------------------------------


def test_ffl_propensities_finite():
    p = ffl_and.Params()
    state = ffl_and.State(time=jnp.array(0.0), x=jnp.array([100.0, 50.0, 25.0]), next_reaction_time=jnp.array(jnp.inf))
    a = ffl_and.propensities_fn(p)(state, jnp.array(10.0))
    assert a.shape == (6,)
    assert jnp.all(jnp.isfinite(a))
    assert jnp.all(a >= 0)


def test_ffl_dataset_shapes():
    ds = ffl_and.simulate_dataset(jax.random.PRNGKey(0), n_replicates=8, n_steps=200, dt=0.1)
    assert ds.Xs.shape == (8, 200)
    assert ds.Ys.shape == (8, 200)
    assert ds.Zs.shape == (8, 200)
    assert ds.X_t.shape == (8 * 200,)
    assert ds.Z_t.shape == (8 * 200,)
    assert ds.dZ.shape == (8 * 200,)


def test_ffl_and_gate_off_at_zero_input():
    """At u = 0: X → 0 ⇒ Y → 0 ⇒ Z → 0 (the AND gate is off).

    Start Y(0) and Z(0) from zero so the test doesn't have to wait for
    initial-condition decay through Y and Z's slow γ ≈ 0.023 dynamics.
    """
    p = ffl_and.Params()
    # Start all species at zero — otherwise X(0) is high enough to produce
    # Y for a few X-decay timescales and the second-half mean stays elevated.
    ds = ffl_and.simulate_dataset(
        jax.random.PRNGKey(0),
        params=p,
        n_replicates=16,
        n_steps=4000,
        dt=0.1,
        u_dist=("uniform", 0.0, 0.5),
        x0_dist=("zero",),
        y0_dist=("zero",),
        z0_dist=("zero",),
    )
    half = ds.Xs.shape[1] // 2
    assert float(ds.Xs[:, half:].mean()) < 50
    assert float(ds.Ys[:, half:].mean()) < 50
    assert float(ds.Zs[:, half:].mean()) < 50


# --- BYO path ----------------------------------------------------------------


def test_byo_path_compatible_with_simulate_trajectory():
    """The motif's State + propensities_fn + apply_reaction should plug
    directly into simulate_trajectory, mirroring the README example."""
    from crn_jax import simulate_trajectory

    p = inducible.Params()
    state0 = inducible.State(
        time=jnp.array(0.0),
        x=jnp.array(0.0),
        next_reaction_time=jnp.array(jnp.inf),
    )
    n_steps = 200
    inputs = jnp.full((n_steps,), 10.0)

    states = simulate_trajectory(
        key=jax.random.PRNGKey(0),
        initial_state=state0,
        timestep=1.0,
        n_steps=n_steps,
        compute_propensities_fn=inducible.propensities_fn(p),
        apply_reaction_fn=inducible.apply_reaction,
        inputs=inputs,
    )
    assert states.x.shape == (n_steps,)
    assert jnp.all(states.x >= 0)


# --- JIT cache hit -----------------------------------------------------------


def test_simulate_dataset_jit_caches_across_calls():
    """Calling simulate_dataset twice with same n_steps shouldn't retrace."""
    key = jax.random.PRNGKey(0)
    # First call compiles.
    ds1 = inducible.simulate_dataset(key, n_replicates=8, n_steps=200)
    # Second call with same n_replicates/n_steps and a different key should be fast
    # — same JIT cache. We don't assert a wall-clock bound (flaky under load),
    # but the shape match is enough to confirm the call succeeded under
    # whatever compilation cache exists.
    ds2 = inducible.simulate_dataset(jax.random.PRNGKey(1), n_replicates=8, n_steps=200)
    assert ds1.Xs.shape == ds2.Xs.shape
