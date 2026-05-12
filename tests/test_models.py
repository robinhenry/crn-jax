"""Tests for ``crn_jax.models`` — the GRN benchmark library.

Three classes of checks:

1. **Per-model smoke tests** (parametrised over every module in
   ``ALL_MODELS``): propensities are finite/non-negative, ``simulate_dataset``
   returns the documented shapes, ``Params.easy()`` and ``Params.hard()`` are
   distinct.
2. **Parameter consistency**: load the packaged ``library.json`` and verify
   ``Params.easy()`` / ``Params.hard()`` match the JSON field-by-field.
3. **A few behavioural sanity checks**: birth-death steady state,
   negative-autoreg equilibrium, coherent-FFL pulse-through, toggle bimodality.

Plus the carried-over BYO and JIT-cache tests.
"""

from __future__ import annotations

import dataclasses
import json
from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crn_jax import models

# --- Mapping from JSON motif_name → model module ---------------------------
# Matches the order in src/crn_jax/models/library.json.
_JSON_NAME_TO_MODULE = {
    "Birth-death": models.birth_death,
    "Two-stage transcription-translation": models.two_stage,
    "Telegraph promoter": models.telegraph,
    "Negative autoregulation": models.negative_autoreg,
    "Positive autoregulation": models.positive_autoreg,
    "Bistable self-activation": models.bistable,
    "Linear activation chain": models.linear_chain,
    "Toggle switch": models.toggle,
    "Activator-repressor pair": models.activator_repressor,
    "Mutual activation": models.mutual_activation,
    "Coherent feed-forward loop (AND gate)": models.coherent_ffl,
    "Incoherent feed-forward loop": models.incoherent_ffl,
    "Repressilator": models.repressilator,
    "Cyclic hybrid ring": models.cyclic_ring,
}


def _representative_state(module) -> models.State:
    """Build a non-degenerate state for propensity sanity-checking.

    Every species gets a count of 1.0 — large enough to exercise all
    propensity terms, small enough that overflow isn't a concern.
    """
    n_species = len(module.SPECIES)
    return module.State(
        time=jnp.array(0.0),
        x=jnp.ones((n_species,)),
        next_reaction_time=jnp.array(jnp.inf),
    )


# --- Per-model smoke tests --------------------------------------------------


@pytest.mark.parametrize("module", models.ALL_MODELS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_propensities_finite_and_nonneg(module):
    p = module.Params()
    state = _representative_state(module)
    a = module.propensities_fn(p)(state, jnp.array(0.0))
    assert a.ndim == 1, f"propensities must be 1-D, got shape {a.shape}"
    assert jnp.all(jnp.isfinite(a)), f"propensities have non-finite entries: {a}"
    assert jnp.all(a >= 0), f"propensities must be non-negative: {a}"


@pytest.mark.parametrize("module", models.ALL_MODELS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_simulate_dataset_shapes(module):
    n_rep, n_steps = 8, 50
    ds = module.simulate_dataset(jax.random.PRNGKey(0), n_replicates=n_rep, n_steps=n_steps)
    n_species = len(module.SPECIES)
    assert ds.species == module.SPECIES
    assert ds.xs.shape == (n_rep, n_steps, n_species)
    assert ds.x0.shape == (n_rep, n_species)
    assert ds.X_t.shape == (n_rep * n_steps, n_species)
    assert ds.dX.shape == (n_rep * n_steps, n_species)
    assert ds.times.shape == (n_steps,)
    assert ds.X_t.dtype == np.float32
    assert ds.dX.dtype == np.float32


@pytest.mark.parametrize("module", models.ALL_MODELS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_easy_and_hard_params_differ(module):
    """Every model in the library has distinct easy / hard regimes."""
    assert module.Params.easy() != module.Params.hard()


# --- Parameter consistency against library.json ------------------------------


def _library_motifs():
    with resources.files("crn_jax.models").joinpath("library.json").open() as f:
        return json.load(f)["motifs"]


def test_library_json_covers_all_modules():
    """Every JSON entry maps to a module and every module is in the JSON."""
    json_names = {m["motif_name"] for m in _library_motifs()}
    mapped_names = set(_JSON_NAME_TO_MODULE)
    assert json_names == mapped_names, (
        f"library.json names ↔ module mapping mismatch.\n"
        f"  in JSON, not mapped: {json_names - mapped_names}\n"
        f"  mapped, not in JSON: {mapped_names - json_names}"
    )


@pytest.mark.parametrize("motif_entry", _library_motifs(), ids=lambda m: m["motif_name"])
def test_params_match_library_json(motif_entry):
    """``Params.easy()`` and ``Params.hard()`` match the JSON values."""
    module = _JSON_NAME_TO_MODULE[motif_entry["motif_name"]]
    for regime in ("easy", "hard"):
        expected = motif_entry[f"params_{regime}"]
        actual = dataclasses.asdict(getattr(module.Params, regime)())
        # The JSON is the source of truth; assert every JSON field is reflected
        # in Params, and that the Params dataclass has no extra fields.
        assert set(actual) == set(expected), (
            f"{motif_entry['motif_name']} / {regime}: fields differ (actual={set(actual)}, expected={set(expected)})"
        )
        for k, v in expected.items():
            np.testing.assert_allclose(
                actual[k],
                v,
                rtol=1e-7,
                atol=0.0,
                err_msg=f"{motif_entry['motif_name']} / {regime}: field {k!r}",
            )


# --- Behavioural sanity checks ----------------------------------------------


def test_birth_death_steady_state():
    """``⟨X⟩ → α/δ`` at steady state under the easy regime."""
    p = models.birth_death.Params.easy()
    ds = models.birth_death.simulate_dataset(
        jax.random.PRNGKey(0),
        params=p,
        n_replicates=128,
        n_steps=2000,
        dt=0.05,
        x0_dist=("constant", p.alpha / p.delta),
    )
    analytic = p.alpha / p.delta
    second_half = ds.xs[:, ds.xs.shape[1] // 2 :, 0]
    mean = float(second_half.mean())
    assert abs(mean - analytic) / analytic < 0.10, f"mean={mean}, analytic={analytic}"


def test_negative_autoreg_steady_state():
    """Late-time ⟨X⟩ matches the numerically-solved equilibrium."""
    p = models.negative_autoreg.Params.easy()

    def f(X):
        return p.beta_0 + p.beta_1 * (p.K**p.n) / (p.K**p.n + X**p.n) - p.delta * X

    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    analytic = 0.5 * (lo + hi)

    ds = models.negative_autoreg.simulate_dataset(
        jax.random.PRNGKey(0),
        params=p,
        n_replicates=128,
        n_steps=2000,
        dt=0.05,
    )
    mean = float(ds.xs[:, ds.xs.shape[1] // 2 :, 0].mean())
    assert abs(mean - analytic) / max(analytic, 1.0) < 0.15, f"mean={mean}, analytic={analytic}"


def test_coherent_ffl_pulse_through():
    """X(0) above threshold → Z eventually nonzero; X(0) below threshold → Z stays at 0."""
    p = models.coherent_ffl.Params.easy()
    # X above threshold for all replicates.
    ds_on = models.coherent_ffl.simulate_dataset(
        jax.random.PRNGKey(0),
        params=p,
        n_replicates=32,
        n_steps=200,
        dt=0.05,
        x0_dist=("constant", 5.0),
        y0_dist=("zero",),
        z0_dist=("zero",),
    )
    # X always below threshold.
    ds_off = models.coherent_ffl.simulate_dataset(
        jax.random.PRNGKey(1),
        params=p,
        n_replicates=32,
        n_steps=200,
        dt=0.05,
        x0_dist=("zero",),
        y0_dist=("zero",),
        z0_dist=("zero",),
    )
    z_on_max = float(ds_on.xs[..., 2].max())
    z_off_max = float(ds_off.xs[..., 2].max())
    assert z_on_max > 0, "expected coherent-FFL Z to pulse when X starts above threshold"
    assert z_off_max == 0, f"expected Z to stay at 0 with X≡0, got max={z_off_max}"


def test_telegraph_promoter_stays_binary():
    """The promoter state S must remain in {0, 1} for the duration."""
    ds = models.telegraph.simulate_dataset(
        jax.random.PRNGKey(0),
        n_replicates=32,
        n_steps=300,
    )
    s = ds.xs[..., 0]  # S species
    unique = np.unique(s)
    assert set(unique.tolist()).issubset({0.0, 1.0}), f"S left {{0,1}}: unique values {unique}"


# --- BYO path test (carried over from old motifs suite) ---------------------


def test_byo_path_compatible_with_simulate_trajectory():
    """A model's State + propensities_fn + apply_reaction should plug
    straight into simulate_trajectory."""
    from crn_jax import simulate_trajectory

    p = models.birth_death.Params()
    state0 = models.birth_death.State(
        time=jnp.array(0.0),
        x=jnp.array([0.0]),
        next_reaction_time=jnp.array(jnp.inf),
    )
    n_steps = 100
    inputs = jnp.zeros((n_steps,))
    states = simulate_trajectory(
        key=jax.random.PRNGKey(0),
        initial_state=state0,
        timestep=0.1,
        n_steps=n_steps,
        compute_propensities_fn=models.birth_death.propensities_fn(p),
        apply_reaction_fn=models.birth_death.apply_reaction,
        inputs=inputs,
    )
    assert states.x.shape == (n_steps, 1)
    assert jnp.all(states.x >= 0)


# --- JIT cache hit ----------------------------------------------------------


def test_simulate_dataset_reuses_compiled_simulator():
    """``_build_simulator(n_steps, params)`` must return the same object on
    repeated calls — otherwise every ``simulate_dataset`` invocation re-traces
    a fresh ``@jax.jit`` closure."""
    p = models.birth_death.Params()
    run1 = models.birth_death._build_simulator(200, p)
    run2 = models.birth_death._build_simulator(200, p)
    assert run1 is run2, "expected the same compiled simulator object"

    run3 = models.birth_death._build_simulator(400, p)
    assert run3 is not run1, "different n_steps must be a separate cache entry"

    run4 = models.birth_death._build_simulator(200, models.birth_death.Params(alpha=42.0))
    assert run4 is not run1, "different params must be a separate cache entry"
