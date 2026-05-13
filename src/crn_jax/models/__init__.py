"""Pre-built canonical GRN models — the ``crn_jax`` benchmark library.

Each submodule owns one model. They all expose the same surface:

* ``SPECIES`` — tuple of species names.
* ``Params`` — frozen dataclass of biochemical constants, with
  ``Params.easy()`` and ``Params.hard()`` classmethods for the two
  parameter regimes shipped in :doc:`library.json`. Defaults match the
  easy regime.
* ``propensities_fn(params)`` — closure ``(state, u) -> Array[M]`` ready
  to plug into :func:`crn_jax.simulate_trajectory`. All models are
  autonomous, so the ``u`` argument is accepted but ignored.
* ``apply_reaction(state, j)`` — JAX-compatible state update for reaction
  index ``j``, built from the module-level ``_STOICH`` matrix.
* ``simulate_dataset(key, …)`` — one-call helper that samples initial
  conditions, runs a JIT'd vmap'd batch simulator (cached per
  ``(n_steps, params)``), and returns a :class:`Dataset` with the full
  trajectory tensor ``xs`` of shape ``(n_replicates, n_steps, n_species)``
  plus flat one-step transitions ``(X_t, dX)``.

The packaged :doc:`library.json` is the authoritative source for
``params_easy`` / ``params_hard`` values and is asserted against the
``Params`` classmethods in :mod:`tests.test_models`.

Quickstart
----------
::

    import jax
    from crn_jax.models import repressilator

    ds = repressilator.simulate_dataset(jax.random.PRNGKey(0))
    ds.xs.shape       # (n_replicates, n_steps, 3)
    ds.species        # ("A", "B", "C")
    ds.X_t, ds.dX     # (n_replicates * n_steps, 3) flat transitions

Bring-your-own
--------------
The primitives plug straight into :func:`crn_jax.simulate_trajectory` when
the convenience helper isn't enough (custom schedules, non-uniform x0)::

    from crn_jax import simulate_trajectory
    from crn_jax.models import toggle

    p_fn = toggle.propensities_fn(toggle.Params.hard())
    state0 = toggle.State(time=0.0, x=jnp.array([0.0, 50.0]), next_reaction_time=jnp.inf)
    states = simulate_trajectory(
        ...,
        compute_propensities_fn=p_fn,
        apply_reaction_fn=toggle.apply_reaction,
    )
"""

from . import (
    activator_repressor,
    birth_death,
    bistable,
    coherent_ffl,
    cyclic_ring,
    incoherent_ffl,
    linear_chain,
    mutual_activation,
    negative_autoreg,
    positive_autoreg,
    repressilator,
    telegraph,
    toggle,
    two_stage,
)
from ._common import Dataset, State  # noqa: F401 — re-exported via __all__

# Convenience: an ordered tuple of every model module in the library.
# Useful for iterating ("for m in ALL_MODELS: ...") and for parameter
# consistency tests that loop over the JSON entries.
ALL_MODELS = (
    birth_death,
    two_stage,
    telegraph,
    negative_autoreg,
    positive_autoreg,
    bistable,
    linear_chain,
    toggle,
    activator_repressor,
    mutual_activation,
    coherent_ffl,
    incoherent_ffl,
    repressilator,
    cyclic_ring,
)

__all__ = sorted(["ALL_MODELS", "Dataset", "State", *(m.__name__.rsplit(".", 1)[-1] for m in ALL_MODELS)])
