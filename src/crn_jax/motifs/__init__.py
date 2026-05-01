"""Standard GRN motifs for benchmarking system identification.

Each submodule here owns one canonical reaction network:

* :mod:`~crn_jax.motifs.inducible` — Hill-modulated birth-death.
* :mod:`~crn_jax.motifs.autoreg` — negative autoregulation (Hill repressor).
* :mod:`~crn_jax.motifs.cascade` — two-stage u → X → Y cascade.
* :mod:`~crn_jax.motifs.ffl_and` — C1 feed-forward loop with AND logic.

Each module exports the same five names: a NamedTuple ``State``, a frozen
``Params`` dataclass with truth defaults, a ``propensities_fn(params)``
factory returning the JAX propensity closure, a JAX-compatible
``apply_reaction(state, j)``, and a one-call ``simulate_dataset(key, ...)``
that returns a per-motif ``Dataset`` NamedTuple of trajectories plus flat
one-step transition triples.

Two usage patterns are supported.

**Convenience.** Generate a full dataset in one call::

    from crn_jax.motifs import cascade

    ds = cascade.simulate_dataset(jax.random.PRNGKey(0))
    ds.X_t, ds.Y_t, ds.u_per_triple, ds.dX, ds.dY  # ready for moment matching.

**BYO.** Pull just the propensity / reaction primitives and feed them
into :func:`crn_jax.simulate_trajectory` yourself when the convenience
helper isn't enough (e.g. custom u schedule, non-uniform x0)::

    from crn_jax import simulate_trajectory
    from crn_jax.motifs import cascade

    p_fn = cascade.propensities_fn(cascade.Params(beta_X=50.0))
    state0 = cascade.State(time=0.0, x=jnp.zeros(2), next_reaction_time=jnp.inf)
    states = simulate_trajectory(
        ..., compute_propensities_fn=p_fn,
        apply_reaction_fn=cascade.apply_reaction,
    )
"""

from __future__ import annotations

from . import autoreg, cascade, ffl_and, inducible

__all__ = ["autoreg", "cascade", "ffl_and", "inducible"]
