"""Shared building blocks for the standard GRN models.

Every model module in this package follows the same skeleton:

* a frozen :class:`Params` dataclass with ``Params.easy()`` / ``Params.hard()``
  classmethods drawn from the packaged :mod:`library.json`,
* a ``propensities_fn(params) -> (state, u) -> Array[M]`` closure factory,
* a module-level stoichiometry matrix plus an :func:`apply_reaction` built
  from it,
* an ``@functools.lru_cache``-d ``_build_simulator(n_steps, params)`` that
  returns a JIT'd batch simulator, and
* a one-call ``simulate_dataset(key, …)`` that delegates to :func:`run_dataset`.

This file owns the shared state shape (:class:`State`), the shared output
shape (:class:`Dataset`), the JIT+vmap batch-simulator factory
(:func:`make_vmap_simulator`), initial-condition sampling, and the glue
function :func:`run_dataset` that the per-model ``simulate_dataset``
wrappers call into.

All 14 models in this library are autonomous: their ``propensities_fn``
takes ``(state, u)`` only because the Gillespie driver's signature requires
it, and the ``u`` argument is ignored. :func:`make_vmap_simulator` calls
:func:`simulate_trajectory` with ``inputs=None`` so the driver skips the
input-change invalidation it would otherwise perform.
"""

from typing import Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..gillespie import simulate_trajectory
from ..types import PRNGKey

# --- State shape -------------------------------------------------------------
# One State class works for every model: the species count lives in the
# runtime shape of ``x`` (always a length-``n_species`` 1-D vector),
# not in the type. ``x[i]`` is the count of the i-th species named in the
# model's ``SPECIES`` tuple.


class State(NamedTuple):
    """State carried through the Gillespie SSA loop.

    ``x`` is always a 1-D ``(n_species,)`` JAX array, even for single-species
    models — this keeps :func:`run_dataset` uniform across the library.
    """

    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


# --- Output shape ------------------------------------------------------------


class Dataset(NamedTuple):
    """Output of every model's :func:`simulate_dataset`.

    All arrays are NumPy (host-side) because downstream usage is
    bin/analyse, not further JAX work. The per-species trajectory tensor
    ``xs`` and the flat one-step transition arrays ``X_t`` / ``dX`` make
    both shapes available without recomputation.
    """

    times: np.ndarray  # (n_steps,) — sample times in `dt` units
    species: tuple[str, ...]  # species names, len == n_species
    x0: np.ndarray  # (n_replicates, n_species) — sampled initial counts
    xs: np.ndarray  # (n_replicates, n_steps, n_species) — full trajectories
    X_t: np.ndarray  # (n_replicates * n_steps, n_species) — flat per-step state
    dX: np.ndarray  # (n_replicates * n_steps, n_species) — flat one-step delta


# --- vmap-jit simulator factory ---------------------------------------------


class BatchSimulator(Protocol):
    """The function returned by :func:`make_vmap_simulator`.

    Given per-replicate keys and initial ``state.x`` values (shape
    ``(n_replicates, n_species)``) plus a shared timestep, returns a
    stacked :class:`State` PyTree with a leading replicate axis.
    """

    def __call__(
        self,
        keys: jax.Array,
        x0_batch: jax.Array,
        dt: float | jax.Array,
    ) -> State: ...


def make_vmap_simulator(
    n_steps: int,
    propensities_fn: Callable[[State, jax.Array], jax.Array],
    apply_reaction_fn: Callable[[State, jax.Array], State],
) -> BatchSimulator:
    """Build a JIT'd vmap'd batch simulator for a fixed model and ``n_steps``.

    ``simulate_trajectory`` requires ``n_steps`` to be a Python int (it sets
    ``jax.lax.scan``'s trip count). Wrapping it in a closure makes
    ``n_steps`` a compile-time constant while still letting the caller vary
    it across calls.

    Args:
        n_steps: Number of intervals each trajectory runs for.
        propensities_fn: Closure ``(state, u) -> Array[M]`` already bound to
            its model parameters.
        apply_reaction_fn: ``(state, reaction_idx) -> state`` for the model.

    Returns:
        A :class:`BatchSimulator` that vmaps the per-trajectory simulation
        over ``keys`` and ``x0_batch``. ``inputs=None`` is forwarded to
        :func:`simulate_trajectory` so the driver skips per-step
        input-change invalidation (all models in this library are
        autonomous).
    """

    def simulate_one(key, x0, dt):
        state0 = State(
            time=jnp.array(0.0),
            x=x0,
            next_reaction_time=jnp.array(jnp.inf),
        )
        return simulate_trajectory(
            key=key,
            initial_state=state0,
            timestep=dt,
            n_steps=n_steps,
            compute_propensities_fn=propensities_fn,
            apply_reaction_fn=apply_reaction_fn,
            inputs=None,
        )

    @jax.jit
    def run(keys, x0_batch, dt):
        return jax.vmap(simulate_one, in_axes=(0, 0, None))(keys, x0_batch, dt)

    return run


# --- Stoichiometry-driven reaction application -------------------------------


def make_apply_reaction(stoichiometry: tuple[tuple[int, ...], ...]) -> Callable[[State, jax.Array], State]:
    """Build an ``apply_reaction(state, j)`` from a stoichiometry matrix.

    ``stoichiometry`` has one row per reaction and one column per species;
    row ``j`` is the state delta when reaction ``j`` fires. Counts are
    floored at zero defensively (no reaction in this library can push a
    species negative when fired with non-zero propensity, but the floor
    protects against floating-point edge cases at very low counts).
    """
    stoich = jnp.asarray(stoichiometry, dtype=jnp.float32)

    def apply_reaction(state: State, j: jax.Array) -> State:
        return state._replace(x=jnp.maximum(0.0, state.x + stoich[j]))

    return apply_reaction


# --- Initial-condition sampling ---------------------------------------------

DistSpec = tuple  # ("uniform", lo, hi) | ("zero",) | ("bernoulli", p) | ("constant", v)


def sample_initial_state(
    key: jax.Array,
    shape: tuple[int, ...],
    dist: DistSpec,
) -> jax.Array:
    """Sample initial conditions for one species across replicates.

    Supported distributions:

    * ``("uniform", lo, hi)`` — continuous uniform over ``[lo, hi)``.
    * ``("zero",)`` — constant zero.
    * ``("constant", v)`` — constant ``v`` (useful when you want a
      non-zero deterministic IC for one species).
    * ``("bernoulli", p)`` — 0/1 draws with ``P(1) = p``. Used by
      :mod:`crn_jax.models.telegraph` for the binary promoter state.

    Callers needing fancier samplers should sample ``x0`` themselves and
    skip ``simulate_dataset`` in favour of the BYO path that drives
    :func:`crn_jax.simulate_trajectory` directly.
    """
    kind = dist[0]
    if kind == "uniform":
        _, lo, hi = dist
        return jax.random.uniform(key, shape, minval=float(lo), maxval=float(hi))
    if kind == "zero":
        return jnp.zeros(shape)
    if kind == "constant":
        _, v = dist
        return jnp.full(shape, float(v))
    if kind == "bernoulli":
        _, p = dist
        return (jax.random.uniform(key, shape) < float(p)).astype(jnp.float32)
    raise ValueError(f"Unknown distribution spec: {dist!r}")


# --- Trajectory flattening + one-call dataset --------------------------------


def flatten_species(x0: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a multi-species trajectory tensor into one-step transition triples.

    Args:
        x0: Initial counts, shape ``(n_replicates, n_species)``.
        xs: Full trajectories, shape ``(n_replicates, n_steps, n_species)``.

    Returns:
        ``(X_t, dX)`` each of shape ``(n_replicates * n_steps, n_species)``,
        where ``X_t[k]`` is the state at step ``k`` of some replicate and
        ``dX[k] = X_t[k+1] - X_t[k]`` (or ``xs[..., 0] - x0`` for the very
        first triple of each replicate).
    """
    x_full = np.concatenate([x0[:, None, :], xs], axis=1)
    n_species = x_full.shape[-1]
    X_t = x_full[:, :-1, :].reshape(-1, n_species).astype(np.float32)
    X_next = x_full[:, 1:, :].reshape(-1, n_species).astype(np.float32)
    dX = (X_next - X_t).astype(np.float32)
    return X_t, dX


def run_dataset(
    key: PRNGKey,
    *,
    species: tuple[str, ...],
    simulator: BatchSimulator,
    n_replicates: int,
    n_steps: int,
    dt: float,
    x0_dists: tuple[DistSpec, ...],
) -> Dataset:
    """Run a model's batch simulator and pack the results into a :class:`Dataset`.

    Per-model ``simulate_dataset`` functions are thin wrappers around this
    helper: they choose model-specific defaults for ``n_replicates``,
    ``n_steps``, ``dt``, and the ``x0_dists`` tuple (one DistSpec per
    species), then forward everything here.
    """
    if len(x0_dists) != len(species):
        raise ValueError(
            f"x0_dists has length {len(x0_dists)} but species has length {len(species)}; "
            "one IC distribution per species is required.",
        )

    n_species = len(species)
    key_sim, *key_x0 = jax.random.split(key, n_species + 1)

    x0_cols = [sample_initial_state(k, (n_replicates,), dist) for k, dist in zip(key_x0, x0_dists, strict=True)]
    x0 = jnp.stack(x0_cols, axis=-1)

    keys = jax.random.split(key_sim, n_replicates)
    states = simulator(keys, x0, dt)

    xs = np.asarray(states.x)
    x0_np = np.asarray(x0)
    X_t, dX = flatten_species(x0_np, xs)

    times = np.asarray(jnp.arange(1, n_steps + 1) * dt)
    return Dataset(
        times=times,
        species=tuple(species),
        x0=x0_np,
        xs=xs,
        X_t=X_t,
        dX=dX,
    )
