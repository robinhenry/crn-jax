"""Shared building blocks for the standard GRN models.

Every model module in this package is just the math:

* ``SPECIES`` — tuple of species names.
* ``_STOICH`` — reaction-by-species stoichiometry matrix.
* ``Params`` — frozen dataclass with ``easy()`` / ``hard()`` factories.
* ``propensities_fn(params) -> (state, u) -> Array[M]`` closure factory.
* ``apply_reaction = make_apply_reaction(_STOICH)``.

The one-call entry point is :func:`simulate_dataset` in this module; it
takes a model module plus caller-supplied ``x0`` and returns a
:class:`Dataset`. Default timescale is ``n_steps=1000, dt=0.1`` for every
model; the caller picks something appropriate per system.

Every model in this library is autonomous: its ``propensities_fn`` takes
``(state, u)`` only because the Gillespie driver's signature requires it,
and the ``u`` argument is ignored. :func:`make_vmap_simulator` calls
:func:`simulate_trajectory` with ``inputs=None`` so the driver skips the
input-change invalidation it would otherwise perform.
"""

import functools
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
    models — this keeps :func:`simulate_dataset` uniform across the library.
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
    x0: np.ndarray  # (n_replicates, n_species) — initial counts supplied by caller
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
        input-change invalidation (every model in this library is
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


# --- Top-level entry point --------------------------------------------------


class ModelModule(Protocol):
    """Structural contract for a model module passed to :func:`simulate_dataset`.

    Every module in :data:`crn_jax.models.ALL_MODELS` satisfies this Protocol;
    custom user-defined models can too, without subclassing anything.
    """

    SPECIES: tuple[str, ...]
    Params: type

    @staticmethod
    def propensities_fn(params) -> Callable[[State, jax.Array], jax.Array]: ...

    @staticmethod
    def apply_reaction(state: State, j: jax.Array) -> State: ...


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
    X_t = x_full[:, :-1, :].reshape(-1, n_species)
    X_next = x_full[:, 1:, :].reshape(-1, n_species)
    dX = X_next - X_t
    return X_t, dX


@functools.lru_cache(maxsize=128)
def _cached_batch_simulator(
    propensities_factory: Callable,
    apply_reaction_fn: Callable,
    params,
    n_steps: int,
) -> BatchSimulator:
    """Build (and cache) a JIT'd vmap'd batch simulator for one model.

    Cache key is ``(propensities_factory, apply_reaction_fn, params,
    n_steps)``. The factory and apply-reaction are functions (hashable by
    identity); ``params`` is a frozen dataclass (hashable by value). This
    means repeated ``simulate_dataset`` calls with the same model and
    same params reuse the compiled XLA artifact.

    ``maxsize=128`` caps memory growth in long-running sessions that sweep
    many distinct ``(params, n_steps)`` combinations (each entry is a
    compiled XLA artifact that can be tens of MB).
    """
    return make_vmap_simulator(n_steps, propensities_factory(params), apply_reaction_fn)


def simulate_dataset(
    model: ModelModule,
    key: PRNGKey,
    x0: jax.Array,
    *,
    params=None,
    n_steps: int = 1000,
    dt: float = 0.1,
) -> Dataset:
    """Run ``model`` on caller-supplied ``x0`` and return a :class:`Dataset`.

    Args:
        model: A model module from :mod:`crn_jax.models` (or any object
            satisfying the :class:`ModelModule` Protocol).
        key: PRNG key threaded into the per-replicate Gillespie scans.
        x0: ``(n_replicates, len(model.SPECIES))`` non-negative initial
            counts. ``n_replicates`` is inferred from ``x0.shape[0]``.
            The library deliberately does not sample ICs for you; the
            sensible distribution is problem-specific.
        params: Optional model parameters; defaults to ``model.Params()``
            (the easy regime).
        n_steps: Number of fixed-interval steps to scan. Default 1000.
        dt: Width of each interval. Default 0.1.

    Raises:
        ValueError: ``x0`` has the wrong shape or any negative entry.
    """
    if params is None:
        params = model.Params()

    x0 = jnp.asarray(x0, dtype=jnp.float32)
    n_species = len(model.SPECIES)
    if x0.ndim != 2 or x0.shape[1] != n_species:
        raise ValueError(
            f"x0 must have shape (n_replicates, {n_species}) for species {model.SPECIES}; got shape {tuple(x0.shape)}.",
        )
    if jnp.any(x0 < 0).item():
        raise ValueError("x0 must be non-negative (species counts can't be negative).")

    simulator = _cached_batch_simulator(model.propensities_fn, model.apply_reaction, params, n_steps)
    n_replicates = x0.shape[0]
    keys = jax.random.split(key, n_replicates)
    states = simulator(keys, x0, dt)

    xs = np.asarray(states.x)
    x0_np = np.asarray(x0)
    X_t, dX = flatten_species(x0_np, xs)

    times = np.arange(1, n_steps + 1) * dt
    return Dataset(
        times=times,
        species=tuple(model.SPECIES),
        x0=x0_np,
        xs=xs,
        X_t=X_t,
        dX=dX,
    )
