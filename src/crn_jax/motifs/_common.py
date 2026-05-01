"""Shared building blocks for the standard GRN motifs.

The motif modules in this package all share the same simulation skeleton: a NamedTuple ``State``
with a ``time`` / ``x`` / ``next_reaction_time`` triple, a per-replicate
``simulate_one`` closure that calls :func:`crn_jax.simulate_trajectory`,
and a ``vmap`` over independent replicates. This module factors those
ingredients out so each motif file only owns its propensity function,
``apply_reaction`` dispatch, default parameters, and the ``simulate_dataset``
glue that picks initial conditions and inputs.
"""

from typing import Callable, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from ..gillespie import simulate_trajectory

# --- State shape -------------------------------------------------------------
# The Gillespie driver only requires a NamedTuple-like state with `time`,
# `next_reaction_time`, and `_replace`. The species count is encoded in the
# *runtime shape* of `x` (scalar / length-2 / length-3) rather than in the
# type — there is nothing motif-specific at the static level, so one State
# class covers all four motifs.


class State(NamedTuple):
    """State carried through the Gillespie SSA loop.

    The shape of ``x`` is motif-dependent:

    * scalar for 1-species motifs (``inducible``, ``autoreg``),
    * length-2 vector ``[X, Y]`` for ``cascade``,
    * length-3 vector ``[X, Y, Z]`` for ``ffl_and``.
    """

    time: jax.Array
    x: jax.Array
    next_reaction_time: jax.Array


# --- vmap-jit simulator factory ---------------------------------------------


class BatchSimulator(Protocol):
    """The function returned by :func:`make_vmap_simulator`.

    Given per-replicate keys, initial ``state.x`` values, a shared timestep,
    and per-replicate scalar inputs, returns a stacked :class:`State`
    PyTree with a leading replicate axis.
    """

    def __call__(
        self,
        keys: jax.Array,
        x0_batch: jax.Array,
        dt: float | jax.Array,
        u_batch: jax.Array,
    ) -> State: ...


def make_vmap_simulator(
    n_steps: int,
    propensities_fn: Callable[[State, jax.Array], jax.Array],
    apply_reaction_fn: Callable[[State, jax.Array], State],
) -> BatchSimulator:
    """Build a JIT'd vmap'd batch simulator for a fixed motif and ``n_steps``.

    ``simulate_trajectory`` requires ``n_steps`` to be a Python int (it sets
    ``jax.lax.scan``'s trip count). Wrapping it in a closure makes
    ``n_steps`` a compile-time constant while still letting the caller vary
    it across calls.

    Args:
        n_steps: Number of intervals each trajectory runs for.
        propensities_fn: Closure ``(state, u) -> Array[M]`` already bound to
            its motif parameters (see e.g. :func:`inducible.propensities_fn`).
        apply_reaction_fn: ``(state, reaction_idx) -> state`` for the motif.

    Returns:
        A :class:`BatchSimulator` that vmaps the per-trajectory simulation
        over ``keys``, ``x0_batch`` (per-replicate initial ``state.x``), and
        ``u_batch`` (per-replicate scalar input, held constant across the
        trajectory). For input-free motifs pass ``jnp.zeros((n_replicates,))``.

        The constant-u-per-trajectory inputs array ``jnp.full((n_steps,), u)``
        is materialised *inside* the vmapped per-replicate function. That
        matches the per-replicate full-array pattern users typically write by
        hand. Pre-broadcasting outside the vmap (via ``jnp.broadcast_to``)
        traces to a different XLA program whose floating-point reduction
        order differs in a tiny fraction of replicates, causing some
        trajectories to diverge — so we keep the materialisation per-replicate.
    """

    def simulate_one(key, x0, dt, u_scalar):
        state0 = State(
            time=jnp.array(0.0),
            x=x0,
            next_reaction_time=jnp.array(jnp.inf),
        )
        inputs = jnp.full((n_steps,), u_scalar)
        return simulate_trajectory(
            key=key,
            initial_state=state0,
            timestep=dt,
            n_steps=n_steps,
            compute_propensities_fn=propensities_fn,
            apply_reaction_fn=apply_reaction_fn,
            inputs=inputs,
        )

    @jax.jit
    def run(keys, x0_batch, dt, u_batch):
        return jax.vmap(simulate_one, in_axes=(0, 0, None, 0))(keys, x0_batch, dt, u_batch)

    return run


# --- Initial-condition sampling ---------------------------------------------

DistSpec = tuple  # ("uniform", lo, hi) | ("zero",)


def sample_initial_state(
    key: jax.Array,
    shape: tuple[int, ...],
    dist: DistSpec,
) -> jax.Array:
    """Sample initial conditions for one species across replicates.

    Supported distributions:

    * ``("uniform", lo, hi)``: continuous uniform over ``[lo, hi)``.
    * ``("zero",)``: constant zero.

    Mixture / stratified samplers are intentionally out of scope for the
    v0.2 motif library — they're experimental-design knobs, not motif
    specifications. Callers needing them should sample ``x0`` themselves
    and skip ``simulate_dataset`` in favour of the BYO path.
    """
    kind = dist[0]
    if kind == "uniform":
        _, lo, hi = dist
        return jax.random.uniform(key, shape, minval=float(lo), maxval=float(hi))
    if kind == "zero":
        return jnp.zeros(shape)
    raise ValueError(f"Unknown distribution spec: {dist!r}")


# --- Triple flattening -------------------------------------------------------


def flatten_species(x0: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn ``(n_replicates,)`` initial values + ``(n_replicates, n_steps)`` trajectories
    into flat ``(X_t, dX)`` arrays of length ``n_replicates * n_steps``.

    Each trajectory contributes ``n_steps`` one-step transitions:
    ``X_t[k]`` is the state at step ``k`` and ``dX[k] = X_t[k+1] - X_t[k]``.
    """
    x_full = np.concatenate([x0[:, None], xs], axis=1)  # (n_replicates, n_steps + 1)
    X_t = x_full[:, :-1].reshape(-1).astype(np.float32)
    X_next = x_full[:, 1:].reshape(-1).astype(np.float32)
    dX = (X_next - X_t).astype(np.float32)
    return X_t, dX


def repeat_input_per_triple(u_arr: np.ndarray, n_steps: int) -> np.ndarray:
    """Broadcast a constant-per-trajectory input to per-triple values."""
    return np.repeat(u_arr.astype(np.float32), n_steps)
