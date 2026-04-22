"""Gillespie SSA driver plus ergonomic wrappers for fixed-interval stepping.

This module ships both the low-level algorithm and a couple of thin
convenience layers built on top of it:

* :func:`simulate_until` — exact Stochastic Simulation Algorithm for
  chemical reaction networks, operating on an arbitrary state PyTree.
* :func:`simulate_interval` — advance a state by one fixed-length interval,
  threading the pending reaction time through state fields.
* :func:`simulate_trajectory` — scan :func:`simulate_interval` for ``n_steps``
  intervals and stack the per-step states.

Future stochastic drivers (τ-leaping, next-reaction method, CLE) will live
alongside this module under their own filenames.

Mathematical Background
-----------------------
For a system with M reaction channels and propensities :math:`a_j(x)` where
:math:`x` is the state vector, the Gillespie algorithm samples from the
Chemical Master Equation:

1. **Time to next reaction**: :math:`\\tau \\sim \\mathrm{Exp}(a_0)` where
   :math:`a_0 = \\sum_{j=1}^{M} a_j`.
2. **Which reaction occurs**: :math:`P(\\text{reaction } j) = a_j / a_0`.

The exponential distribution has the memoryless property:
:math:`P(\\tau > t + s \\mid \\tau > t) = P(\\tau > s)`.

This means that if :math:`\\tau` is sampled at time :math:`t` and the reaction
is scheduled for :math:`t + \\tau`, but the simulation is advanced only to some
boundary :math:`t_{\\text{end}} < t + \\tau`, then:

- The reaction is still pending at :math:`t + \\tau`.
- It should NOT be resampled when continuing from :math:`t_{\\text{end}}`.
- Resampling is statistically equivalent but physically incorrect: it makes
  trajectories depend on the simulation discretisation boundaries.

Implementation Design
---------------------
:func:`simulate_until` therefore preserves pending reaction times across
simulation-interval boundaries. Specifically:

- When :math:`t + \\tau > t_{\\text{end}}`, the pending time :math:`t + \\tau`
  is returned alongside the state so the caller can thread it into the next
  call.
- When the control input (``action``) changes between calls, the pending time
  is invalidated (propensities have changed and the old schedule is stale).
- When the state changes because a reaction occurred, a fresh :math:`\\tau` is
  sampled automatically inside the loop.

State contract for the wrappers
-------------------------------
:func:`simulate_interval` and :func:`simulate_trajectory` assume that the state object
is NamedTuple-like with

* a scalar ``time`` field,
* a scalar ``next_reaction_time`` field,
* a ``_replace(**kwargs)`` method returning an updated copy

(Python ``typing.NamedTuple`` and ``flax.struct.dataclass`` both satisfy this).

If your state shape is different, call :func:`simulate_until` directly
with custom ``get_time_fn`` / ``update_time_fn`` lambdas.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from .types import PRNGKey


def simulate_until(
    key: PRNGKey,
    initial_state: Any,
    action: Array,
    target_time: float | Array,
    max_steps: int,
    compute_propensities_fn: Callable[[Any, Array], Array],
    apply_reaction_fn: Callable[[Any, Array], Any],
    get_time_fn: Callable[[Any], Array],
    update_time_fn: Callable[[Any, float | Array], Any],
    pending_reaction_time: Array | None = None,
    previous_action: Array | None = None,
) -> tuple[Any, Array]:
    """Execute Gillespie SSA until ``target_time``, preserving pending reactions.

    Simulates the chemical system from the current time until ``target_time``,
    executing reactions as they occur. Pending reaction times are preserved
    across calls so that trajectories are independent of the simulation
    discretisation boundaries.

    Args:
        key: PRNG key for stochastic sampling.
        initial_state: Starting state (arbitrary PyTree). The only structural
            requirement is that ``get_time_fn`` / ``update_time_fn`` can read
            and update a scalar time field from it.
        action: Current control input. Passed to ``compute_propensities_fn``
            and used (together with ``previous_action`` if supplied) to decide
            whether the pending reaction time is still valid.
        target_time: Simulate until this absolute time.
        max_steps: Safety upper bound on the number of reactions in this call.
        compute_propensities_fn: ``(state, action) -> Array[M]`` returning the
            non-negative propensities for M reaction channels. The caller is
            expected to close over any extra parameters (kinetic constants,
            config, etc.).
        apply_reaction_fn: ``(state, reaction_idx) -> state`` applying a single
            reaction to the state. Must be JAX-compatible (e.g. use
            ``jax.lax.switch`` internally, not Python ``if``).
        get_time_fn: ``state -> Array`` reading the scalar simulation time.
        update_time_fn: ``(state, time) -> state`` writing the scalar time.
        pending_reaction_time: Scheduled reaction time from the previous call,
            or ``None`` / ``inf`` to sample fresh.
        previous_action: Action from the previous call. If provided and
            different from ``action``, the pending reaction time is
            invalidated (propensities have changed).

    Returns:
        final_state: State after simulating to ``target_time``.
        next_reaction_time: Scheduled time of the next reaction, which may be
            larger than ``target_time`` (in which case the caller should thread
            it into the next call as ``pending_reaction_time``).
    """

    def sample_tau(key: PRNGKey, propensities: Array) -> Array:
        """Sample time until next reaction: :math:`\\tau \\sim \\mathrm{Exp}(a_0)`."""
        a0 = jnp.sum(propensities)
        return jax.random.exponential(key) / jnp.maximum(a0, 1e-10)

    def sample_reaction(key: PRNGKey, propensities: Array) -> Array:
        """Sample which reaction: :math:`P(j) = a_j / a_0`."""
        a0 = jnp.sum(propensities)
        probs = propensities / jnp.maximum(a0, 1e-10)
        return jax.random.choice(key, len(propensities), p=probs)

    if pending_reaction_time is None:
        pending_reaction_time = jnp.array(jnp.inf)

    # Invalidate pending reaction if the action changed (propensities differ).
    # Element-wise inequality is used so this is correct for both binary and
    # continuous actions — a boolean XOR would silently cast floats to bool.
    if previous_action is not None:
        action_changed = jnp.any(jnp.asarray(previous_action) != jnp.asarray(action))
        pending_reaction_time = jnp.where(
            action_changed, jnp.array(jnp.inf), pending_reaction_time
        )

    # Sample an initial reaction time if we don't have one carried over.
    initial_time = get_time_fn(initial_state)
    key, key_init = jax.random.split(key)
    needs_sample = jnp.isinf(pending_reaction_time)
    initial_propensities = compute_propensities_fn(initial_state, action)
    sampled_tau = sample_tau(key_init, initial_propensities)
    next_reaction_time = jnp.where(
        needs_sample, initial_time + sampled_tau, pending_reaction_time
    )

    def cond_fn(carry):
        state, next_rxn_time, step, key = carry
        return (next_rxn_time < target_time) & (step < max_steps)

    def body_fn(carry):
        state, next_rxn_time, step, key = carry
        key, key_reaction, key_time = jax.random.split(key, 3)

        # Advance time to when the reaction occurs.
        state = update_time_fn(state, next_rxn_time)

        # Sample which reaction (using pre-reaction propensities) and apply it.
        propensities = compute_propensities_fn(state, action)
        reaction_idx = sample_reaction(key_reaction, propensities)
        state = apply_reaction_fn(state, reaction_idx)

        # Post-reaction propensities have changed; sample the next tau.
        new_propensities = compute_propensities_fn(state, action)
        tau = sample_tau(key_time, new_propensities)
        new_next_rxn_time = get_time_fn(state) + tau

        return state, new_next_rxn_time, step + 1, key

    final_state, final_next_rxn_time, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (initial_state, next_reaction_time, jnp.array(0), key),
    )

    return final_state, final_next_rxn_time


def simulate_interval(
    key: PRNGKey,
    state: Any,
    action: Array,
    *,
    timestep: float | Array,
    max_steps: int,
    compute_propensities_fn: Callable[[Any, Array], Array],
    apply_reaction_fn: Callable[[Any, Array], Any],
    previous_action: Array | None = None,
    interval_start: float | Array | None = None,
) -> Any:
    """Advance a state by one fixed-length interval using the Gillespie SSA.

    This is a thin convenience wrapper around :func:`simulate_until` that
    reads / writes ``state.time`` and ``state.next_reaction_time`` for you.

    Args:
        key: PRNG key.
        state: Current state; must expose ``time``, ``next_reaction_time``,
            and ``_replace`` (see module docstring).
        action: Control input passed to ``compute_propensities_fn`` and used
            to invalidate the pending reaction time when it changes.
        timestep: Length of the interval to simulate.
        max_steps: Safety upper bound on reactions executed in this call.
        compute_propensities_fn: ``(state, action) -> Array[M]``.
        apply_reaction_fn: ``(state, reaction_idx) -> state``.
        previous_action: Optional — action from the previous interval.
        interval_start: Absolute start time of this interval. Defaults to
            ``state.time`` (suitable for free-running simulation).

    Returns:
        Updated state with ``time`` set to ``interval_start + timestep`` and
        ``next_reaction_time`` set to the next scheduled reaction time.
    """
    if interval_start is None:
        interval_start = state.time
    target_time = interval_start + timestep

    final_state, next_reaction_time = simulate_until(
        key=key,
        initial_state=state,
        action=action,
        target_time=target_time,
        max_steps=max_steps,
        compute_propensities_fn=compute_propensities_fn,
        apply_reaction_fn=apply_reaction_fn,
        get_time_fn=lambda s: s.time,
        update_time_fn=lambda s, t: s._replace(time=t),
        pending_reaction_time=state.next_reaction_time,
        previous_action=previous_action,
    )
    # ``simulate_until`` leaves ``state.time`` at the time of the last
    # reaction that fired inside the interval (or at ``interval_start`` if
    # none fired). For iterated RL-style stepping we want it to reflect
    # wall-clock, so advance to ``target_time`` here.
    return final_state._replace(time=target_time, next_reaction_time=next_reaction_time)


def simulate_trajectory(
    key: PRNGKey,
    initial_state: Any,
    *,
    timestep: float | Array,
    n_steps: int,
    compute_propensities_fn: Callable[[Any, Array], Array],
    apply_reaction_fn: Callable[[Any, Array], Any],
    actions: Array | None = None,
    max_steps: int = 10_000,
) -> Any:
    """Scan :func:`simulate_interval` ``n_steps`` times and stack the per-step states.

    This is a thin ``jax.lax.scan`` wrapper that advances the state in
    ``n_steps`` fixed-length intervals and records the state at the end of
    each interval.

    Args:
        key: PRNG key (split internally into ``n_steps`` sub-keys).
        initial_state: Starting state; must satisfy the state contract
            documented at the top of this module.
        timestep: Length of each interval.
        n_steps: Number of intervals to simulate. Must be a Python ``int``
            (it fixes ``jax.lax.scan``'s trip count).
        compute_propensities_fn: ``(state, action) -> Array[M]``.
        apply_reaction_fn: ``(state, reaction_idx) -> state``.
        actions: Optional ``(n_steps, ...)`` array of per-step control
            inputs. If ``None``, a zero scalar is used for every interval
            and no action-change invalidation is performed. If supplied,
            the previous action is threaded through the scan so that
            changing actions correctly invalidate pending reactions.
        max_steps: Safety upper bound on reactions executed per interval.

    Returns:
        A PyTree with the same structure as ``initial_state`` whose leaves
        have an additional leading dimension of size ``n_steps``. Element
        ``i`` along that axis is the state at time
        ``initial_state.time + (i + 1) * timestep``.
    """
    keys = jax.random.split(key, n_steps)

    if actions is None:
        zero_action = jnp.array(0.0)

        def body(state, k):
            new_state = simulate_interval(
                key=k,
                state=state,
                action=zero_action,
                timestep=timestep,
                max_steps=max_steps,
                compute_propensities_fn=compute_propensities_fn,
                apply_reaction_fn=apply_reaction_fn,
            )
            return new_state, new_state

        _, states = jax.lax.scan(body, initial_state, keys)
        return states

    actions = jnp.asarray(actions)
    if actions.shape[0] != n_steps:
        raise ValueError(
            f"actions leading dimension ({actions.shape[0]}) must match "
            f"n_steps ({n_steps})"
        )

    def body_with_action(carry, inputs):
        state, prev_action = carry
        k, action = inputs
        new_state = simulate_interval(
            key=k,
            state=state,
            action=action,
            timestep=timestep,
            max_steps=max_steps,
            compute_propensities_fn=compute_propensities_fn,
            apply_reaction_fn=apply_reaction_fn,
            previous_action=prev_action,
        )
        return (new_state, action), new_state

    # Seeding prev_action with actions[0] means the first step sees "no change"
    # and trusts the pending reaction stored in initial_state.
    (_, _), states = jax.lax.scan(
        body_with_action, (initial_state, actions[0]), (keys, actions)
    )
    return states
