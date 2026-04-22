"""Gillespie algorithm for exact stochastic simulation of chemical kinetics.

This module implements the Stochastic Simulation Algorithm (SSA) for simulating
chemical reaction networks with discrete molecular counts.

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
``run_gillespie_loop`` therefore preserves pending reaction times across
simulation-interval boundaries. Specifically:

- When :math:`t + \\tau > t_{\\text{end}}`, the pending time :math:`t + \\tau`
  is returned alongside the state so the caller can thread it into the next
  call.
- When the control input (``action``) changes between calls, the pending time
  is invalidated (propensities have changed and the old schedule is stale).
- When the state changes because a reaction occurred, a fresh :math:`\\tau` is
  sampled automatically inside the loop.

Reference
---------
Gillespie, D. T. (1977). "Exact stochastic simulation of coupled chemical
reactions." *The Journal of Physical Chemistry*, 81(25), 2340-2361.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from .types import PRNGKey


def run_gillespie_loop(
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
