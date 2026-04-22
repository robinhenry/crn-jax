"""Ready-to-use wrappers around :func:`crn_jax.run_gillespie_loop`.

These are *templates*, not core functionality. They encode the common "advance
a state one fixed interval of wall-clock time" pattern so users doing routine
RL- or data-collection-style simulation don't have to wire the callables up
themselves. Drop down to :func:`crn_jax.run_gillespie_loop` if you need finer
control.

State contract
--------------
The helper here assumes that the state object is NamedTuple-like with

* a scalar ``time`` field,
* a scalar ``next_reaction_time`` field,
* a ``_replace(**kwargs)`` method returning an updated copy

(Python ``typing.NamedTuple`` and ``flax.struct.dataclass`` both satisfy this).

If your state shape is different, call :func:`crn_jax.run_gillespie_loop`
directly with custom ``get_time_fn`` / ``update_time_fn`` lambdas.
"""

from __future__ import annotations

from typing import Any, Callable

from jax import Array

from .core import run_gillespie_loop
from .types import PRNGKey


def step_interval(
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

    This is a thin convenience wrapper around :func:`run_gillespie_loop` that
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
        Updated state with ``time`` set to ``interval_start + timestep`` (or
        the time of the last reaction within the interval, whichever logic
        your ``update_time_fn`` implements) and ``next_reaction_time`` set to
        the next scheduled reaction time.
    """
    if interval_start is None:
        interval_start = state.time
    target_time = interval_start + timestep

    final_state, next_reaction_time = run_gillespie_loop(
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
    # ``run_gillespie_loop`` leaves ``state.time`` at the time of the last
    # reaction that fired inside the interval (or at ``interval_start`` if
    # none fired). For iterated RL-style stepping we want it to reflect
    # wall-clock, so advance to ``target_time`` here.
    return final_state._replace(time=target_time, next_reaction_time=next_reaction_time)
