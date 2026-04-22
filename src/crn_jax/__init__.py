"""crn-jax: chemical reaction networks in JAX.

Currently this package ships an exact Gillespie SSA driver
(:func:`run_gillespie_loop`) plus a handful of ergonomic helpers. The scope
is "stochastic reaction-network simulation on the GPU"; future releases are
expected to add τ-leaping, the modified next-reaction method, and a Chemical
Langevin Equation driver under this same umbrella.

Quickstart
----------
>>> import jax, jax.numpy as jnp
>>> from typing import NamedTuple
>>> from crn_jax import run_gillespie_loop
>>>
>>> class State(NamedTuple):
...     time: jax.Array
...     x: jax.Array
...
>>> def propensities(s, a):  # birth at rate 1, death at rate 0.1 * x
...     return jnp.array([1.0, 0.1 * s.x])
>>>
>>> def apply_reaction(s, j):
...     dx = jnp.where(j == 0, 1.0, -1.0)
...     return s._replace(x=s.x + dx)
>>>
>>> state = State(time=jnp.array(0.0), x=jnp.array(0.0))
>>> final, _ = run_gillespie_loop(
...     key=jax.random.PRNGKey(0),
...     initial_state=state,
...     action=jnp.array(0.0),
...     target_time=100.0,
...     max_steps=10_000,
...     compute_propensities_fn=propensities,
...     apply_reaction_fn=apply_reaction,
...     get_time_fn=lambda s: s.time,
...     update_time_fn=lambda s, t: s._replace(time=t),
... )
"""

from .core import run_gillespie_loop
from .plotting import plot_trajectories
from .types import PRNGKey

__all__ = ["run_gillespie_loop", "plot_trajectories", "PRNGKey"]

__version__ = "0.1.0"
