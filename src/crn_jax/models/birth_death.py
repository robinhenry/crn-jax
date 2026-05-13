"""Birth-death — minimal one-species stochastic baseline.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  α                ν = (+1,)
    R1:  X → ∅     at rate  δ · X            ν = (-1,)

Stationary distribution is Poisson with mean ``⟨X⟩ = α / δ``.
"""

import dataclasses
import functools
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..types import PRNGKey
from ._common import (
    Dataset,
    State,
    make_apply_reaction,
    make_vmap_simulator,
    run_dataset,
)

SPECIES: tuple[str, ...] = ("X",)
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1,),
    (-1,),
)


@dataclasses.dataclass(frozen=True)
class Params:
    alpha: float = 5.0
    delta: float = 1.0

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(alpha=1.0)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        return jnp.array([params.alpha, params.delta * X])

    return f


apply_reaction = make_apply_reaction(_STOICH)


@functools.lru_cache(maxsize=None)
def _build_simulator(n_steps: int, params: Params):
    return make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction)


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_replicates: int = 256,
    n_steps: int = 1000,
    dt: float = 0.05,
    x0_dist: tuple = ("uniform", 0.0, 10.0),
) -> Dataset:
    return run_dataset(
        key,
        species=SPECIES,
        simulator=_build_simulator(n_steps, params),
        n_replicates=n_replicates,
        n_steps=n_steps,
        dt=dt,
        x0_dists=(x0_dist,),
    )
