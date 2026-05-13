"""Bistable self-activation — same reactions as :mod:`positive_autoreg`,
parameterised in the bistable regime (sharper Hill, lower leakage).

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Xⁿ / (Kⁿ + Xⁿ)        ν = (+1,)
    R1:  X → ∅     at rate  δ · X                            ν = (-1,)

Initial conditions sampled wide enough to populate both basins.
"""

import dataclasses
import functools
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function
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
    beta_0: float = 0.01
    beta_1: float = 8.0
    K: float = 1.0
    n: float = 2.0
    delta: float = 1.0

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(beta_0=0.001, beta_1=12.0, n=4.0)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        activation = hill_function(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * activation,
                params.delta * X,
            ]
        )

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
    n_steps: int = 1500,
    dt: float = 0.05,
    x0_dist: tuple = ("uniform", 0.0, 15.0),
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
