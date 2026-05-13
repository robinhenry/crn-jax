"""Birth-death — minimal one-species stochastic baseline.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  α                ν = (+1,)
    R1:  X → ∅     at rate  δ · X            ν = (-1,)

Stationary distribution is Poisson with mean ``⟨X⟩ = α / δ``.
"""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ._common import (
    State,
    make_apply_reaction,
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
