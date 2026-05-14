"""A birth-death stochastic process.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  α                ν = (+1,)
    R1:  X → ∅     at rate  δ · X            ν = (-1,)

Stationary distribution is Poisson with mean ``⟨X⟩ = α / δ``.
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    alpha: float = 5.0
    delta: float = 1.0

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("X",)
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1,),  # R0: ∅ → X
    (-1,),  # R1: X → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        return jnp.array(
            [
                params.alpha,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
