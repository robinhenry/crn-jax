"""Telegraph promoter — two-state bursting model for single-cell expression.

State: three species ``[S, M, P]`` where ``S ∈ {0, 1}`` is the promoter
state (off / on), ``M`` is mRNA count, and ``P`` is protein count.

Reactions
---------
    R0:  S=0 → S=1         at rate  k_on · (1 − S)         ν = (+1,  0,  0)
    R1:  S=1 → S=0         at rate  k_off · S              ν = (-1,  0,  0)
    R2:  S=1 → S=1 + M     at rate  β · S                  ν = ( 0, +1,  0)
    R3:  M       → ∅       at rate  γ_M · M                ν = ( 0, -1,  0)
    R4:  M       → M + P   at rate  k_tl · M               ν = ( 0,  0, +1)
    R5:  P       → ∅       at rate  δ_P · P                ν = ( 0,  0, -1)

Sources
-------
* https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012118
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    k_on: float = 1.0
    k_off: float = 1.0
    beta: float = 5.0
    gamma_M: float = 1.0
    k_tl: float = 10.0
    delta_P: float = 0.2

    @classmethod
    def easy(cls) -> Self:
        return cls()

    @classmethod
    def hard(cls) -> Self:
        return cls(k_on=0.2, k_off=0.2, beta=10.0)


SPECIES: SpeciesNames = ("S", "M", "P")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0, 0),  # R0: S=0 → S=1
    (-1, 0, 0),  # R1: S=1 → S=0
    (0, +1, 0),  # R2: S=1 → S=1 + M
    (0, -1, 0),  # R3: M → ∅
    (0, 0, +1),  # R4: M → M + P
    (0, 0, -1),  # R5: P → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        S = state.x[0]
        M = state.x[1]
        P = state.x[2]
        return jnp.array(
            [
                params.k_on * (1.0 - S),  # R0
                params.k_off * S,  # R1
                params.beta * S,  # R2
                params.gamma_M * M,  # R3
                params.k_tl * M,  # R4
                params.delta_P * P,  # R5
            ]
        )

    return f
