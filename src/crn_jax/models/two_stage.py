"""Two-stage transcription-translation — mRNA + protein with hidden-state lag.

State: two species ``[M, P]`` (mRNA, protein).

Reactions
---------
    R0:  ∅ → M       at rate  α_M       ν = (+1,  0)
    R1:  M → ∅       at rate  γ_M · M   ν = (-1,  0)
    R2:  M → M + P   at rate  k_tl · M  ν = ( 0, +1)
    R3:  P → ∅       at rate  δ_P · P   ν = ( 0, -1)

Sources
-------
* https://link.springer.com/chapter/10.1007/978-3-030-85633-5_13
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    alpha_M: float = 5.0
    gamma_M: float = 1.0
    k_tl: float = 10.0
    delta_P: float = 0.2

    @classmethod
    def easy(cls) -> Self:
        return cls()

    @classmethod
    def hard(cls) -> Self:
        return cls(alpha_M=1.0, k_tl=5.0, delta_P=0.5)


SPECIES: SpeciesNames = ("M", "P")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0),  # R0: ∅ → M
    (-1, 0),  # R1: M → ∅
    (0, +1),  # R2: M → M + P
    (0, -1),  # R3: P → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        M = state.x[0]
        P = state.x[1]
        return jnp.array(
            [
                params.alpha_M,  # R0
                params.gamma_M * M,  # R1
                params.k_tl * M,  # R2
                params.delta_P * P,  # R3
            ]
        )

    return f
