"""Two-stage transcription-translation — mRNA + protein with hidden-state lag.

State: two species ``[M, P]`` (mRNA, protein).

Reactions
---------
    R0:  ∅       → M       at rate  α_M             ν = (+1,  0)
    R1:  M       → ∅       at rate  γ_M · M          ν = (-1,  0)
    R2:  M       → M + P   at rate  k_tl · M         ν = ( 0, +1)
    R3:  P       → ∅       at rate  δ_P · P          ν = ( 0, -1)
"""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ._common import (
    State,
    make_apply_reaction,
)

SPECIES: tuple[str, ...] = ("M", "P")
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1, 0),
    (-1, 0),
    (0, +1),
    (0, -1),
)


@dataclasses.dataclass(frozen=True)
class Params:
    alpha_M: float = 5.0
    gamma_M: float = 1.0
    k_tl: float = 10.0
    delta_P: float = 0.2

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(alpha_M=1.0, k_tl=5.0, delta_P=0.5)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        M = state.x[0]
        P = state.x[1]
        return jnp.array(
            [
                params.alpha_M,
                params.gamma_M * M,
                params.k_tl * M,
                params.delta_P * P,
            ]
        )

    return f


apply_reaction = make_apply_reaction(_STOICH)
