"""Linear activation chain — A produced constitutively, B activated by A.

State: two species ``[A, B]``.

Reactions
---------
    R0:  ∅ → A     at rate  α_A                                    ν = (+1,  0)
    R1:  A → ∅     at rate  δ_A · A                                ν = (-1,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · A^n_A / (K_A^n_A + A^n_A)  ν = ( 0, +1)
    R3:  B → ∅     at rate  δ_B · B                                ν = ( 0, -1)
"""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function
from ._common import (
    State,
    make_apply_reaction,
)

SPECIES: tuple[str, ...] = ("A", "B")
_STOICH: tuple[tuple[int, ...], ...] = (
    (+1, 0),
    (-1, 0),
    (0, +1),
    (0, -1),
)


@dataclasses.dataclass(frozen=True)
class Params:
    alpha_A: float = 5.0
    delta_A: float = 1.0
    beta_B0: float = 0.05
    beta_B1: float = 5.0
    K_A: float = 1.0
    n_A: float = 1.0
    delta_B: float = 1.0

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(alpha_A=1.0, beta_B0=0.01, n_A=2.0)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        b_activation = hill_function(A, params.K_A, params.n_A)
        return jnp.array(
            [
                params.alpha_A,
                params.delta_A * A,
                params.beta_B0 + params.beta_B1 * b_activation,
                params.delta_B * B,
            ]
        )

    return f


apply_reaction = make_apply_reaction(_STOICH)
