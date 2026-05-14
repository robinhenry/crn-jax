"""Mutual activation — A and B activate each other.

State: two species ``[A, B]``.

Reactions
---------
    R0:  ∅ → A     at rate  β_A0 + β_A1 · B^n_B / (K_B^n_B + B^n_B)     ν = (+1,  0)
    R1:  A → ∅     at rate  δ_A · A                                     ν = (-1,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · A^n_A / (K_A^n_A + A^n_A)     ν = ( 0, +1)
    R3:  B → ∅     at rate  δ_B · B                                     ν = ( 0, -1)

Positive mutual feedback that can amplify small fluctuations into
multistability under sharp Hill cooperativity.

Sources
-------
* https://www.weizmann.ac.il/mcb/alon/sites/mcb.UriAlon/files/network_motifs_nature_genetics_review.pdf
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function
from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    beta_A0: float = 0.05
    beta_A1: float = 5.0
    K_B: float = 1.0
    n_B: float = 1.0
    delta_A: float = 1.0
    beta_B0: float = 0.05
    beta_B1: float = 5.0
    K_A: float = 1.0
    n_A: float = 1.0
    delta_B: float = 1.0

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("A", "B")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0),  # R0: ∅ → A
    (-1, 0),  # R1: A → ∅
    (0, +1),  # R2: ∅ → B
    (0, -1),  # R3: B → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        activate_a_by_b = hill_function(B, params.K_B, params.n_B)
        activate_b_by_a = hill_function(A, params.K_A, params.n_A)
        return jnp.array(
            [
                params.beta_A0 + params.beta_A1 * activate_a_by_b,  # R0
                params.delta_A * A,  # R1
                params.beta_B0 + params.beta_B1 * activate_b_by_a,  # R2
                params.delta_B * B,  # R3
            ]
        )

    return f
