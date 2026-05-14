"""Incoherent feed-forward loop — A activates B; A activates C; B represses C.

State: three species ``[A, B, C]``. The opposing direct and indirect paths
from A to C generate a pulse / adaptive response on C.

Reactions
---------
    R0:  ∅ → A     at rate  α_A                                       ν = (+1,  0,  0)
    R1:  A → ∅     at rate  δ_A · A                                   ν = (-1,  0,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · A^n_A / (K_A^n_A + A^n_A)   ν = ( 0, +1,  0)
    R3:  B → ∅     at rate  δ_B · B                                   ν = ( 0, -1,  0)
    R4:  ∅ → C     at rate  β_C0 + β_C1 · A^n_A / (K_A^n_A + A^n_A)
                                  · K_B^n_B / (K_B^n_B + B^n_B)       ν = ( 0,  0, +1)
    R5:  C → ∅     at rate  δ_C · C                                   ν = ( 0,  0, -1)

Sources
-------
* https://www.weizmann.ac.il/mcb/alon/sites/mcb.UriAlon/files/network_motifs_nature_genetics_review.pdf
* https://www.omicsdi.org/dataset/biomodels/BIOMD0000000696
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..kinetics import hill_function, repressive_hill
from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    alpha_A: float = 5.0
    delta_A: float = 1.0
    beta_B0: float = 0.05
    beta_B1: float = 5.0
    K_A: float = 1.0
    n_A: float = 1.0
    delta_B: float = 1.0
    beta_C0: float = 0.05
    beta_C1: float = 5.0
    K_B: float = 1.0
    n_B: float = 1.0
    delta_C: float = 1.0

    @classmethod
    def easy(cls) -> Self:
        return cls()

    @classmethod
    def hard(cls) -> Self:
        return cls(alpha_A=1.0, beta_B0=0.01, n_A=2.0, beta_C0=0.01, n_B=2.0)


SPECIES: SpeciesNames = ("A", "B", "C")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0, 0),  # R0: ∅ → A
    (-1, 0, 0),  # R1: A → ∅
    (0, +1, 0),  # R2: ∅ → B
    (0, -1, 0),  # R3: B → ∅
    (0, 0, +1),  # R4: ∅ → C
    (0, 0, -1),  # R5: C → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        A = state.x[0]
        B = state.x[1]
        C = state.x[2]
        activate_b_by_a = hill_function(A, params.K_A, params.n_A)
        repress_c_by_b = repressive_hill(B, params.K_B, params.n_B)
        return jnp.array(
            [
                params.alpha_A,  # R0
                params.delta_A * A,  # R1
                params.beta_B0 + params.beta_B1 * activate_b_by_a,  # R2
                params.delta_B * B,  # R3
                params.beta_C0 + params.beta_C1 * activate_b_by_a * repress_c_by_b,  # R4
                params.delta_C * C,  # R5
            ]
        )

    return f
