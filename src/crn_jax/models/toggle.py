"""Toggle switch — Gardner-Cantor-Collins mutual-inhibition circuit (2000).

State: two species ``[A, B]``; each represses the other.

Reactions
---------
    R0:  ∅ → A     at rate  β_A0 + β_A1 · K_B^n_B / (K_B^n_B + B^n_B)   ν = (+1,  0)
    R1:  A → ∅     at rate  δ_A · A                                     ν = (-1,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · K_A^n_A / (K_A^n_A + A^n_A)   ν = ( 0, +1)
    R3:  B → ∅     at rate  δ_B · B                                     ν = ( 0, -1)

The easy regime uses the published BIOMD0000000507 parameters
(``n_B = 2.5`` — non-integer Hill on purpose, matching the BioModels
source); the hard regime moves closer to the switching boundary.

Sources
-------
* https://www.nature.com/articles/35002131
* https://www.imagwiki.nibib.nih.gov/sites/default/files/jsim/models/biomodels/BIOMD0000000507.mod
* https://www.omicsdi.org/dataset/biomodels/BIOMD0000000507
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..kinetics import repressive_hill
from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    beta_A0: float = 15.6
    beta_A1: float = 156.25
    beta_B0: float = 0.0
    beta_B1: float = 15.6
    K_A: float = 2.0015
    K_B: float = 2.9618e-05
    n_A: float = 1.0
    n_B: float = 2.5
    delta_A: float = 1.0
    delta_B: float = 1.0

    @classmethod
    def easy(cls) -> Self:
        return cls()

    @classmethod
    def hard(cls) -> Self:
        return cls(
            beta_A0=10.0,
            beta_A1=100.0,
            beta_B1=10.0,
            K_A=1.5,
            K_B=0.01,
            n_A=2.0,
            n_B=2.0,
        )


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
        repress_a_by_b = repressive_hill(B, params.K_B, params.n_B)
        repress_b_by_a = repressive_hill(A, params.K_A, params.n_A)
        return jnp.array(
            [
                params.beta_A0 + params.beta_A1 * repress_a_by_b,  # R0
                params.delta_A * A,  # R1
                params.beta_B0 + params.beta_B1 * repress_b_by_a,  # R2
                params.delta_B * B,  # R3
            ]
        )

    return f
