"""Repressilator — Elowitz-Leibler synthetic oscillator.

State: three species ``[A, B, C]`` forming a single repressive ring:
C ⊣ A ⊣ B ⊣ C. With sufficient cooperativity and a sharp enough Hill
curve, the system limit-cycles.

Reactions
---------
    R0:  ∅ → A     at rate  β_A0 + β_A1 · K_Cⁿ / (K_Cⁿ + Cⁿ)     ν = (+1,  0,  0)
    R1:  A → ∅     at rate  δ_A · A                                ν = (-1,  0,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · K_Aⁿ / (K_Aⁿ + Aⁿ)     ν = ( 0, +1,  0)
    R3:  B → ∅     at rate  δ_B · B                                ν = ( 0, -1,  0)
    R4:  ∅ → C     at rate  β_C0 + β_C1 · K_Bⁿ / (K_Bⁿ + Bⁿ)     ν = ( 0,  0, +1)
    R5:  C → ∅     at rate  δ_C · C                                ν = ( 0,  0, -1)

A single shared Hill exponent ``n`` is used for all three nodes (matches
the BioModels source). Defaults give δ ≈ 0.347 (lifetime ≈ 3 min) and
β₁ chosen so the deterministic system limit-cycles.

Sources
-------
* https://www.ebi.ac.uk/biomodels/BIOMD0000000012
* https://github.com/biomodels/BIOMD0000000012/blob/master/README.md
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
    beta_A0: float = 0.03
    beta_A1: float = 29.97
    beta_B0: float = 0.03
    beta_B1: float = 29.97
    beta_C0: float = 0.03
    beta_C1: float = 29.97
    K_A: float = 40.0
    K_B: float = 40.0
    K_C: float = 40.0
    n: float = 2.0
    delta_A: float = 0.347
    delta_B: float = 0.347
    delta_C: float = 0.347

    @classmethod
    def default(cls) -> Self:
        return cls()


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
        repress_a_by_c = repressive_hill(C, params.K_C, params.n)
        repress_b_by_a = repressive_hill(A, params.K_A, params.n)
        repress_c_by_b = repressive_hill(B, params.K_B, params.n)
        return jnp.array(
            [
                params.beta_A0 + params.beta_A1 * repress_a_by_c,  # R0
                params.delta_A * A,  # R1
                params.beta_B0 + params.beta_B1 * repress_b_by_a,  # R2
                params.delta_B * B,  # R3
                params.beta_C0 + params.beta_C1 * repress_c_by_b,  # R4
                params.delta_C * C,  # R5
            ]
        )

    return f
