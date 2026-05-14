"""Linear activation chain — A produced constitutively, B activated by A.

State: two species ``[A, B]``.

Reactions
---------
    R0:  ∅ → A     at rate  α_A                                      ν = (+1,  0)
    R1:  A → ∅     at rate  δ_A · A                                  ν = (-1,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · A^n_A / (K_A^n_A + A^n_A)  ν = ( 0, +1)
    R3:  B → ∅     at rate  δ_B · B                                  ν = ( 0, -1)

Default parameters
------------------
Alon-style dimensionless analytical case (α=5, β₀=0.05, β₁=5, n=1, δ=1)
applied to both cascade stages, rescaled into min⁻¹ via δ = ln(2)/60
(matching ``single_gene``'s 60-min protein half-life — same convention as
the other models in this library).

The Hill threshold K_A is set to ⟨A⟩ = 5 (not the textbook's K=1) so that
B sits at the responsive half-max of the activation curve and actually
tracks A's fluctuations. With K_A = 1 (Alon's analytical default),
Hill(⟨A⟩) ≈ 0.83 is saturated and B sees an almost-constant input.

    α_A   = 5    · ln(2)/60 ≈ 0.0578   ⇒ ⟨A⟩ = α_A/δ_A = 5
    δ_A   = ln(2)/60        ≈ 0.01155   (half-life 60 min)
    β_B0  = 0.05 · ln(2)/60 ≈ 5.776e-4  (small leak)
    β_B1  = 5    · ln(2)/60 ≈ 0.0578
    K_A   = 5                            (= ⟨A⟩: B at Hill half-max)
    n_A   = 1                            (Alon's analytical case)
    δ_B   = ln(2)/60        ≈ 0.01155   (half-life 60 min)

Steady state ⟨B⟩ = β_B0/δ_B + (β_B1/δ_B) · Hill(⟨A⟩; K_A, n_A)
                 = 0.05 + 5 · 0.5 = 2.55.

Sources
-------
* Alon U (2007). An Introduction to Systems Biology, Ch. 2 (simple
  regulation) and Ch. 5 (long transcriptional cascades) — parameters
  above are the textbook analytical-case dimensionless values, with K_A
  tuned to ⟨A⟩ for cascade pedagogy.
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
    alpha_A: float = 0.05776
    delta_A: float = 0.01155
    beta_B0: float = 0.0005776
    beta_B1: float = 0.05776
    K_A: float = 5.0
    n_A: float = 1.0
    delta_B: float = 0.01155

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
        b_activation = hill_function(A, params.K_A, params.n_A)
        return jnp.array(
            [
                params.alpha_A,  # R0
                params.delta_A * A,  # R1
                params.beta_B0 + params.beta_B1 * b_activation,  # R2
                params.delta_B * B,  # R3
            ]
        )

    return f
