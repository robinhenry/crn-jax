"""Linear activation chain: A produced constitutively, B activated by A.

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
applied to both cascade stages, rescaled into min⁻¹ with a 30-min
protein half-life (fast-growth E. coli convention) and scaled up 10× in
copy-number space so that ⟨A⟩ = 50 sits in the realistic E. coli TF
range.

The Hill threshold K_A is set to ⟨A⟩ = 50 (matched to the operating
point of A) so that B sits at the responsive half-max of the activation
curve and tracks A's fluctuations. K_A = 1 (Alon's analytical default)
would saturate the cascade and make B see an almost-constant input.

    α_A   = 50   · ln(2)/30 ≈ 1.1552   ⇒ ⟨A⟩ = α_A/δ_A = 50
    δ_A   = ln(2)/30        ≈ 0.02310  (half-life 30 min)
    β_B0  = 0.5  · ln(2)/30 ≈ 1.155e-2 (small leak)
    β_B1  = 50   · ln(2)/30 ≈ 1.1552
    K_A   = 50                          (= ⟨A⟩: B at Hill half-max)
    n_A   = 2                           (TF dimer cooperativity)
    δ_B   = ln(2)/30        ≈ 0.02310  (half-life 30 min)

Steady state ⟨B⟩ = β_B0/δ_B + (β_B1/δ_B) · Hill(⟨A⟩; K_A, n_A)
                 = 0.5 + 50 · 0.5 = 25.5.
With K_A = ⟨A⟩ the Hill value is 0.5 regardless of n, so ⟨B⟩ is the
same as in the n=1 analytical case — but n=2 makes B's response to A's
fluctuations sharper (more cascade signal-like).

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
    alpha_A: float = 1.1552
    delta_A: float = 0.02310
    beta_B0: float = 0.01155
    beta_B1: float = 1.1552
    K_A: float = 50.0
    n_A: float = 2.0
    delta_B: float = 0.02310

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
