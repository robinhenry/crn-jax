"""Incoherent feed-forward loop (I1-FFL): A activates B, A activates C, B represses C.

State: three species ``[A, B, C]``.
A plays the role of the upstream input X in Mangan & Alon (2003):
the AND-like combination of direct activation (A → C) and indirect
repression (A → B ⊣ C) produces a transient pulse on C.
C rises while B is still low, then falls as B catches up and represses it.

Reactions
---------
    R0:  ∅ → A     at rate  α_A                                       ν = (+1,  0,  0)
    R1:  A → ∅     at rate  δ_A · A                                   ν = (-1,  0,  0)
    R2:  ∅ → B     at rate  β_B0 + β_B1 · A^n_A / (K_A^n_A + A^n_A)   ν = ( 0, +1,  0)
    R3:  B → ∅     at rate  δ_B · B                                   ν = ( 0, -1,  0)
    R4:  ∅ → C     at rate  β_C0 + β_C1 · A^n_A / (K_A^n_A + A^n_A)
                                  · K_B^n_B / (K_B^n_B + B^n_B)       ν = ( 0,  0, +1)
    R5:  C → ∅     at rate  δ_C · C                                   ν = ( 0,  0, -1)

Default parameters
------------------
Mangan & Alon (2003) Methods dimensionless values (H = 2, β = α = 1,
B_y = B_z = 0, K_xy / X_high ≈ 0.2, K_yz / Y_ss ≈ 0.1), rescaled to
min⁻¹ with a 30-min protein half-life (fast-growth E. coli) and scaled
to copy numbers ~50 so the stochastic pulse on C is visible above
Poisson noise (Mangan-Alon's dimensionless units would put the pulse at
~1 molecule).

All dimensionless ratios are preserved exactly:
    α_A   = 50 · ln(2)/30 ≈ 1.1552   ⇒ ⟨A⟩ = 50
    δ_A   = ln(2)/30      ≈ 0.02310
    β_B1  = 50 · ln(2)/30 ≈ 1.1552   ⇒ ⟨B⟩ ≈ 48 when A is saturated
    K_A   = 10                        (K_A / ⟨A⟩ = 0.2, Mangan-Alon ratio)
    n_A   = 2                         (Mangan-Alon H = 2; needed for pulse)
    δ_B   = ln(2)/30      ≈ 0.02310
    β_C1  = 50 · ln(2)/30 ≈ 1.1552
    K_B   = 5                         (K_B / ⟨B⟩ ≈ 0.1, Mangan-Alon ratio)
    n_B   = 2                         (Mangan-Alon H = 2)
    δ_C   = ln(2)/30      ≈ 0.02310
    β_B0  = β_C0 = 0                  (Mangan-Alon B_y = B_z = 0)

Steady state: ⟨A⟩ = 50, ⟨B⟩ ≈ 48, ⟨C⟩_ss ≈ 0.5 (strongly suppressed by B).

Sources
-------
* Mangan S, Alon U (2003). Structure and function of the feed-forward
  loop network motif. PNAS 100:11980–11985 — canonical FFL motif paper;
  default parameters above are taken from Methods (H = 2, B_y = B_z = 0,
  K_yz / K_xy ratio ≈ 1/2) and rescaled to min⁻¹.
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
    alpha_A: float = 1.1552
    delta_A: float = 0.02310
    beta_B0: float = 0.0
    beta_B1: float = 1.1552
    K_A: float = 10.0
    n_A: float = 2.0
    delta_B: float = 0.02310
    beta_C0: float = 0.0
    beta_C1: float = 1.1552
    K_B: float = 5.0
    n_B: float = 2.0
    delta_C: float = 0.02310

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
