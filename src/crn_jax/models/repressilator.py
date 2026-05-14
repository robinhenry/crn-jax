"""Repressilator: Elowitz-Leibler synthetic oscillator.

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

A single shared Hill exponent ``n`` is used for all three nodes.

Default parameters
------------------
Adiabatic reduction of the Elowitz & Leibler (2000) 6-species
(mRNA + protein) model with fitted parameters from their E. coli synthetic
repressilator construct, rescaled into a 3-species Hill form. mRNA
dynamics are ignored in this implementation.

    β_A1 = β_B1 = β_C1 = α · K · δ  ≈  600  (max protein production /min)
    β_A0 = β_B0 = β_C0 = 0.1% · β_1 ≈  0.6  (basal leak, Elowitz's α₀/α = 10⁻³)
    K_A  = K_B  = K_C  = 40                 (Elowitz K_M = 40 monomers/cell)
    n                  = 3                  (Hill cooperativity; see note)
    δ_A  = δ_B  = δ_C  = ln(2)/10 ≈ 0.0693  (10-min protein half-life — LVA tag)

Dimensionless `α = β₁ / (δ · K) ≈ 216` — well above the Hopf
bifurcation threshold for n=3 (α_crit ≈ 3.78), giving robust limit
cycles. Predicted oscillation period of order ~150 min (matches
Elowitz's experimental observation in E. coli).

Note on Hill coefficient
-------------------------
Elowitz's experimental fit gives n ≈ 2 (single TF dimer binding).
However, the 3-species Hill-form reduced model with n = 2 does *not*
oscillate at any α, the Hopf bifurcation requires n > 2 (Goodwin-style
analysis for the symmetric 3-cycle). The original 6-species Elowitz
model does oscillate at n = 2 because the mRNA-protein two-stage
dynamics add an extra time delay that lowers the bifurcation threshold,
but that delay is lost in the adiabatic reduction. Bumping to n = 3 is
fixes it and remains biologically defensible: real TF binding to
promoters with multiple cooperative operator sites (e.g., λ-cI binding
OR1+OR2) gives an effective Hill exponent ≈ 2.5–3.

Note on timescale and copy numbers
----------------------------------
The 10-min protein half-life is deliberately *faster* than this
library's default 30-min convention. Elowitz used LVA degradation tags
specifically to accelerate protein turnover. Without this, dilution-
limited proteins (~30-min half-life) would have cycles too slow for
clean oscillation. The 10-min value is consistent with the
specific synthetic construct.

⟨A⟩_max ≈ β_A1 / δ_A ≈ 8650 monomers is high relative to typical TF
copy numbers, but representative of the strong pL promoter expression
Elowitz observed in this actual E. coli circuit.

Sources
-------
* Elowitz MB, Leibler S (2000). A synthetic oscillatory network of
  transcriptional regulators. Nature 403:335–338 — original landmark;
  parameters above are taken from their Methods (adiabatically reduced),
  with n bumped from 2 to 3 to preserve oscillation in the reduced model.
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
    beta_A0: float = 0.6
    beta_A1: float = 600.0
    beta_B0: float = 0.6
    beta_B1: float = 600.0
    beta_C0: float = 0.6
    beta_C1: float = 600.0
    K_A: float = 40.0
    K_B: float = 40.0
    K_C: float = 40.0
    n: float = 3.0
    delta_A: float = 0.0693
    delta_B: float = 0.0693
    delta_C: float = 0.0693

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
