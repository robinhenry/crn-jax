"""Single gene: constitutive transcription and translation, no regulation.

State: two species ``[R, P]`` (mRNA, protein). Notation follows Thattai &
van Oudenaarden (2001).

Reactions
---------
    Rxn 0:  ∅ → R       at rate  k_R         ν = (+1,  0)
    Rxn 1:  R → ∅       at rate  γ_R · R     ν = (-1,  0)
    Rxn 2:  R → R + P   at rate  k_P · R     ν = ( 0, +1)
    Rxn 3:  P → ∅       at rate  γ_P · P     ν = ( 0, -1)

Default parameters
------------------
E. coli base case from Thattai & van Oudenaarden (2001), in min⁻¹.
Thattai's protein half-life was 1 h (slow-growth E. coli); we use 30 min
to match fast-growth conditions (typical doubling time).
Thattai's burst size b = k_P/γ_R = 20 is preserved exactly.

    k_R  = 0.6                (transcription initiation, 0.01 s⁻¹)
    γ_R  = ln(2)/2 ≈ 0.347    (mRNA half-life 2 min — fast-growth E. coli)
    k_P  = 20 · γ_R ≈ 6.93    (burst size b = k_P/γ_R = 20)
    γ_P  = ln(2)/30 ≈ 0.0231  (protein half-life 30 min — fast-growth E. coli)

Steady state: ⟨R⟩ ≈ 1.7, ⟨P⟩ ≈ 520 (Thattai's 1030 halved by the faster
protein decay, but still biologically realistic for an E. coli protein).

Sources
-------
* Thattai M, van Oudenaarden A (2001). Intrinsic noise in gene regulatory
  networks. PNAS 98(15):8614–8619. https://doi.org/10.1073/pnas.151588598
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    k_R: float = 0.6
    gamma_R: float = 0.3466
    k_P: float = 6.931
    gamma_P: float = 0.02310

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("R", "P")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0),  # Rxn 0: ∅ → R
    (-1, 0),  # Rxn 1: R → ∅
    (0, +1),  # Rxn 2: R → R + P
    (0, -1),  # Rxn 3: P → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        R = state.x[0]
        P = state.x[1]
        return jnp.array(
            [
                params.k_R,  # Rxn 0
                params.gamma_R * R,  # Rxn 1
                params.k_P * R,  # Rxn 2
                params.gamma_P * P,  # Rxn 3
            ]
        )

    return f
