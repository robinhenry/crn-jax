"""CcaS/CcaR optogenetic gene-expression circuit, an input-driven E. coli system.

The model's propensities depend on a *time-varying
control input* ``u(t) ∈ [0, 1]`` (the fraction of green vs red light hitting
the cell). Useful for demonstrating active steering, optimal control,
active-learning, etc.

The CcaS/CcaR system (Tabor lab, Olson et al. 2014) is a two-component
light sensor adapted from Synechocystis into E. coli. CcaS is a green-light
absorbing histidine kinase: in green light it autophosphorylates and
transfers phosphate to CcaR, and in red light CcaS becomes a phosphatase and
dephosphorylates CcaR. Phosphorylated CcaR (CcaR~P) is a transcription
factor that drives expression from the pcpcG2 promoter.
So green light → R high → P high, red light → R decays → P falls.

State: two species ``[R, P]`` where
    R = phosphorylated CcaR (active TF, light-gated)
    P = reporter protein driven by R via the cpcG2 promoter

Reactions
---------
    Rxn 0:  ∅ → R     at rate  k_R · u                              ν = (+1,  0)
    Rxn 1:  R → ∅     at rate  γ_R · R                              ν = (-1,  0)
    Rxn 2:  ∅ → P     at rate  β_P0 + β_P1 · R^n / (K^n + R^n)      ν = ( 0, +1)
    Rxn 3:  P → ∅     at rate  γ_P · P                              ν = ( 0, -1)

The input ``u`` is the green-light fraction (0 = pure red → R production
off, 1 = pure green → R produced at rate k_R). CcaR turnover is set by
the basal phosphatase activity ``γ_R``. We lump red-light-driven
dephosphorylation into the same constant for simplicity (in the full
system, ``γ_R`` would itself depend on (1-u), but the constant-rate
approximation is the standard reduction used in pulse-frequency control
demonstrations).

Default parameters
------------------
Two parameters fall out directly from published measurements; the rest
are within standard E. coli ranges. All rates in min⁻¹.

    k_R   = 4.62                   (chosen so R_ss(u=1) ≈ 100 monomers —
                                    typical CcaR copy number in Tabor lab
                                    constructs)
    γ_R   = ln(2)/15 ≈ 0.0462      (Olson 2014: ~15-min half-time for
                                    CcaR phosphorylation response to a light
                                    step)
    β_P0  = 0.167                  (calibrated so Pmax/Pmin ≈ 25 — Schmidl
                                    2014's reported fold change for the
                                    *optimized* CcaS/CcaR variant in E. coli)
    β_P1  = 5.0                    (max P production from saturated cpcG2;
                                    realistic strong-promoter range giving
                                    Pmax ≈ 180 monomers per cell)
    K     = 50                     (= R_ss(u=1)/2; standard operating-point
                                    design — Hill at half-saturation when
                                    light is at saturating green)
    n     = 2                      (Hill cooperativity for CcaR~P dimer
                                    binding to the cpcG2 operator)
    γ_P   = ln(2)/30 ≈ 0.02310     (reporter half-life 30 min; fast-growth
                                    E. coli — library convention)

Steady-state map u → P
----------------------
    ⟨R⟩_ss(u) = k_R · u / γ_R                ≈ 100 · u
    Hill(R; K=50, n=2) at R = 100u           = (2u)² / ((2u)² + 1)
    ⟨P⟩_ss(u)                                = (β_P0 + β_P1 · Hill) / γ_P

At u=0: ⟨R⟩ = 0, Hill = 0, ⟨P⟩ ≈ 7.2 (basal floor).
At u=1: ⟨R⟩ ≈ 100, Hill = 0.8, ⟨P⟩ ≈ 180.
Pmax / Pmin ≈ 25 — Schmidl 2014's reported fold change for the optimized
CcaS/CcaR system. The much weaker original Olson 2014 system has ~7×
fold change; we use the Schmidl-style optimized parameters because they
give a more useful dynamic range for control demonstrations.

Sources
-------
* Olson EJ, Hartsough LA, Landry BP, Shroff R, Tabor JJ (2014).
  Characterizing bacterial gene circuit dynamics with optically programmed
  gene expression signals. Nature Methods 11:449–455 — the 15-min CcaR
  response time used above is from their dynamic characterization.
* Schmidl SR, Sheth RU, Wu A, Tabor JJ (2014). Refactoring and
  optimization of light-switchable Escherichia coli two-component
  systems. ACS Synthetic Biology 3:820–831 — the ~25× green/red fold
  change used above is from their optimized CcaS/CcaR variant.
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
    k_R: float = 4.62
    gamma_R: float = 0.0462
    beta_P0: float = 0.167
    beta_P1: float = 5.0
    K: float = 50.0
    n: float = 2.0
    gamma_P: float = 0.02310

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("R", "P")
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1, 0),  # Rxn 0: ∅ → R
    (-1, 0),  # Rxn 1: R → ∅
    (0, +1),  # Rxn 2: ∅ → P
    (0, -1),  # Rxn 3: P → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, u: Array) -> Array:
        R = state.x[0]
        P = state.x[1]
        activation = hill_function(R, params.K, params.n)
        return jnp.array(
            [
                params.k_R * u,  # Rxn 0
                params.gamma_R * R,  # Rxn 1
                params.beta_P0 + params.beta_P1 * activation,  # Rxn 2
                params.gamma_P * P,  # Rxn 3
            ]
        )

    return f
