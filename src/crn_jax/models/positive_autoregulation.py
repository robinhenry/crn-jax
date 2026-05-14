"""Positive autoregulation — one-node Hill self-activation.

State: one species ``X``.

Reactions
---------
    R0:  ∅ → X     at rate  β₀ + β₁ · Xⁿ / (Kⁿ + Xⁿ)   ν = (+1,)
    R1:  X → ∅     at rate  δ · X                      ν = (-1,)

For sub-cooperative Hill (``n ≤ 1``) the equilibrium is monostable and
graded; cooperativity ``n ≥ 2`` plus low leakage gives switch-like
behaviour — see :mod:`~crn_jax.models.bistable` for the bistable regime.

Default parameters
------------------
Adapted from Alon, Fig 3.5 (n=1 analytical case), rescaled into min⁻¹ via
δ = ln(2)/60 (matching ``single_gene``'s 60-min protein half-life). Same
time-unit calibration as ``negative_autoregulation``, but the leak β₀ is
deliberately set higher than the textbook β₀ = 0 to keep the stochastic
model from getting trapped at the absorbing state X = 0 (see note below).
    β₀ = 0.5 · ln(2)/60 ≈ 5.776e-3    (escape leak; mean wait at X=0 ≈ 173 min)
    β₁ = 5   · ln(2)/60 ≈ 0.0578
    K  = 1                            (count)
    n  = 1                            (Alon's analytical case)
    δ  = ln(2)/60        ≈ 0.01155    (protein half-life 60 min)

Deterministic steady state ⟨X⟩ ≈ 4.6 (vs Alon's β₀=0 case of 4.0 —
a ~15% upward shift caused by the larger leak).
Response time T₁/₂ ~ 2/δ ≈ 173 min — slower than the simple-decay 60 min
(Alon's Fig 3.5 qualitative result).

Note on the leak: at low copy numbers (⟨X⟩ ≈ 4), the discrete stochastic
model has a real absorbing-state problem near X = 0 that Alon's
deterministic ODE doesn't capture. A tiny β₀ (e.g. β₀ = 0.01 in Alon
units) preserves the steady state at 4 but leaves a mean wait at X=0 of
~8700 min — longer than typical simulations. β₀ = 0.5 (Alon units) gives
a mean wait of ~173 min (≈ 2 decay times) so trajectories that fluctuate
to 0 can recover.

Sources
-------
* Alon U (2007). An Introduction to Systems Biology, Ch. 3.5 (Fig 3.5) —
  dimensionless parameters above; time-unit conversion as described.
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
    beta_0: float = 0.005776
    beta_1: float = 0.05776
    K: float = 1.0
    n: float = 1.0
    delta: float = 0.01155

    @classmethod
    def default(cls) -> Self:
        return cls()


SPECIES: SpeciesNames = ("X",)
_STOICHIOMETRY: StoichiometryMatrix = (
    (+1,),  # R0: ∅ → X
    (-1,),  # R1: X → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        activation = hill_function(X, params.K, params.n)
        return jnp.array(
            [
                params.beta_0 + params.beta_1 * activation,  # R0
                params.delta * X,  # R1
            ]
        )

    return f
