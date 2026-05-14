"""Coherent feed-forward loop with AND logic.

State: three species ``[X, Y, Z]``. The model has *no* X-production
reaction — X enters as an initial condition and decays. Useful for
testing pulse-through dynamics and threshold detection.

Reactions
---------
    R0:  X → ∅     at rate  a · X                                    ν = (-1,  0,  0)
    R1:  ∅ → Y     at rate  a · 𝟙[X ≥ Ty]                            ν = ( 0, +1,  0)
    R2:  Y → ∅     at rate  a · Y                                    ν = ( 0, -1,  0)
    R3:  ∅ → Z     at rate  𝟙[X ≥ Ty] · 𝟙[Y ≥ Tz]                    ν = ( 0,  0, +1)
    R4:  Z → ∅     at rate  a · Z                                    ν = ( 0,  0, -1)

Sources
-------
* https://www.ebi.ac.uk/biomodels/BIOMD0000000316
* https://www.sciencedirect.com/science/article/pii/S0022283603012038
"""

import dataclasses
from typing import Self

import jax.numpy as jnp
from jax import Array

from ..types import PropensitiesFn, SpeciesNames, State, StoichiometryMatrix
from ._common import make_apply_reaction


@dataclasses.dataclass(frozen=True)
class Params:
    a: float = 1.0
    Ty: float = 0.5
    Tz: float = 0.5

    @classmethod
    def easy(cls) -> Self:
        return cls()

    @classmethod
    def hard(cls) -> Self:
        return cls(Ty=0.7, Tz=0.3)


SPECIES: SpeciesNames = ("X", "Y", "Z")
_STOICHIOMETRY: StoichiometryMatrix = (
    (-1, 0, 0),  # R0: X → ∅
    (0, +1, 0),  # R1: ∅ → Y
    (0, -1, 0),  # R2: Y → ∅
    (0, 0, +1),  # R3: ∅ → Z
    (0, 0, -1),  # R4: Z → ∅
)
apply_reaction = make_apply_reaction(_STOICHIOMETRY)


def propensities_fn(params: Params) -> PropensitiesFn:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        Y = state.x[1]
        Z = state.x[2]
        x_on = jnp.where(X >= params.Ty, 1.0, 0.0)
        y_on = jnp.where(Y >= params.Tz, 1.0, 0.0)
        return jnp.array(
            [
                params.a * X,  # R0
                params.a * x_on,  # R1
                params.a * Y,  # R2
                x_on * y_on,  # R3
                params.a * Z,  # R4
            ]
        )

    return f
