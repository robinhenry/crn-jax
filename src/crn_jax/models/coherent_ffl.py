"""Coherent feed-forward loop with AND logic (BIOMD0000000316).

State: three species ``[X, Y, Z]``. The model has *no* X-production
reaction — X enters as an initial condition and decays. Useful for
testing pulse-through dynamics and threshold detection.

Reactions
---------
    R0:  X → ∅     at rate  a · X                                   ν = (-1,  0,  0)
    R1:  ∅ → Y     at rate  a · 𝟙[X ≥ Ty]                            ν = ( 0, +1,  0)
    R2:  Y → ∅     at rate  a · Y                                   ν = ( 0, -1,  0)
    R3:  ∅ → Z     at rate  𝟙[X ≥ Ty] · 𝟙[Y ≥ Tz]                    ν = ( 0,  0, +1)
    R4:  Z → ∅     at rate  a · Z                                   ν = ( 0,  0, -1)

Thresholds use Heaviside step functions on the *internal* species X, Y.
Pick ``x0[:, 0]`` to span both above- and below-threshold values if you
want to see the pulse / no-pulse split across replicates.
"""

import dataclasses
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ._common import (
    State,
    make_apply_reaction,
)

SPECIES: tuple[str, ...] = ("X", "Y", "Z")
_STOICH: tuple[tuple[int, ...], ...] = (
    (-1, 0, 0),
    (0, +1, 0),
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
)


@dataclasses.dataclass(frozen=True)
class Params:
    a: float = 1.0
    Ty: float = 0.5
    Tz: float = 0.5

    @classmethod
    def easy(cls) -> "Params":
        return cls()

    @classmethod
    def hard(cls) -> "Params":
        return cls(Ty=0.7, Tz=0.3)


def propensities_fn(params: Params) -> Callable[[State, Array], Array]:
    def f(state: State, _u: Array) -> Array:
        X = state.x[0]
        Y = state.x[1]
        Z = state.x[2]
        x_on = jnp.where(X >= params.Ty, 1.0, 0.0)
        y_on = jnp.where(Y >= params.Tz, 1.0, 0.0)
        return jnp.array(
            [
                params.a * X,
                params.a * x_on,
                params.a * Y,
                x_on * y_on,
                params.a * Z,
            ]
        )

    return f


apply_reaction = make_apply_reaction(_STOICH)
