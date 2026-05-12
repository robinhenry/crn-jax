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
Default ``x0_dist`` covers both above- and below-threshold initial X.
"""

import dataclasses
import functools
from typing import Callable

import jax.numpy as jnp
from jax import Array

from ..types import PRNGKey
from ._common import (
    Dataset,
    State,
    make_apply_reaction,
    make_vmap_simulator,
    run_dataset,
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
        return cls(a=1.0, Ty=0.5, Tz=0.5)

    @classmethod
    def hard(cls) -> "Params":
        return cls(a=1.0, Ty=0.7, Tz=0.3)


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


@functools.lru_cache(maxsize=None)
def _build_simulator(n_steps: int, params: Params):
    return make_vmap_simulator(n_steps, propensities_fn(params), apply_reaction)


def simulate_dataset(
    key: PRNGKey,
    *,
    params: Params = Params(),
    n_replicates: int = 256,
    n_steps: int = 1000,
    dt: float = 0.05,
    x0_dist: tuple = ("uniform", 0.0, 2.0),
    y0_dist: tuple = ("zero",),
    z0_dist: tuple = ("zero",),
) -> Dataset:
    return run_dataset(
        key,
        species=SPECIES,
        simulator=_build_simulator(n_steps, params),
        n_replicates=n_replicates,
        n_steps=n_steps,
        dt=dt,
        x0_dists=(x0_dist, y0_dist, z0_dist),
    )
