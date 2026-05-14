"""Pre-built CRN models.

Each submodule exports ``SPECIES``, ``Params`` (with ``easy()`` / ``hard()``
factory classmethods), ``propensities_fn``, and ``apply_reaction``.
"""

from . import (
    activator_repressor,
    birth_death,
    bistable,
    coherent_ffl,
    cyclic_ring,
    incoherent_ffl,
    linear_chain,
    mutual_activation,
    negative_autoreg,
    positive_autoreg,
    repressilator,
    single_gene,
    toggle,
)
from ._common import sample_trajectories  # noqa: F401 — re-exported via __all__

ALL_MODELS = (
    birth_death,
    single_gene,
    negative_autoreg,
    positive_autoreg,
    bistable,
    linear_chain,
    toggle,
    activator_repressor,
    mutual_activation,
    coherent_ffl,
    incoherent_ffl,
    repressilator,
    cyclic_ring,
)

__all__ = sorted(
    [
        "ALL_MODELS",
        "sample_trajectories",
        *(m.__name__.rsplit(".", 1)[-1] for m in ALL_MODELS),
    ]
)
