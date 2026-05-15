"""Pre-built CRN models.

Each submodule exports ``SPECIES``, ``Params`` (with ``easy()`` / ``hard()``
factory classmethods), ``propensities_fn``, and ``apply_reaction``.
"""

from . import (
    birth_death,
    cca_optogenetic,
    incoherent_ffl,
    linear_cascade,
    negative_autoregulation,
    positive_autoregulation,
    repressilator,
    single_gene,
    toggle_switch,
)
from ._common import sample_trajectories  # noqa: F401 — re-exported via __all__

ALL_MODELS = (
    birth_death,
    single_gene,
    negative_autoregulation,
    positive_autoregulation,
    linear_cascade,
    toggle_switch,
    incoherent_ffl,
    repressilator,
    cca_optogenetic,
)

__all__ = sorted(
    [
        "ALL_MODELS",
        "sample_trajectories",
        *(m.__name__.rsplit(".", 1)[-1] for m in ALL_MODELS),
    ]
)
