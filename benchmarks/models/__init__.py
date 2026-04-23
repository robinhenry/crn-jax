"""Benchmark model encodings for crn-jax, GillesPy2, and jax-smfsb.

Each module exposes ``PARAMS`` (the canonical parameter set) plus three
``run_<lib>(seed_or_key, n_trajectories, *, return_full_trajectory=False)``
functions returning numpy arrays of equivalent shape so that the benchmark
scripts can treat all libraries uniformly.
"""

import importlib

MODEL_NAMES = ("birth_death", "lotka_volterra", "linear_cascade")


def get(name: str):
    return importlib.import_module(f"benchmarks.models.{name}")
