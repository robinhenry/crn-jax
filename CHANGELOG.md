# Changelog

## v0.3.0 (unreleased)

**Breaking:** `crn_jax.motifs` is renamed to `crn_jax.models` and its contents
are replaced with the 14-model benchmark library described in
`src/crn_jax/models/library.json` (birth-death, two-stage transcription-
translation, telegraph promoter, ±autoreg, bistable, linear chain, toggle,
activator-repressor, mutual activation, coherent FFL, incoherent FFL,
repressilator, cyclic hybrid ring). The pre-v0.3 models (`inducible`,
`autoreg`, `cascade`, `ffl_and`) are removed.

- Every model exposes both parameter regimes via `Params.easy()` /
  `Params.hard()` classmethods, drawn from the packaged `library.json`.
- `Dataset` is now a single shared NamedTuple with stacked
  `xs: (n_replicates, n_steps, n_species)` and a `species: tuple[str, ...]`
  field, replacing the per-species `Xs / Ys / Zs / dX / dY / dZ` fields.
- `simulate_dataset` is now a package-level function:
  `crn_jax.models.simulate_dataset(model, key, x0, ...)`. The caller always
  supplies `x0: (n_replicates, n_species)`; the library no longer samples
  initial conditions (the previous per-species `*_dist` kwargs and the
  internal `DistSpec` are gone). `x0` is validated for shape and
  non-negativity. `n_replicates` is inferred from `x0.shape[0]`.
- Each model module is now just the math (`SPECIES`, `Params`,
  `propensities_fn`, `apply_reaction`); the per-model `simulate_dataset`
  and `_build_simulator` wrappers are removed.
- `simulate_dataset` no longer accepts a `u_dist`: every model in the new
  library is autonomous.
- Example `examples/03_grn_motifs.py` renamed to `examples/03_grn_models.py`
  and rewritten around the repressilator and toggle switch.

## v0.2.0

- New `crn_jax.motifs` subpackage: pre-built canonical GRN motifs
  (`inducible`, `autoreg`, `cascade`, `ffl_and`). Each module exports a uniform surface (`State`, `Params`, `propensities_fn`, `apply_reaction`, `simulate_dataset`) so swapping systems in benchmarks is a one-line change. The bare propensity / reaction primitives also drop into `simulate_trajectory` directly.
- New example `examples/03_grn_motifs.py`.
- 14 new motif tests in `tests/test_motifs.py`.

## v0.1.2

- Republish release on a fresh commit so `poetry-dynamic-versioning` picks the
  correct tag (the earlier `v0.1.1` tag shared a commit with `v0.1`, causing
  the build to resolve to `0.1` and collide with PyPI).

## v0.1

- Initial PyPI release.
