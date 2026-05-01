# Changelog

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
