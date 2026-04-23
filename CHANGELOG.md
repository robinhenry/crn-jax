# Changelog

## v0.1.2

- Republish release on a fresh commit so `poetry-dynamic-versioning` picks the
  correct tag (the earlier `v0.1.1` tag shared a commit with `v0.1`, causing
  the build to resolve to `0.1` and collide with PyPI).

## v0.1

- Initial PyPI release.
