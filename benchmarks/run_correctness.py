"""Compare crn-jax against GillesPy2 on each model.

For each model, simulate N trajectories with each library at the same time
horizon, then test whether crn-jax samples from the same distribution as
GillesPy2 (the reference implementation).

Statistical tests
-----------------
- 2-sample Kolmogorov-Smirnov per species — gate of distribution equality.
  Critical value at level α=0.001 is c(α)·sqrt((n+m)/(nm)) ≈ 1.95·sqrt(2/N).
  At N=10 000 → KS_crit ≈ 0.028.
- Wasserstein-1 distance per species — informational; sensitive to tails.
- |Δmean| / max(|mean|) — interpretable physical sanity check; gate at 5%.

PASS requires KS test passes for every species AND mean within tolerance for
every species.

Saves the raw final-state arrays to results/correctness_<model>.npz so plot
generation can produce histograms + ECDFs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.models import MODEL_NAMES, get  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 2-sample KS critical value at α=0.001 (~3.3σ): c(α) ≈ 1.95.
KS_ALPHA_C = 1.95
# Mean tolerance: 5% of the larger absolute mean (interpretable, scale-free).
MEAN_REL_TOL = 0.05
# Floor on mean tolerance for near-zero means (avoid pathological gating).
MEAN_ABS_TOL = 0.5


def _final_state(run_fn, key_or_seed, n_traj: int) -> np.ndarray:
    """(n_traj, n_species) ndarray of final states; collapses 1D outputs."""
    out = run_fn(key_or_seed, n_traj, return_full_trajectory=False)
    if out.ndim == 1:
        out = out[:, None]
    return out


def _ks_2samp(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic D = sup |F_a(x) - F_b(x)|."""
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-Wasserstein distance between two equal-length empirical samples."""
    return float(np.mean(np.abs(np.sort(a) - np.sort(b))))


def _ks_critical(n: int, m: int, c: float = KS_ALPHA_C) -> float:
    return c * math.sqrt((n + m) / (n * m))


def _compare(model_name: str, n_trajectories: int, seed: int) -> dict:
    model = get(model_name)
    key = jax.random.PRNGKey(seed)

    print(f"\n=== {model_name} (N={n_trajectories}) ===")
    finals = {}
    for lib, run_fn, arg in [
        ("crn_jax", model.run_crn_jax, key),
        ("gillespy2", model.run_gillespy2, seed),
    ]:
        print(f"  running {lib}…", flush=True)
        finals[lib] = _final_state(run_fn, arg, n_trajectories)
        a = finals[lib]
        print(f"    shape={a.shape} mean={a.mean(0).round(2)} std={a.std(0).round(2)}")

    ref = "gillespy2"
    test = "crn_jax"
    n_species = finals[ref].shape[1]
    ks_crit = _ks_critical(len(finals[ref]), len(finals[test]))

    print(f"\n  Reference: {ref}  Tested: {test}  KS_crit (α=0.001) = {ks_crit:.4f}")

    rows = []
    for i in range(n_species):
        x_ref = finals[ref][:, i]
        x_tst = finals[test][:, i]
        mean_ref = x_ref.mean()
        mean_tst = x_tst.mean()
        scale = max(abs(mean_ref), abs(mean_tst), MEAN_ABS_TOL / MEAN_REL_TOL)
        dmean = abs(mean_tst - mean_ref)
        dmean_rel = dmean / scale
        # Pooled MC standard error of the mean — needed because heavy-tailed
        # distributions (LV) have huge MC variance on the mean even when
        # KS confirms the distributions match.
        n = len(x_ref)
        se_pooled = math.sqrt(x_ref.var() / n + x_tst.var() / n)
        mean_ok_i = (dmean_rel < MEAN_REL_TOL) or (dmean < 4 * se_pooled)
        ks = _ks_2samp(x_tst, x_ref)
        w1 = _wasserstein1(x_tst, x_ref)
        rows.append(
            dict(
                species=i,
                mean_ref=mean_ref,
                mean_tst=mean_tst,
                dmean=dmean,
                dmean_rel=dmean_rel,
                se_pooled=se_pooled,
                mean_ok=mean_ok_i,
                ks=ks,
                w1=w1,
            )
        )

    print(f"  {'spec':>4s}  {'mean_ref':>10s}  {'mean_tst':>10s}  {'Δmean':>8s}  {'4·SE':>8s}  {'KS':>7s}  {'W1':>9s}")
    for r in rows:
        flag_mean = " " if r["mean_ok"] else "*"
        flag_ks = " " if r["ks"] < ks_crit else "*"
        print(
            f"  {r['species']:>4d}  {r['mean_ref']:>10.3f}  {r['mean_tst']:>10.3f}  "
            f"{r['dmean']:>8.3f}{flag_mean} {4 * r['se_pooled']:>7.3f}  {r['ks']:>6.4f}{flag_ks}  {r['w1']:>9.4f}"
        )

    mean_ok = all(r["mean_ok"] for r in rows)
    ks_ok = all(r["ks"] < ks_crit for r in rows)
    ok = mean_ok and ks_ok
    status = "PASS" if ok else "FAIL"
    print(f"  → {status}  (mean_ok={mean_ok}, ks_ok={ks_ok})")

    np.savez(
        RESULTS_DIR / f"correctness_{model_name}.npz",
        crn_jax=finals["crn_jax"],
        gillespy2=finals["gillespy2"],
    )
    return dict(name=model_name, n=n_trajectories, ok=ok, mean_ok=mean_ok, ks_ok=ks_ok, ks_crit=ks_crit, rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    args = p.parse_args()

    summaries = [_compare(name, args.n, args.seed) for name in args.models]

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  {s['name']:18s}  {'PASS' if s['ok'] else 'FAIL'}")
    if not all(s["ok"] for s in summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
