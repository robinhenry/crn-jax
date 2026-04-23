"""Throughput sweep: trajectories/second at fixed time horizon T=20.

For each (library, model, n_trajectories) cell:
  - 1 warm-up call (incurs JIT compile for JAX libs).
  - ``DEFAULT_REPS`` timed calls; median + IQR reported.
  - JIT-compile time captured separately as the (warmup_time - median_run_time).

Two timings per run are recorded for JAX libs:
  - ``compute`` — ends at ``block_until_ready()`` (pure on-device SSA cost).
  - ``total``   — also includes the ``np.asarray()`` device→host transfer
                  (apples-to-apples with GillesPy2, whose output is already
                  on the host).
For GillesPy2 the two are identical.

GillesPy2 SSACSolver (C++ backend) is used as the CPU baseline.

Outputs results/throughput_<device>.json — one record per cell.

Platform selection: parses --device early and sets ``JAX_PLATFORMS`` *before*
``import jax`` so the requested backend is actually used. (Setting the env var
after JAX initialises is a no-op.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# --- Platform selection: must run before `import jax` -----------------------
def _early_device() -> str:
    for i, a in enumerate(sys.argv):
        if a == "--device" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--device="):
            return a.split("=", 1)[1]
    return "gpu"


_DEVICE = _early_device()
os.environ["JAX_PLATFORMS"] = "cpu" if _DEVICE == "cpu" else "cuda"

import jax  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# (lib, supports_gpu). GillesPy2's SSACSolver is C++-CPU only.
LIBRARIES = [
    ("crn_jax", True),
    ("gillespy2", False),
]

# CPU sweep — crn-jax has no parallelism on CPU, so the slope continues
# linearly past N=1 000. We cap there to keep wall-clock bounded; the
# narrative is GPU-vs-CPU, not the exact CPU asymptote.
DEFAULT_CPU_NS = (10, 100, 1_000)
# 10x increments — ride the GPU to its memory ceiling. Per-cell OOMs are
# caught and recorded so the sweep keeps going. (crn-jax only; GillesPy2
# is CPU-only.)
DEFAULT_GPU_NS = (10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000)
DEFAULT_REPS = 3


def _time_one(run_fn, key_or_seed_fn, n_traj, n_reps=DEFAULT_REPS, warmup=True):
    """Run warmup + n_reps timed runs.

    Each timed call records two phases:
      compute — ends at ``block_until_ready()`` (pure on-device cost).
      total   — also includes the ``np.asarray()`` device→host copy.
    For libs that return host arrays, compute == total.

    Returns (warmup_secs, compute_times_array, total_times_array).
    """

    def _run_and_time(key_or_seed):
        t0 = time.perf_counter()
        out = run_fn(key_or_seed, n_traj)
        # Compute phase: wait for any async on-device work to finish.
        if hasattr(out, "block_until_ready"):
            out.block_until_ready()
        t_compute = time.perf_counter() - t0
        # Transfer phase: pull the result to the host.
        np.asarray(out)
        t_total = time.perf_counter() - t0
        return t_compute, t_total

    warm_t = None
    if warmup:
        t0 = time.perf_counter()
        _run_and_time(key_or_seed_fn(0))
        warm_t = time.perf_counter() - t0

    compute_times, total_times = [], []
    for rep in range(n_reps):
        tc, tt = _run_and_time(key_or_seed_fn(rep + 1))
        compute_times.append(tc)
        total_times.append(tt)
    return warm_t, np.array(compute_times), np.array(total_times)


def _make_runners(model_name: str):
    from benchmarks.models import get

    model = get(model_name)

    # Apples-to-apples: both libraries produce a full (N, n_steps, n_species)
    # trajectory grid sampled at dt=0.1 (no "final state only" shortcuts).
    runners = {
        "crn_jax": (
            lambda i: jax.random.PRNGKey(i),
            lambda key, n: model.run_crn_jax(key, n, return_full_trajectory=True),
        ),
        "gillespy2": (
            lambda i: i + 1,  # gillespy2 wants a positive int seed
            lambda seed, n: model.run_gillespy2(seed, n, return_full_trajectory=True),
        ),
    }
    return runners


def _sweep(model_name: str, ns: list[int], libs: list[str], n_reps: int = DEFAULT_REPS) -> list[dict]:
    runners = _make_runners(model_name)
    rows = []
    for n in ns:
        for lib in libs:
            key_fn, run_fn = runners[lib]
            print(f"  {model_name:>16s} {lib:>10s} N={n:>8d} …", end="", flush=True)
            try:
                warm, ctimes, ttimes = _time_one(run_fn, key_fn, n, n_reps=n_reps)
                c_med = float(np.median(ctimes))
                t_med = float(np.median(ttimes))
                c_tps = n / c_med
                t_tps = n / t_med
                jit_secs = max(0.0, (warm or t_med) - t_med)
                print(
                    f"  compute={c_med * 1000:7.1f}ms total={t_med * 1000:7.1f}ms "
                    f"({c_tps:>11,.0f} | {t_tps:>11,.0f} traj/s)  JIT≈{jit_secs * 1000:6.0f}ms"
                )
                rows.append(
                    dict(
                        model=model_name,
                        lib=lib,
                        n=n,
                        warm_s=warm,
                        # Backward-compat fields (total = compute + D→H transfer).
                        median_s=t_med,
                        q25_s=float(np.quantile(ttimes, 0.25)),
                        q75_s=float(np.quantile(ttimes, 0.75)),
                        traj_per_s=t_tps,
                        # Compute-only fields (ends at block_until_ready).
                        median_compute_s=c_med,
                        q25_compute_s=float(np.quantile(ctimes, 0.25)),
                        q75_compute_s=float(np.quantile(ctimes, 0.75)),
                        traj_per_s_compute=c_tps,
                        jit_compile_s=jit_secs,
                    )
                )
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {str(e)[:80]}")
                rows.append(dict(model=model_name, lib=lib, n=n, error=str(e)))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "gpu"], required=True)
    p.add_argument("--models", nargs="*", default=["birth_death", "linear_cascade"])
    p.add_argument("--n", nargs="*", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--reps",
        type=int,
        default=DEFAULT_REPS,
        help=f"timed reps per cell (default: {DEFAULT_REPS}, plus 1 untimed warmup)",
    )
    args = p.parse_args()

    expected = "cpu" if args.device == "cpu" else "gpu"
    actual = jax.devices()[0].platform
    print(f"JAX devices: {jax.devices()}  (expected {expected})")
    assert (expected == "cpu") == (actual == "cpu"), (
        f"JAX platform mismatch: requested {expected}, got {actual}. JAX_PLATFORMS may have been set too late."
    )

    if args.n is None:
        ns = list(DEFAULT_GPU_NS if args.device == "gpu" else DEFAULT_CPU_NS)
    else:
        ns = args.n

    libs = [lib for lib, gpu_ok in LIBRARIES if args.device == "cpu" or gpu_ok]
    print(f"Libraries on {args.device}: {libs}")

    all_rows = []
    for model_name in args.models:
        print(f"\n--- {model_name} ---")
        rows = _sweep(model_name, ns, libs, n_reps=args.reps)
        all_rows.extend(rows)

    out_path = args.out or RESULTS_DIR / f"throughput_{args.device}.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
