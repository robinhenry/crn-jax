# crn-jax benchmarks

Throughput and correctness comparison of `crn-jax` against
[GillesPy2](https://github.com/StochSS/GillesPy2)'s C++ `SSACSolver` on two
reaction networks: a 1-species birth-death process and a 10-species linear
cascade. See the headline chart in the main [README](../README.md).

## Reproducing

```bash
# crn-jax with CUDA (or `pip install -e .` for CPU only) + benchmark deps.
pip install -e ".[cuda]"
pip install -e ./benchmarks

# Correctness: 2-sample KS test vs GillesPy2 at N=10 000.
python benchmarks/run_correctness.py

# Throughput sweeps on CPU and GPU.
python benchmarks/run_throughput.py --device cpu
python benchmarks/run_throughput.py --device gpu

# Render plots into benchmarks/figures/.
python benchmarks/plot_results.py
```
