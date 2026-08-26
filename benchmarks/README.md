# Benchmarks

Scripts here measure sampler behaviour. They are not part of the test suite and are not run in
CI, because they take minutes rather than seconds.

A full run of the mass-matrix benchmark took about **15 minutes** on a laptop i7-12700H, nearly
all of it in Part B. `DATA_SEEDS`, `STEP_SIZES` and `CHAIN_SEEDS` at the top of the script can be
trimmed for a shorter run, at the cost of a noisier median.

## `hmc_mass_matrix_benchmark.py`

Checks the diagonal mass matrix convention in `nsEVDx/hmc_engine.py`, which #8 changed from
posterior variance to posterior precision.

```
python benchmarks/hmc_mass_matrix_benchmark.py
```

Prints two tables and writes `hmc_mass_matrix_benchmark.json` alongside itself.

**Part A** supplies the mass matrix by hand and drives `HMCEngine._hmc_step` directly, so it
measures the convention without depending on how the package estimates M. The target is an
independent Gaussian with very different scales per dimension, which is where the choice shows
up. Under `K(p) = p^T M^-1 p / 2` and the leapfrog update in `hmc_engine.py`, M should scale
with posterior precision (Neal 2011, section 4.1). Both arms adapt the step size separately, so
neither runs at a step size tuned for the other.

**Part B** runs the package's own adaptation through `MH_Hmc` on a stationary GEV fit, sweeping
three datasets, three step sizes and two chain seeds, and reports the median minimum effective
sample size. It sweeps rather than running once because single runs are noisy enough to point
the wrong way.

Effective sample size uses Geyer's initial positive sequence.
