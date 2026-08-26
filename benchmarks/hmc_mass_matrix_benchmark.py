"""
HMC mass-matrix benchmark for nsEVDx.

Written for #8, which changed the diagonal mass matrix from posterior variance
to posterior precision. Kept here so that convention stays checkable.

Run:
    python hmc_mass_matrix_benchmark.py

Requires nsEVDx importable, plus numpy and scipy. Prints two tables and writes
hmc_mass_matrix_benchmark.json next to itself. A full run takes roughly a quarter
of an hour, almost all of it in Part B, so it is a benchmark rather than a unit
test. Trim DATA_SEEDS, STEP_SIZES or CHAIN_SEEDS below for a shorter run.

What it does
------------
Part A pins down the convention question on its own. It drives
HMCEngine._hmc_step directly with a mass matrix supplied by hand, so it does not
depend on how the package estimates M. The target is an independent Gaussian
with very different scales per dimension, which is the case where the choice of
convention actually shows up. Under K(p) = p^T M^-1 p / 2 and the leapfrog
position update used in hmc_engine.py, M should scale with posterior precision
(Neal 2011 section 4.1). Passing variance instead inverts the preconditioning,
so the wide dimension gets the smaller effective step.

Part B is the end-to-end check: it runs the sampler's own adaptation
(_init_mass_matrix and _warmup) through MH_Hmc on a stationary GEV fit and
reports minimum ESS, so a change to the convention shows up here without any
hand-supplied M. It sweeps three datasets, three step sizes and two chain seeds
and reports the median, because single runs are noisy enough to point the wrong
way: one seed in this sweep degrades after a change that improves the median by
more than an order of magnitude.

Both arms in Part A adapt the step size separately, so the comparison is not one
arm running at a step size tuned for the other.
"""

import json
import os
import time

import numpy as np
import scipy.stats as st

from nsEVDx import NonStationaryEVD
from nsEVDx.hmc_engine import HMCEngine

SEED = 21

# Part B sweep. Trim any of these for a shorter run, at the cost of a noisier
# median: a single configuration is not enough to tell the two conventions apart.
DATA_SEEDS = (2026, 7, 99)
STEP_SIZES = (0.05, 0.02, 0.1)
CHAIN_SEEDS = (123, 456)


# ----------------------------------------------------------------- ESS

def ess_geyer(x):
    """Effective sample size, Geyer initial positive sequence."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return float(n)
    rho, s = [], 0.0
    for lag in range(1, min(n - 1, 2000)):
        r = np.dot(x[:-lag], x[lag:]) / denom
        rho.append(r)
        if lag % 2 == 0:
            pair = rho[-2] + rho[-1]
            if pair < 0:
                break
            s += pair
    return float(min(n, n / (1.0 + 2.0 * s)))


def min_ess(samples):
    return float(min(ess_geyer(samples[:, j]) for j in range(samples.shape[1])))


# ------------------------------------------------- anisotropic Gaussian target

def gaussian_model(sigmas):
    """A NonStationaryEVD whose posterior is an independent Gaussian.

    The log-posterior and its gradient are overridden so the exact answer is
    known and the sampler can be checked against it. Everything else, including
    the leapfrog integrator and the adaptation, is the package's own code.
    """
    dummy = st.genextreme.rvs(c=-0.1, loc=30, scale=10, size=50,
                              random_state=1)
    covar = np.arange(50, dtype=float).reshape(1, -1)
    model = NonStationaryEVD([0, 0, 0], dummy, covar, "genextreme")
    model.prior_specs = []  # never consulted once the posterior is overridden
    sig = np.asarray(sigmas, dtype=float)

    def logp(params):
        return float(-0.5 * np.sum((np.asarray(params) / sig) ** 2))

    def grad(params):
        return -np.asarray(params, dtype=float) / sig ** 2

    model._posterior_log_prob = logp
    model._grad_log_posterior = grad
    return model


def run_arm(engine, M_diag, n_dim, n_keep=4000, n_adapt=1500, n_leapfrog=20,
            seed=SEED):
    """Adapt the step size, then sample, with M_diag held fixed throughout."""
    np.random.seed(seed)
    da = engine._dual_average_init(0.1, 0.8)
    params = np.zeros(n_dim)
    step = 0.1
    step_bar = step
    for _ in range(n_adapt):
        prop, log_alpha = engine._hmc_step(params, step, n_leapfrog, M_diag, 1.0)
        if np.log(np.random.rand()) < log_alpha:
            params = prop
        step, step_bar = engine._dual_average_update(da, log_alpha)
    step = step_bar

    samples = np.empty((n_keep, n_dim))
    accepted = 0
    t0 = time.perf_counter()
    for i in range(n_keep):
        prop, log_alpha = engine._hmc_step(params, step, n_leapfrog, M_diag, 1.0)
        if np.log(np.random.rand()) < log_alpha:
            params = prop
            accepted += 1
        samples[i] = params
    elapsed = time.perf_counter() - t0

    return {
        "adapted_step_size": round(float(step), 5),
        "acceptance": round(accepted / n_keep, 3),
        "ess_per_dim": [round(ess_geyer(samples[:, j]), 1) for j in range(n_dim)],
        "sampled_std": [round(float(samples[:, j].std()), 3) for j in range(n_dim)],
        "seconds": round(elapsed, 2),
    }


def part_a():
    """Convention comparison at three anisotropy ratios."""
    out = {}
    for label, sigmas in (("50_to_1", [50.0, 1.0]),
                          ("10_to_1", [10.0, 1.0]),
                          ("100_to_1", [100.0, 1.0])):
        sig = np.array(sigmas)
        engine = HMCEngine(gaussian_model(sig))
        out[label] = {
            "target_std": sigmas,
            "M_as_variance": run_arm(engine, sig ** 2, len(sig)),
            "M_as_precision": run_arm(engine, 1.0 / sig ** 2, len(sig)),
        }
    return out


def part_b():
    """End-to-end stationary GEV fit using the package's own adaptation.

    Swept over datasets, step sizes and chain seeds. Report the median: a single
    run of this is noisy enough to reverse the apparent direction.
    """
    priors = [("normal", {"loc": 30.0, "scale": 20.0}),
              ("normal", {"loc": 10.0, "scale": 10.0}),
              ("normal", {"loc": 0.0, "scale": 0.3})]
    bounds = [(0.0, 100.0), (0.1, 50.0), (-0.5, 0.5)]
    covar = np.ones((1, 100))
    runs = []

    for data_seed in DATA_SEEDS:
        data = st.genextreme.rvs(c=-0.1, loc=30, scale=10, size=100,
                                 random_state=data_seed)
        model = NonStationaryEVD([0, 0, 0], data, covar, "genextreme",
                                 prior_specs=priors, bounds=bounds)
        start = [float(v) for v in np.ravel(
            model.frequentist_nsEVD([30.0, 10.0, 0.1]))][:3]

        for step in STEP_SIZES:
            for chain_seed in CHAIN_SEEDS:
                np.random.seed(chain_seed)
                t0 = time.perf_counter()
                res = model.MH_Hmc(2000, start, step_size=step,
                                   num_leapfrog_steps=15, burn_in=1000,
                                   num_chains=1, show_progress=False)
                elapsed = time.perf_counter() - t0
                samples = np.asarray(res["chains"])
                if samples.ndim == 3:
                    samples = samples[0]
                runs.append({
                    "data_seed": data_seed,
                    "step_size": step,
                    "chain_seed": chain_seed,
                    "reported_acceptance": round(float(res["a_rate"]), 4),
                    "min_ess": round(min_ess(samples), 1),
                    "posterior_mean": [round(float(v), 3)
                                       for v in samples.mean(axis=0)],
                    "seconds": round(elapsed, 2),
                })

    ess = [r["min_ess"] for r in runs]
    return {
        "truth": [30.0, 10.0, 0.1],
        "n_data": 100,
        "iters": 3000,
        "kept": 2000,
        "component_order": ["location", "scale", "shape"],
        "median_min_ess": round(float(np.median(ess)), 1),
        "worst_min_ess": round(float(np.min(ess)), 1),
        "best_min_ess": round(float(np.max(ess)), 1),
        "n_runs": len(runs),
        "runs": runs,
    }


def main():
    import nsEVDx
    results = {
        "nsEVDx_version": getattr(nsEVDx, "__version__", "unknown"),
        "numpy": np.__version__,
        "seed": SEED,
        "A_convention_hand_supplied_M": part_a(),
        "B_end_to_end_gev_package_adaptation": part_b(),
    }

    print("\nPart A: convention, M supplied by hand")
    print("%-11s %-11s %-9s %-6s %-9s %-11s %s"
          % ("target", "convention", "step", "acc", "ESS wide", "ESS narrow",
             "std wide"))
    for label, block in results["A_convention_hand_supplied_M"].items():
        wide = block["target_std"][0]
        for name in ("M_as_variance", "M_as_precision"):
            r = block[name]
            print("%-11s %-11s %-9s %-6s %-9s %-11s %.2f (target %.0f)"
                  % (label, name.replace("M_as_", ""), r["adapted_step_size"],
                     r["acceptance"], r["ess_per_dim"][0], r["ess_per_dim"][1],
                     r["sampled_std"][0], wide))

    b = results["B_end_to_end_gev_package_adaptation"]
    print("\nPart B: end to end, package adaptation, %d runs" % b["n_runs"])
    print("%-11s %-7s %-7s %-9s %-9s %s"
          % ("data seed", "eps", "chain", "acc", "min ESS", "posterior mean"))
    for r in b["runs"]:
        print("%-11s %-7s %-7s %-9s %-9s %s"
              % (r["data_seed"], r["step_size"], r["chain_seed"],
                 r["reported_acceptance"], r["min_ess"], r["posterior_mean"]))
    print("  median min ESS %s, worst %s, best %s"
          % (b["median_min_ess"], b["worst_min_ess"], b["best_min_ess"]))

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "hmc_mass_matrix_benchmark.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote %s" % path)


if __name__ == "__main__":
    main()
