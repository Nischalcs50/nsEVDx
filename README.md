# nsEVDx: A Python Library for Modeling Non-Stationary Extreme Value Distributions

| Usage | Release | Development |
|------|--------|-------------|
| ![Python](https://img.shields.io/badge/python-3.9%2B-blue) | [![PyPI version](https://img.shields.io/pypi/v/nsEVDx)](https://pypi.org/project/nsEVDx/) | ![CI](https://github.com/nischalcs50/nsEVDx/actions/workflows/test.yml/badge.svg) |
| ![License](https://img.shields.io/badge/license-MIT-green) | [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15850043-blue)](https://doi.org/10.5281/zenodo.15850043) | [![codecov](https://codecov.io/gh/nischalcs50/nsEVDx/branch/main/graph/badge.svg)](https://codecov.io/gh/nischalcs50/nsEVDx) |
| [![PyPI downloads](https://pepy.tech/badge/nsEVDx)](https://pepy.tech/project/nsEVDx) | [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://Nischalcs50.github.io/nsEVDx/) | [![GitHub issues](https://img.shields.io/github/issues/nischalcs50/nsEVDx)](https://github.com/nischalcs50/nsEVDx/issues) |
| [![GitHub stars](https://img.shields.io/github/stars/nischalcs50/nsEVDx?style=social)](https://github.com/nischalcs50/nsEVDx) | [![pyOpenSci Peer-Reviewed](https://pyopensci.org/badges/peer-reviewed.svg)](https://github.com/pyOpenSci/software-review/issues/265) |  |


`nsEVDx` is a Python library for estimating the parameters of Generalized Extreme Value (GEV) and Generalized Pareto Distributions (GPD), collectively referred to as extreme value distributions (EVDs), under both stationary and non-stationary assumptions, using frequentist and Bayesian methods. Designed for hydrologists, climate scientists, and engineers, especially those working on extreme rainfall or flood frequency analysis, it supports time-varying covariates, MCMC samplings (Metropolis hasting-Randomwalk, Adjusted Langevin Algorithm, Hamiltonian Monte Carlo) and essential model diagnostics. Although developed for environmental extremes, its features are broadly applicable to financial risk modeling and other domains concerned with rare, high-impact events.

## Features

-   Fits stationary and nonstationary EVDs
-   Supports Frequentist and Bayesian inference
-   Transparent, fully customizable MCMC engine implemented in NumPy
-   Advanced samplers: Metropolis Hasting RandomWalk, Metropolis Adjusted Langevin Algorithm (MALA), and Hamiltonian Monte Carlo (HMC)
-   Support arbitratry covariates in location, scale and shape parameters
-   Integrated diagnostic tools: trace plots, acceptance rates, and bayesian metrics
-   Visualization tool for posterior summaries
-   Lightweight and minimal dependency, only `numpy, scipy, matplotlib, seaborn`

## Implementation

The core `NonStationaryEVD` class handles parameter parsing, log-likelihood construction, prior specification, and proposal generation. Frequentist estimation uses `scipy.optimize` to minimize the negative log-likelihood, while Bayesian MCMC methods are implemented in `numpy` for transparency and flexibility.

Non-stationarity is controlled via a configuration vector `config = [a, b, c]`, where each entry specifies the number of covariates used to model the location, scale, and shape parameters of the EVD. Entry with a value of `0` implies stationarity, while values `> 0` indicate non-stationary modeling using that many covariates.

In Bayesian estimation, `nsEVDx` can infer prior specifications based on the data and configuration or accept user-defined priors. In the frequentist mode, it can determine suitable parameter bounds automatically. However, user defined priors or bounds are recommended for better convergence and interpretability.

### Config design

This package allows modeling the extreme value distribution (EVD) parameters as functions of covariates. Each parameter can be configured independently:

-   `location_model`: "constant" or "linear"
-   `scale_model`: "constant" or "exponential"
-   `shape_model`: "constant" or "linear"

Internally, these options apply a regression of the form:

-   Linear: θ(t) = θ₀ + θ₁·X(t)
-   Exponential: θ(t) = exp(θ₀ + θ₁·X(t))

This gives flexibility to model non-stationarity while maintaining parsimony.


Note: Polynomial relationships between the covariates and the parameters can be modeled by raising the power of the covariates before passing them into the model.

Splines : Comming soon...

## Installation

**For regular users**

``` bash
pip install nsEVDx  

# Or clone from GitHub:
git clone https://github.com/Nischalcs50/nsEVDx
cd nsEVDx
pip install .
```

**For developers/contributors**

``` bash
git clone https://github.com/Nischalcs50/nsEVDx
cd nsEVDx
pip install -e .[dev]
```

## Quick Start Example

``` python
import nsEVDx as ns
import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt

## GIVEN, NON_STATIONARY TIME_SERIES OF EXTREMES
np.random.seed(112)
config = [1, 0, 0] # means location parameter is non-stationary and scale and shape parameters are stationary
# See Usage.md or https://nischalcs50.github.io/nsEVDx/ for more details on config vector
# checking the parameters corresponding to the config
print(ns.NonStationaryEVD.get_param_description(config=config, n_cov=1)) # checking the parameters corresponding to the config
cov  = np.array(range(50))
data = ns.NonStationaryEVD.ns_EVDrvs(genextreme, [30, 0.1, 7, 0.2], cov, config, size=50)
plt.plot(data)


## SETTING PRIORS
# Prior: normal for regression coefficients of location parameter, half-normal for scale, normal for shape
prior_specs = [
    ('normal', {'loc': 30, 'scale':10 }),  
    ('normal', {'loc': 0, 'scale': 0.5}),  
    ('halfnormal', {'loc': 5, 'scale': 5 }),   
    ('normal', {'loc': 0, 'scale': 0.4})  
]
sampler = ns.NonStationaryEVD(config, data, cov,dist=genextreme,
                                  prior_specs=prior_specs)
print(sampler.descriptions)

## RUNNING BAYESIAN ALGORITHM
# fitting a non-stationary GEV model to the data using Hamiltonian Monte Carlo (HMC) sampler
results = sampler.MH_Hmc(
    num_samples=1000, burn_in=1500,
    initial_params=[28, 0, 6, 0],
    num_chains=2,
    T = 1
)

## PRINT RESULTS
print(f"acceptance_rate : {np.round(results['acceptance_rates'], 3)}")
print(f"sample mean : {np.round(np.vstack(results['chains']).mean(axis=0), 3)}")
print(f"r_hats : {np.round(results['r_hats'], 3)}")

## PLOT CONVERGENCE & POSTERIORS
ns.plot_trace(results['chains'], config, fig_size=(8,8))
ns.plot_posterior(results['chains'], config, fig_size=(8,8))
```
full version of this example is available here: [quick_start](examples/Quick_start_example.ipynb)

## Documentation
-   Webpage manual is here [user manual](https://Nischalcs50.github.io/nsEVDx/)
-   Quick start example 2 is [here](examples/Quick_start_example.ipynb) 
-   See examples such as, [bayesian inference of non-stationary GEV parameters](examples/example_GEV.ipynb), [bayesian metrics example](examples/example_bayesian_metrics.ipynb), [frequentist estimation of non-stationary GPD parameter and likelihood ratio test](examples/example_GPD_frequentist.ipynb), and [generation of random variates from non-stationary GEV](examples/example_generating_rv_from_nsEVD.ipynb) . These examples highlight the library's key capabilities, including parameter estimation and simulation under non-stationary conditions.

## Usage

The usage document is available [here](https://Nischalcs50.github.io/nsEVDx/). For more details, see the usage examples in the Jupyter notebooks [here](examples/).

## Dependencies

-   numpy
-   scipy
-   matplotlib, seaborn (for plots)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use `nsEVDx` in your research, please cite:

Kafle, N., & Meier, C. I. (2025). nsEVDx: A Python library for modeling Non-Stationary Extreme Value Distributions. arXiv preprint [arXiv:2509.07261](https://arxiv.org/abs/2509.07261).

Kafle, N., & Meier, C. (2025). nsEVDx: A Python Library for Modeling Non-Stationary Extreme Value Distributions (v0.2.3). Zenodo. https://doi.org/10.5281/zenodo.21286163
Jul 10, 2026

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project, and refer to our [Code of Conduct](CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.


## References

Abril-Pla, O., Andreani, V., Carroll, C., Dong, L. Y., Fonnesbeck, C., Kochurov, M., Kumar, R., Lao, J., Luhmann, C. C., Martin, O. A., Osthege, M., Vieira, R., Wiecki, T. V., & Zinkov, R. (2023). PyMC: A modern, and comprehensive probabilistic programming framework in Python. *PeerJ Computer Science, 9*, e1516. https://doi.org/10.7717/peerj-cs.1516

Betancourt, M. (2017). *A conceptual introduction to Hamiltonian Monte Carlo*. arXiv. https://arxiv.org/abs/1701.02434

Castillo, E. (1988). *Extreme value theory in engineering*. Academic Press.

Eccles, R., Syktus, J., Trancoso, R., Chapman, S., Wasko, C., Evans, J. P., Thatcher, M., Di Virgilio, G., & Stassen, C. (2025). Substantial increases in future precipitation extremes—Insights from a large ensemble of downscaled CMIP6 models. *npj Natural Hazards, 2*(1), 60. https://doi.org/10.1038/s44304-025-00107-1

Gilleland, E. (2025). *extRemes: Extreme value analysis* (Version 2.2-1) [R package]. Comprehensive R Archive Network. https://CRAN.R-project.org/package=extRemes

Heffernan, J. E., Stephenson, A. G., & Gilleland, E. (2003). *ismev: An introduction to statistical modeling of extreme values* (Version 1.42) [R package]. Comprehensive R Archive Network. https://CRAN.R-project.org/package=ismev

Hosking, J. R. M., & Wallis, J. R. (1997). *Regional frequency analysis: An approach based on L-moments*. Cambridge University Press.

Jayaweera, L., Wasko, C., & Nathan, R. (2025). Evidence for non-stationarity in the GEV shape parameter when modeling extreme rainfall. *Water Resources Research, 61*(5), e2023WR036426. https://doi.org/10.1029/2023WR036426

Kafle, N. (2026). *Rain-gauge network effects on the uncertainty and trends in short-duration extreme precipitation* [Doctoral dissertation, The University of Memphis].

Kafle, N., Dell'Aira, F., Chadwick, C., & Meier, C. I. (2026). Robustness of regionally derived, short-duration rainfall depth-duration-frequency estimates to the choice of minimum interevent time: Evidence across climates, raingauge densities, and regionalization approaches. *Journal of Hydrologic Engineering*. https://doi.org/10.1061/JHYEFF/HEENG-6729

Kafle, N., Peleg, N., & Meier, C. I. (2025). Detecting spatially consistent trends in sub-hourly extreme rainfall using a neighborhood-based method. *AGU Fall Meeting Abstracts*, H13G-07. https://ui.adsabs.harvard.edu/abs/2025AGUFMH13G...07K/abstract

Katz, R. W., Parlange, M. B., & Naveau, P. (2002). Statistics of extremes in hydrology. *Advances in Water Resources, 25*(8–12), 1287–1304. https://doi.org/10.1016/S0309-1708(02)00056-8

McNeil, A. J., & Frey, R. (2000). Estimation of tail-related risk measures for heteroscedastic financial time series: An extreme value approach. *Journal of Empirical Finance, 7*(3–4), 271–300. https://doi.org/10.1016/S0927-5398(00)00012-8

Paciorek, C. (2016). *climextRemes: Tools for analyzing climate extremes* (Version 0.3.1) [R package]. Comprehensive R Archive Network. https://CRAN.R-project.org/package=climextRemes

Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv*. https://doi.org/10.48550/arXiv.1912.11554

Prosdocimi, I., Kjeldsen, T. R., & Miller, J. D. (2015). Detection and attribution of urbanization effect on flood extremes using nonstationary flood-frequency models. *Water Resources Research, 51*(6), 4244–4262. https://doi.org/10.1002/2015WR017065

Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli, 2*(4), 341–363. https://doi.org/10.2307/3318418

Robert, C. P., & Casella, G. (2009). *Introducing Monte Carlo methods with R*. Springer. https://doi.org/10.1007/978-1-4419-1576-4

Stan Development Team. (2023a). *CmdStan: The command-line interface to Stan*. https://mc-stan.org/users/interfaces/cmdstan

Stan Development Team. (2023b). *PyStan: The Python interface to Stan*. https://pystan.readthedocs.io/

Yves Deville. (2026). *NSGEV: Non-stationary GEV time series* (Version 0.2.3) [R package]. IRSN. https://github.com/IRSN/NSGEV/
