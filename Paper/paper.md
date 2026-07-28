---
title: 'nsEVDx: A Python library for stationary and non-stationary extreme value modeling'
authors:
- name: Nischal Kafle
  orcid: 0009-0004-3187-4920
  affiliation: 1
  corresponding: true
- name: Claudio Meier
  affiliation: 1
affiliations:
- name: Department of Civil Engineering, University of Memphis, USA
  index: 1
date: 2026-07-05
software_repository_url: https://github.com/Nischalcs50/nsEVDx
archive_url: https://zenodo.org/record/XXXXXXX
license: MIT
tags:
  - Generalized extreme value
  - Generalized pareto
  - Partial duration series
  - Hydroclimatology
  - Bayesian
  - MCMC
  - Hamiltonian
bibliography: paper.bib
csl: apa.csl
---
# Summary
Infrastructure design, insurance pricing, and financial planning all rely on understanding the probability of rare, extreme events, such as extreme floods, rainfall events, and stock market crashes. Traditionally, researchers estimated the probability of extreme events of a given magnitude by assuming that their probability of occurrence remained stationary over time. However, due to changing global conditions such as climate change, urbanization, and economic shifts, the probability of extreme events has become non-stationary.
`nsEVDx` is an open-source Python package developed to help researchers quantify these changing risks by fitting both stationary and non-stationary Extreme Value Distributions (EVDs) to extreme value data. A key feature of `nsEVDx` is its flexible support for non-stationary modeling, where the location, scale, and shape parameters can each depend on user-defined covariates. This enables practical applications such as linking rainfall extremes to temperature or maximum stock market losses to market volatility indices. Overall, `nsEVDx` aims to serve as a practical, easy-to-use, and extensible tool for researchers and practitioners analyzing extreme events in non-stationary environments. The software packages complex statistical modeling techniques into an accessible interface, offering both traditional frequentist and advanced Bayesian methods.


# Statement of Need

Traditional extreme value frequency analysis assumes that the statistical properties of extreme events remain stationary over time. However, driven by environmental and socioeconomic shifts such as climate change, rapid urbanization, and structural market volatility, the probability distribution of extremes are increasingly non-stationary [cite_something]. Accurately quantifying shifting risks and return periods requires statistical tools that can seamlessly incorporate arbitrary, time-varying covariates into all parameters of extreme value distributions (EVDs).

`nsEVDx` is designed to solve the critical accessibility gap between highly flexible but complex general-purpose probabilistic programming languages (PPLs) and rigid, domain-specific packages that lack native Python execution. Building custom, mathematically sound, non-stationary EVD models with custom prior boundaries and gradient-based Bayesian samplers requires significant programmatic and statistical expertise, which presents a steep barrier to entry for applied researchers. 

The target audience for `nsEVDx` consists of hydrologists, climate scientists, infrastructure engineers, and financial risk analysts who require a transparent, reliable, and mathematically rigorous pipeline for non-stationary extreme value modeling. By lowering the programmatic overhead needed to implement both frequentist optimization and advanced Markov Chain Monte Carlo (MCMC) inference algorithms (e.g., MALA and HMC), the software allows domain experts to focus on interpreting non-stationary trends and making defensible engineering decisions under evolving environmental conditions.


# Statement of the Field

Probabilistic modeling of extreme events is essential across disciplines, from resilient infrastructure design and climate adaptation to insurance pricing and financial risk management. In many real-world processes, the statistical properties of the extremes are often non-stationary, driven by long-term changes such as climate change, urbanization, or economic shifts. Accurately estimating return periods and risks under these evolving conditions requires fitting non-stationary extreme value distributions (EVDs) to observations.

Several R packages currently support EVD modeling, including `ismev` [@heffernan_j_e_ismev_2003], `extRemes` [@gilleland_extremes_2025], `climextRemes` [@paciorek_climextremes_2016], and `NSGEV` [@irsn_nsgev_2024]. However, these packages differ in their ability to handle non-stationary models and Bayesian inference. Moreover, extending their functionality and integrating modern inference techniques can be challenging. Probabilistic programming frameworks, such as python-based `PyMC` [@oriol_abril-pla_pymc_2023], and `C++` based Stan with interfaces like `PyStan` [@noauthor_pystan_2023] and `CmdStan` [@noauthor_cmdstan_2023], offer powerful tools for building custom statistical models, including those for extreme value analyses. However, these tools require significant expertise in both statistics and programming to develop, tune, and validate the models effectively. As a result, they may be too complex for domain experts like hydrologists, climate scientists, or risk analysts seeking easy-to-use methods. These limitations motivate the need for a python tool that balances flexibility and ease of use, while supporting arbitrary covariates, parameter constraints, custom priors, and advanced MCMC algorithms such as MALA [@roberts_exponential_1996], HMC [@michael_betancourt_conceptual_2017], for fitting non-stationary Generalized Extreme Value (GEV) and Generalized Pareto (GPD) distributions, the two most prominent EVDs. 
We advance `nsEVDx`, a flexible, user-friendly python package that streamlines non-stationary EVD modeling without compromising statistical rigor. `nsEVDx` has been applied in hydrology [@kafle_evaluating_2025] and is applicable to fields like climate science, finance, and engineering, where it is critical to understand the frequency and intensity of extremes under non-stationarity conditions. Its application is also reflected in an upcoming technical paper on trends in short-duration extreme rainfall in the Southeastern U.S. [@kafle_detecting_nodate]. Compared with existing packages, nsEVDx provides a Python-native implementation with built-in Bayesian MCMC algorithms and flexible support for user-specified covariates, while avoiding the need to manually construct EVD models as required in `PyMC` or `NumPyro`.


# Software Design
`nsEVDx` is designed to provide a transparent and researcher-focused framework for fitting stationary and non-stationary extreme value distributions while minimizing the programming burden typically associated with custom Bayesian implementations. The concepts of non-stationarity and MCMC techniques used in `nsEVDx` are based on the foundational texts by @christian_p_robert_introducing_2009 and @coles_introduction_2007. The core class `NonStationaryEVD` handles parameter parsing and specification, log-likelihood evaluation, prior assignment, optimization, and MCMC sampling. This unified design allows users to switch between frequentist and Bayesian workflows without redefining model structures. Frequentist methods uses `scipy.optimize` to minimize the non-stationary negative log likelihood, while the Bayesian MCMC methods are implemented in numpy, allowing full transparency and customization. The implementation of L-moments in some utility methods follows the approach described by @j_r_m_hosking_regional_1997. Currently, `nsEVDx` supports linear modeling for the location and shape parameters, and exponential (log-linear) modeling for the scale parameter, to ensure positivity.

`nsEVDx` controls non-stationarity via a configuration vector `config = [a, b, c]`, where each entry specifies the number of covariates used to model the location, scale, and shape parameters of the EVD. Entry with a value of `0` implies stationarity (i.e., no covariate dependence), while integer values `> 0` indicate non-stationary modeling using the corresponding number of covariates for the parameter. This representation provides a compact and scalable mechanism for defining a wide range of stationary and non-stationary models while maintaining a consistent user interface. Currently, location and shape parameters can be modeled using linear relationships, while the scale parameter is modeled using an exponential link function to ensure positivity. Polynomial relationships can be accommodated by transforming covariates before model fitting. Future releases aim to incorporate spline-based and mixed-population formulations.

In Bayesian estimation, `nsEVDx` can infer prior specifications based on the data and configuration or accept user-defined priors. In the frequentist mode, it can determine suitable parameter bounds automatically. However, user-defined priors or bounds are recommended for better convergence and interpretability. Planned extensions include mixed population models with categorical covariates, an emerging area in hydroclimatic extremes, along with computational optimizations to improve performance.

# Research Impact Statement






# Features

-   Supports both the Generalized Extreme Value (GEV) and Generalized Pareto (GPD) distributions
-   Non-stationary modeling via linear and log-linear relationships between parameters and covariates
-   Independent non-stationarity in location, scale, and shape parameters
-   Frequentist and Bayesian inference support
-   MCMC algorithms: Random Walk, Metropolis-adjusted Langevin (MALA), and Hamiltonian Monte Carlo (HMC)
-   Custom priors, parameter bounds, and temperature scaling for tuning MCMC
-   Integrated diagnostics: trace plots, convergence checks, and posterior visualization
-   Modular and extensible API designed for ease of use by domain scientists
-   Model comparison using the Deviance Information Criterion (DIC), Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and likelihood ratio tests






# Installations

Install the package via pip: `pip install nsEVDx`

or alternatively, clone the repository and install manually:

```         
git clone https://github.com/Nischalcs50/nsEVDx.git
cd nsEVDx
pip install .
```

# Example usage

``` python
import numpy as np
import nsEVDx as ns

# 1. Generate Dummy Data
np.random.seed(42)
n = 30
t = np.linspace(0, 1, n)
data = np.random.gumbel(loc=20 + 5 * t, scale=5, size=n)

# 2. Setup Model
cov = t.reshape(1, -1)
config = [1, 0, 0] 
model = ns.NonStationaryEVD(config, data, cov, "gev")
# config = [1,0,0] means, location parameter is modeled linearly
# with covariate, while scale and shape are treated as stationary
# Priors are inferred from the data if not provided while 
print(model.descriptions) # provides the parameter descriptions 

# 3. Run
# B0, B1, sigma, xi
initial_params = np.array([20.0, 0.1, 5, 0])
samples, acceptance_rate, r_hat = model.MH_RandWalk(
    num_samples=2500,
    initial_params=[10, 0.02 , 5, 0], 
    # B0(location intercept), B1 (location slope), scale, shape
    proposal_widths=[2, 0.08, 0.75, 0.1],
    T=1.5, burn_in = 2000,
    num_chains = 4, n_jobs=4,
    )

# 4. PRINT RESULTS
print(f"acceptance_rate : {acceptance_rate}")
print(f"r_hat : {r_hat}")
np.set_printoptions(suppress=True, precision=6)
sample_all_chains = np.vstack(samples) 
sample_mean = sample_all_chains.mean(axis=0)
print(f"Sample mean : {sample_mean}")

# 5. PLOT CONVERGENCE & POSTERIORS
ns.plot_trace(samples, config, fig_size=(7, 10))
ns.plot_posterior(samples, config, fig_size=(7, 8))
```

See full documentation at: <https://nischalcs50.github.io/nsEVDx/>

# AI Usage Disclosure

AI tools were used during the programming and development of this library, as well as for drafting, editing, and improving the clarity of the manuscript. All underlying scientific content, core software architecture, analyses, and final interpretations were conceived, verified, and approved entirely by the authors.

# Acknowledgements

I gratefully acknowledge the support and encouragement of my wife, Koshika Timsina, whose constant belief in me has been a source of strength throughout this project. I also extend my heartfelt thanks to my family for their unwavering love, patience, and support.

# References
