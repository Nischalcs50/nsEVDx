---
title: 'nsEVDx: A Python Library for Stationary and Non-Stationary Extreme Value Modeling'
authors:
- name: Nischal Kafle
  orcid: 0009-0004-3187-4920
  affiliation: 1
  corresponding: true
- name: Claudio Meier
  orcid: 
  affiliation: 1
affiliations:
- name: Department of Civil, Construction, and Environmental Engineering, University of Memphis, TN, USA
  index: 1
date: 2026-07-05
software_repository_url: https://github.com/Nischalcs50/nsEVDx
archive_url: https://doi.org/10.5281/zenodo.21286163
license: MIT
tags:
  - Generalized Extreme Value
  - Generalized Pareto
  - Partial Duration Series
  - Hydroclimatic Extremes
  - Bayesian
  - MCMC
  - Hamiltonian Monte Carlo
bibliography: paper.bib
csl: apa.csl
---
# Summary
Infrastructure design, insurance pricing, and financial planning all rely on understanding the probability of rare, extreme events, such as extreme floods, rainfall events, and stock market crashes. Traditionally, researchers estimated the probability of extreme events of a given magnitude by assuming that their probability of occurrence remained stationary over time. However, changing global conditions such as climate change, urbanization, and economic shifts can cause the probability of extreme events to become non-stationary. `nsEVDx` is an open-source Python package developed to help researchers quantify these changing risks by fitting both stationary and non-stationary Extreme Value Distributions (EVDs) to extreme value data. It supports non-stationary modeling in which the location, scale, and shape parameters can each depend on user-defined covariates. This enables practical applications such as linking rainfall extremes to temperature or maximum stock market losses to market volatility indices. Overall, `nsEVDx` provides researchers and practitioners with an easy-to-use, flexible, and extensible Python-native interface for frequentist and Bayesian inference, including MCMC methods.


# Statement of Need

Traditional extreme value frequency analysis assumes that the statistical properties of extreme events remain stationary over time. However, non-stationary behavior in extreme events has been documented across hydroclimatic, engineering, and financial applications. For example, in hydroclimatic systems, non-stationary behavior has been documented in flood and extreme-rainfall processes, with changes linked to factors such as urbanization and climate change [@prosdocimi2015detection;@jayaweera2025evidence;@eccles2025substantial]. Such changes can alter the frequency and magnitude of extreme events over time. Accurately quantifying these evolving risks and return periods requires statistical tools that can incorporate arbitrary, time-varying covariates into all parameters of EVDs.

While several software tools support extreme value analysis, implementing and evaluating non-stationary EVD models often requires substantial statistical and programming expertise, particularly when Bayesian inference and advanced sampling algorithms are involved. `nsEVDx` provides an easy-to-use framework for fitting stationary and non-stationary Generalized Extreme Value (GEV) and Generalized Pareto (GPD) models. It makes advanced sampling methods accessible to users with varying levels of expertise in extreme value modeling while retaining the flexibility required for advanced applications.

The target audience for `nsEVDx` includes hydrologists, climate scientists, infrastructure engineers, and financial risk analysts who require a reliable and flexible workflow for non-stationary extreme value modeling. By reducing the implementation effort  needed to implement both frequentist optimization and advanced Markov Chain Monte Carlo (MCMC) sampling algorithms, the software allows practicioners to focus on understanding changing risks and their implications for decision-making.


# State of the Field

Extreme value distributions (EVDs), particularly the GEV and GPD distributions, are widely used to quantify the frequency and magnitude of rare events in hydrology, climatology, engineering, and finance [@katz2002statistics;@kafle2026robustness;@mcneil2000estimation;@castillo1988extreme]. Increasing evidence of non-stationarity in environmental and socioeconomic systems has motivated the development of methods that allow EVD parameters to vary with time or external covariates.

Several established software packages support extreme value analysis. In the R ecosystem, `ismev` [@heffernan_j_e_ismev_2003], `extRemes` [@gilleland_extremes_2025], `climextRemes` [@paciorek_climextremes_2016], and `NSGEV` [@irsn_nsgev_2024] provide tools for fitting stationary and non-stationary extreme value models. These packages provide important foundations for EVD-based analysis, although their capabilities differ in areas such as non-stationary modeling and Bayesian inference. Probabilistic programming frameworks, such as python-based `PyMC` [@oriol_abril-pla_pymc_2023] and `NumPyro` [@phan2019composable], and `C++` based Stan with interfaces like `PyStan` [@noauthor_pystan_2023] and `CmdStan` [@noauthor_cmdstan_2023], offer flexible tools for building custom statistical models, including extreme value models. However, these tools require users to explicitly formulate non-stationary likelihood functions for EVDs, prior structures, parameter constraints, and sampling configurations for each application. As a result, they may be too complex for domain specific practitioners like hydrologists, climate analysts, or risk analysts seeking easy-to-use tools. 

These differences motivate a dedicated Python tool that combines flexibility with an accessible interface for stationary and non-stationary EVD modeling. In particular, `nsEVDx` supports user-defined covariates, parameter constraints, custom priors, and MCMC algorithms such as MALA [@roberts_exponential_1996] and HMC [@michael_betancourt_conceptual_2017] for fitting non-stationary GEV and GPD models. Compared with general-purpose probabilistic programming frameworks, `nsEVDx` provides predefined EVD likelihoods, parameterizations, and inference workflows, reducing the amount of model construction required for common extreme value analyses. 


# Software Design
`nsEVDx` is designed to provide a transparent and researcher-focused framework for fitting stationary and non-stationary extreme value distributions while minimizing the programming burden typically associated with custom Bayesian implementations. The overall workflow implemented in `nsEVDx` is shown in Figure 1.

![Workflow implemented in `nsEVDx` for specifying, fitting, and diagnosing stationary and non-stationary GEV and GPD models using frequentist optimization or Bayesian MCMC sampling](Fig1.png)

The concepts of non-stationarity and MCMC techniques used in `nsEVDx` are based on the foundational texts by @christian_p_robert_introducing_2009 and @coles_introduction_2001. The core class `NonStationaryEVD` handles parameter parsing and specification, log-likelihood evaluation, prior assignment, optimization, and MCMC sampling. This unified design allows users to switch between frequentist and Bayesian workflows without redefining model structures. Frequentist methods uses `scipy.optimize` to minimize the non-stationary negative log-likelihood, while the Bayesian MCMC methods are implemented in numpy, allowing full transparency and customization. In Bayesian estimation, `nsEVDx` can infer prior specifications based on the data and configuration or accept user-defined priors. In the frequentist mode, it can determine suitable parameter bounds automatically. However, user-defined priors or bounds are recommended for better convergence and interpretability. The implementation of L-moments in some utility methods follows the approach described by @j_r_m_hosking_regional_1997. Currently, `nsEVDx` supports linear modeling for the location and shape parameters, and exponential (log-linear) modeling for the scale parameter, to ensure positivity.

`nsEVDx` controls non-stationarity via a configuration vector `config = [a, b, c]`, where each entry specifies the number of covariates used to model the location, scale, and shape parameters of the EVD. Entry with a value of `0` implies stationarity (i.e., no covariate dependence), while integer values `> 0` indicate non-stationary modeling using the corresponding number of covariates for the parameter. This representation provides a compact and scalable mechanism for defining a wide range of stationary and non-stationary models while maintaining a consistent user interface (Figure 2). 

![Configuration-vector framework used in `nsEVDx` for specifying stationary and non-stationary EVD models.](Fig2.png)

Currently, location and shape parameters can be modeled using linear relationships, while the scale parameter is modeled using an exponential link function to ensure positivity. Polynomial relationships can be accommodated by transforming covariates before model fitting. Future releases aim to incorporate spline-based and mixed-population formulations.

Another key design decision is the implementation of Bayesian samplers directly in `NumPy`. This approach maximizes transparency, enables users to inspect and modify algorithmic details, and reduces dependencies. The package currently provides Random-Walk Metropolis-Hastings (RWMH), Metropolis-Adjusted Langevin Algorithm (MALA), and Hamiltonian Monte Carlo (HMC) samplers, allowing users to balance computational efficiency and inferential robustness according to their application. While RWMH and MALA are integrated within the `NonStationaryEVD` class, HMC is implemented through a dedicated `HMCEngine` component that handles the gradient computations required for Hamiltonian dynamics.

To support model evaluation and reproducibility, `nsEVDx` includes diagnostic tools for trace plots, posterior summaries, acceptance rates, and Gelman-Rubin ($\hat{R}$) convergence metrics. It also provides model selection options, including DIC, AIC, BIC, and likelihood-ratio tests, enabling complete workflows within a single environment.

The package is intentionally lightweight, relying primarily on `NumPy` and `SciPy` for computation and `Matplotlib` and `Seaborn` for visualization. This minimal dependency footprint improves portability and simplifies installation while preserving extensibility for future methodological developments.


# Research Impact Statement

`nsEVDx` is developed to improve accessibility of advanced extreme value modeling methods for researchers working in Python environments. The software lowers the barrier to applying advanced Bayesian methods, including MALA and HMC sampling, by providing predefined extreme value models and inference workflows that accommodate users with different levels of statistical and programming expertise.

The package has already been applied in ongoing hydroclimatic research focused on trend detection and characterization of extreme precipitation[@kafle2026raingauge;@kafle2025detecting], and supports broader applications in climate science, water resources engineering, infrastructure design, and financial risk assessment. Since its public release, nsEVDx has received more than 3,200 downloads during its first year, providing evidence of early adoption by the research community.

The software is openly developed, documented, tested, and distributed through both `GitHub` and `PyPI`, supporting long-term reuse and community contributions. Its combination of accessible model specification, flexible non-stationary formulations, and built-in Bayesian inference fills an important gap between specialized extreme-value software and general-purpose probabilistic programming environments.

# Installation and Example

Installation instructions, tutorials, API documentation, and executable examples are available in the [project GitHub repository](https://github.com/Nischalcs50/nsEVDx) and [full online documentation](https://nischalcs50.github.io/nsEVDx/).

# AI Usage Disclosure

AI tools were used during the programming and development of this library, as well as for drafting, editing, and improving the clarity of the manuscript. All underlying scientific content, core software architecture, analyses, and final interpretations were conceived, verified, and approved entirely by the authors.

# Acknowledgements

We are deeply grateful to the JOSS Editor and reviewers for their thorough and constructive review, which substantially strengthened both the nsEVDx package and its documentation. We also thank the OpenSciPy editors and reviewers for their valuable feedback and contributions to improving the package. We gratefully acknowledge Vincent Gao for identifying and fixing a software bug. Finally, we thank my wife, Koshika Timsina, for her steadfast support and encouragement throughout this project.

# References
