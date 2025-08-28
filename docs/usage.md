# Usage instructions and config design

This package allows modeling the extreme value distribution (EVD) parameters as functions of covariates. Each parameter can be configured independently:

-   `location_model`: "constant" or "linear"
-   `scale_model`: "constant" or "exponential"
-   `shape_model`: "constant" or "linear"

Internally, these options apply a regression of the form:

-   Linear: θ(t) = θ₀ + θ₁·X(t)
-   Exponential: θ(t) = exp(θ₀ + θ₁·X(t))

This gives flexibility to model non-stationarity while maintaining parsimony.

## Example usage

``` python
import nsEVDx as ns 
from scipy.stats import genextreme
```

#### Input data and covariate (e.g., temperature or year)

``` python
data = [...] # list or numpy array of observed extreme values in chronological order 
covariate = [...] # covariate values (same length as data)
```

#### Know the configuration you want to model

`nsEVDx` supports non-stationarity through a configuration vector `config = [a, b, c]`, where each entry specifies the number of covariates used to model the location, scale, and shape parameters of the extreme value distribution (GEV or GPD), respectively. A value of 0 implies that the corresponding parameter is stationary, while values greater than 0 indicate non-stationary modeling using that many covariates.

#### Instantiate the sampler

``` python
sampler = ns.NonStationaryEVD(data, covariate, config=[1,0,0], dist=genextreme)
'''
It is best practice to instantiate the sampler with explicit prior_specs, or assign user defined prior specifications, rather than relying on default or automatically inferred priors by the MCMC sampler.
'''
```

#### View model parameter descriptions

``` python
print(sampler.descriptions)
# It is recommended to check this output, as a mismatch between the number of prior_specs, 
# length of initial_parameters, step_sizes (for MALA), or proposal_width can raise an error.
```

#### Prior distributions for EVD parameters

``` python
'''
Even if users do not specify the priors, the MCMC samplers will automatically infer them, as the following line is embedded within each sampler
'''
sampler.prior_specs = sampler.suggest_priors()

# The same applies to parameter bounds when using the frequentist method.
```

``` python
# For faster convergence and to avoid errors, defining priors explicitly is more appropriate:
prior_specs = [
    ('normal', {'loc': 10, 'scale': 5}),       # B0 (location intercept)
    ('normal', {'loc': 0, 'scale': 0.05}),     # B1 (location slope with covariate 1)
    ('normal', {'loc': 6, 'scale': 3}),        # sigma (scale parameter)
    ('normal', {'loc': 0, 'scale': 0.3})       # xi (shape parameter)
]
# Users can either pass this `prior_specs` list as an argument when instantiating the sampler,
# or assign it to the sampler later.
```

#### Run Metropolis Adjusted Langevin Sampler

``` python
samples, acceptance_rate = sampler.MH_Mala(
                                    num_samples=10000, 
                                    initial_params=[10, 0.02 , 5, 0.1],
                                    step_sizes=[0.01, 0.001, 0.01, 0.001], 
                                    T=7 
                                    )
```

#### Print the results

``` python
print(f"acceptance_rate : {a_rate}")
np.set_printoptions(suppress=True, precision=6)
print(f"Sample mean : {samples.mean(axis=0)}")
```

#### Trace plots and posterior distributions of the parameter

``` python
ns.plot_trace(samples, config)
ns.plot_posterior(samples, config)
```

See examples such as, [bayesian inference of non-stationary GEV parameters](../examples/example_GEV.ipynb), [bayesian metrics example](../examples/example_bayesian_metrics.ipynb), [frequentist estimation of non-stationary GPD parameter and likelihood ratio test](../examples/example_GPD_frequentist.ipynb), and [generation of random variates from non-stationary GEV](../examples/example_generating_rv_from_nsEVD.ipynb) . These examples highlight the library's key capabilities, including parameter estimation and simulation under non-stationary conditions.
