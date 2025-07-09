------------------------------------------------------------------------

------------------------------------------------------------------------

<a id="nsEVD"></a>

# nsEVD

<a id="nsEVD.evd_model"></a>

# nsEVD.evd_model

<a id="nsEVD.evd_model.NonStationaryEVD"></a>

## NonStationaryEVD Objects

``` python
class NonStationaryEVD()
```

<a id="nsEVD.evd_model.NonStationaryEVD.__init__"></a>

## \_\_init\_\_

``` python
def __init__(config, data, cov, dist, prior_specs=None, bounds=None)
```

Instantiate a object(reffered to as 'sampler') of class NonStationaryEVD.

##### Parameters

**config** : list of int. Represents a non-stationarity configuration for [location, scale, shape].

**What is config ?**

    Config denotes the configuration of the non-stationary model. In 
    config = [p, q, r], p, q, and r denote the number of covariates for the location
    , scale, and shape parameters of the EVD. A value of 0 implies stationarity;
    a value >0 indicates non-stationarity with that many covariates.

For examples:

    config = [0,0,0] indicates [stationary_location, stationary_scale
    ,stationary_shape], a full stationary configuration.

    config = [1,0,0] indicates [locatin modeled with 1 covariate, stationary
    scale, stationary shape].

    config = [2,1,0] indicates [location modeled with 2 covariates, scale
    modeled with 1 covariate, stationary shape].

> *Note:* The location and shape parameters are modeled linearly, whereas the scale parameter is modeled exponentially:

    - `location_model`: "constant" or "linear"
    - `scale_model`: "constant" or "exponential"
    - `shape_model`: "constant" or "linear"

    Internally, these options apply a regression of the form:

    -   Linear: θ(t) = θ₀ + θ₁·X(t)

    -   Exponential: θ(t) = exp(θ₀ + θ₁·X(t)), where X represent
        covariates

**data** : array-like Observed extremes in chronlogical order (e.g., annual maxima).

**cov** : array-like Covariate matrix, shape (n_covariates, n_samples).

**dist** : scipy.stats distribution object (genextreme or genpareto).

**prior_specs** : list of tuples Optional prior specifications for each parameter. Required if performing bayesian sampling.

Format: `[(dist_name, params_dict), ...]`

`e.g., [('normal', {'loc': 0, 'scale': 10}), ('uniform',{'loc': 0, 'scale': 5}), ...]`

**bounds** : List of tuples. Optional bounds for each parameter, required if estimating the the parameters by frequentist approach.

##### Returns

**NonStationaryEVD** **object :** An instance of the NonStationaryEVD class initialized with the specified configuration, data, covariates, and distribution.

<a id="nsEVD.evd_model.NonStationaryEVD.get_param_description"></a>

## get_param_description

``` python
@staticmethod
def get_param_description(config: List[int], n_cov: int) -> List[str]
```

Returns a list of strings describing each parameter's role in the parameter vector, based on the provided configuration.

##### Parameters

**config** : list of int Non-stationarity configuration [location, scale, shape].

**n_cov** : int Total number of covariates available.

##### Returns

**list of str**

Descriptions of each parameter in order.

For instance, config = [1,0,0] and n_cov = 1, would have four parameters as follows:

`['B0 (location intercept)', 'B1 (location slope for covariate 1)', 'scale, 'shape']`

Whereas, config = [2,2,2] and n_cov =2, would have nine parameters as follows:

`['B0 (location intercept)', 'B1 (location slope for covariate 1)', 'B2 (location slope for covariate 2)',` `'a0 (scale intercept)', 'a1 (scale slope for covariate 1)', 'a2 (scale slope for covariate 2)',` `'k0 (shape intercept)', 'k1 (shape slope for covariate 1)','k2 (shape slope for covariate 2)']`

<a id="nsEVD.evd_model.NonStationaryEVD.suggest_priors"></a>

## suggest_priors

``` python
def suggest_priors()
```

Suggest default prior distributions for model parameters based on the current configuration and data statistics.

##### Returns

**prior_specs** : list of tuples

List of prior specifications for each parameter in the order expected by the sampler. Each element is a tuple like (distribution_name, distribution_parameters_dict).

<a id="nsEVD.evd_model.NonStationaryEVD.suggest_bounds"></a>

## suggest_bounds

``` python
def suggest_bounds(buffer: float = 0.5) -> List[Tuple[float, float]]
```

Suggests bounds for MLE optimization based on config and distribution type.

##### Parameters

**buffer** : float

Fractional buffer around stationary parameter estimates.

##### Returns

**bounds** : List[Tuple[float, float]]

List of (lower, upper) tuples for each parameter in order.

<a id="nsEVD.evd_model.NonStationaryEVD.log_prior"></a>

## log_prior

``` python
def log_prior(params)
```

Compute the log prior probability of the parameter vector.

This method calculates the sum of log-prior probabilities for each parameter based on the specified prior distributions in `self.prior_specs`. The number and type of parameters are determined by the non-stationarity configuration (`self.config`) provided at initialization.

##### Parameters

**params** : array-like

A 1D array of parameter values corresponding to the linear or exponential models for location, scale, and shape parameters. The number and order of parameters must match the configuration.

##### Returns

**float :** The total log-prior probability of the parameter vector. Returns -np.inf if any prior evaluates to a non-finite value.

*Notes:*

-   Supports 'normal', 'uniform', and 'halfnormal' priors.

-   If no `prior_specs` are provided (i.e., None), returns 0.0 (flat prior).

-   Prior specification format:

    prior_specs = [('normal', {'loc': 0, 'scale': 10}), ...]

<a id="nsEVD.evd_model.NonStationaryEVD.neg_log_likelihood"></a>

## neg_log_likelihood

``` python
def neg_log_likelihood(params)
```

Compute the negative log-likelihood for the given parameter vector.

This method delegates the calculation to the `neg_log_likelihood_ns` function using the class attributes such as data, covariates, model configuration, and distribution type.

##### Parameters

**params** : array-like

A 1D array of model parameters corresponding to the location, scale, and shape components of the non-stationary distribution.

##### Returns

**float :** The negative log-likelihood value.

<a id="nsEVD.evd_model.NonStationaryEVD.posterior_log_prob"></a>

## posterior_log_prob

``` python
def posterior_log_prob(params)
```

Compute the log posterior probability for the given parameter vector.

The posterior is calculated as the sum of the log-prior and the log-likelihood (negated). This is used for Bayesian inference, particularly in MCMC sampling.

##### Parameters

**params** : array-like A 1D array of parameter values matching the model configuration.

##### Returns

**float :** The log posterior probability. If the prior is improper or evaluates to -inf, the result will reflect that.

<a id="nsEVD.evd_model.NonStationaryEVD.numerical_grad_log_posterior"></a>

## numerical_grad_log_posterior

``` python
def numerical_grad_log_posterior(params, h=1e-2)
```

Compute the numerical gradient of the log-posterior with respect to parameters.

This uses the central difference method to approximate the gradient of the log-posterior at the given parameter vector.

##### Parameters

**params** : array-like

A 1D array of parameter values at which to evaluate the gradient.

**h** : float or array-like, optional

The step size for finite difference approximation. Can be a scalar or an array of the same shape as `params` for per-parameter step sizes. Default is 1e-2.

##### Returns

**grad** : ndarray A 1D array containing the approximate gradient of the log-posterior with respect to each parameter.

<a id="nsEVD.evd_model.NonStationaryEVD.MH_Mala"></a>

## MH_Mala

``` python
def MH_Mala(num_samples: int,
            initial_params: Union[list[float], np.ndarray],
            step_sizes: list[float],
            T: float = 1.0) -> tuple[np.ndarray, float]
```

Perform MALA sampling to generate samples from the posterior distribution.

##### Parameters

**num_samples** : int

Number of samples to generate.

**initial_params** : Union[list[float], np.ndarray]

Initial parameter vector to start the Markov chain.

**step_size** : float

Step size epsilon for MALA proposals.

**T** : float, optional

Temperature factor to scale the acceptance ratio. Values greater than 1 make acceptance more lenient, values less than 1 stricter. Default is 1.

##### Returns

**samples** : np.ndarray

Array of shape `(num_samples, n_parameters)` containing sampled parameter vectors.

**acceptance_rate** : float Fraction of proposals accepted.

<a id="nsEVD.evd_model.NonStationaryEVD.MH_RandWalk"></a>

## MH_RandWalk

``` python
def MH_RandWalk(num_samples: int, initial_params: Union[list[float],
                                                        np.ndarray],
                proposal_widths: Union[list[float], np.ndarray],
                T: float) -> tuple[np.ndarray, float]
```

Perform Metropolis-Hastings sampling to generate samples from the posterior distribution.

##### Parameters

**num_samples** : int Number of samples to generate.

**initial_params** : Union[list[float], np.ndarray] Initial parameter vector to start the Markov chain.

**proposal_widths** : Union[list[float], np.ndarray] Standard deviations for the Gaussian proposal distribution (random walk). Length must match the number of parameters.

**T** : float Temperature factor to scale the acceptance ratio. Values greater than 1 make the acceptance more lenient, values less than 1 make it stricter.

###### Raises

ValueError If the length of `proposal_widths` does not match the number of parameters to sample.

##### Returns

**np.ndarray :** Array of shape `(num_samples, n_parameters)` containing the sampled parameter vectors.

**acceptance_rate** : float Fraction of proposals accepted.

<a id="nsEVD.evd_model.NonStationaryEVD.hamiltonian"></a>

## hamiltonian

``` python
def hamiltonian(params, momentum, T)
```

Compute the Hamiltonian (total energy) of the system for HMC sampling.

The Hamiltonian is the sum of the potential energy and kinetic energy. In this context: - Potential energy is defined as the negative log-posterior (scaled by T),which encourages high-probability regions of parameter space. - Kinetic energy is computed as 0.5 \* sum(momentum\^2), assuming a standard Gaussian momentum distribution.

##### Parameters

**params** : array-like Current position in parameter space (model parameters).

**momentum** : array-like Auxiliary momentum variables, typically sampled from a standard normal distribution.

**T** : float. Temperature scaling factor. T=1 corresponds to standard HMC; higher values flatten the posterior (tempering).

##### Returns

**float** : The total Hamiltonian energy (scaled potential + kinetic energy).

<a id="nsEVD.evd_model.NonStationaryEVD.MH_Hmc"></a>

## MH_Hmc

``` python
def MH_Hmc(num_samples: int,
           initial_params: Union[list[float], np.ndarray],
           step_size: float = 0.1,
           num_leapfrog_steps: int = 10,
           T: float = 1.0) -> tuple[np.ndarray, float]
```

Perform HMC sampling to generate samples from the posterior distribution.

##### Parameters

**num_samples** : int Number of samples to generate.

**initial_params** : Union[list[float], np.ndarray] Initial parameter vector to start the Markov chain.

**step_size** : float Step size (epsilon) for the leapfrog integrator.

**num_leapfrog_steps** : int Number of leapfrog steps per iteration.

**T** : float, optional Temperature scaling factor for log-acceptance ratio.

##### Returns

**samples** : np.ndarray Array of shape (num_samples, n_parameters) containing parameter vectors. **acceptance_rate** : float Fraction of proposals accepted.

<a id="nsEVD.evd_model.NonStationaryEVD.frequentist_nsEVD"></a>

## frequentist_nsEVD

``` python
def frequentist_nsEVD(initial_params: Union[List[float], np.ndarray],
                      max_retries: int = 10) -> tuple[np.ndarray, float]
```

Estimate non-stationary EVD parameters via MLE with retries.

##### Parameters

**initial_params** : array-like Initial guess for parameters.

**max_retries** : int Number of retry attempts with modified initial guess.

##### Returns

**params** : array-like Estimated parameters.

<a id="nsEVD.evd_model.NonStationaryEVD.ns_EVDrvs"></a>

## ns_EVDrvs

``` python
@staticmethod
def ns_EVDrvs(dist: rv_continuous, params: Union[List[float], np.ndarray],
              cov: np.ndarray, config: List[int], size: int) -> np.ndarray
```

Generate non-stationary GEV or GPD random samples.

##### Parameters

**dist_name** : rv_continuous SciPy continuous distribution object (e.g., genextreme or genpareto).

**params** : list Flattened parameter list according to config.

**cov** : np.ndarray Covariate matrix, shape (n_covariates, n_samples).

**config** : list of int Non-stationarity config [loc, scale, shape]. size : int Number of random samples to generate.

##### Returns

**np.ndarray** Generated non-stationary random variates.

<a id="nsEVD.utils"></a>

# nsEVD.utils

<a id="nsEVD.utils.neg_log_likelihood"></a>

## neg_log_likelihood

``` python
def neg_log_likelihood(params, data, dist)
```

Compute the negative log-likelihood for given parameters and distribution.

##### Parameters

**params** : list or np.ndarray Parameters [loc, scale, shape] for the distribution.

**data** : array-like Observed data points. dist : scipy.stats distribution object Distribution object (e.g., genpareto or genextreme).

##### Returns

**float** Negative log-likelihood. Returns np.inf if parameters are invalid or evaluation fails.

<a id="nsEVD.utils.neg_log_likelihood_ns"></a>

## neg_log_likelihood_ns

``` python
def neg_log_likelihood_ns(params: Union[List[float],
                                        np.ndarray], data: Union[List[float],
                                                                 np.ndarray],
                          cov: Union[List[List[float]], np.ndarray],
                          config: List[int], dist: rv_continuous) -> float
```

Calculate the negative log-likelihood of the non-stationary extreme value distribution.

##### Parameters

**params** : np.ndarray Parameter vector ordered according to the config.

**data** : list or np.ndarray Observed extreme values (e.g., annual maxima).

**cov** : list of lists or np.ndarray Covariate matrix with shape (n_covariates, n_samples).

**config** : list of int Non-stationarity configuration [location, scale, shape], where 0 = stationary, \>=1 = number of covariates for non-stationary.

**dist** : rv_continuous SciPy continuous distribution object (e.g., genextreme or genpareto).

##### Returns

**float** Negative log-likelihood value. Returns np.inf if invalid parameters.

<a id="nsEVD.utils.EVD_parsViaMLE"></a>

## EVD_parsViaMLE

``` python
def EVD_parsViaMLE(data, dist, verbose=False)
```

Estimate EVD (GEV or GPD) parameters via MLE.

##### Parameters

**data** : array-like Observed data. dist : scipy.stats distribution object genextreme or genpareto distribution.

##### Returns

**np.ndarray** Estimated parameters [xi (shape), mu (location), sigma (scale)].

###### Raises

ValueError If optimization fails.

<a id="nsEVD.utils.comb"></a>

## comb

``` python
def comb(n, k)
```

Compute the binomial coefficient "n choose k".

##### Parameters

**n** : int Total number of items. k : int Number of items to choose.

##### Returns

**float** The binomial coefficient C(n, k).

<a id="nsEVD.utils.l_moments"></a>

## l_moments

``` python
def l_moments(data)
```

Compute L-moments from the given data sample.

##### Parameters

**data** : array-like Sample data array.

##### Returns

**np.ndarray,** Array containing [n, mean, L1, L2, T3, T4], where;

-   n: sample size

-   mean: sample mean

-   L1, L2: first and second L-moments

-   T3, T4: L-skewness and L-kurtosis

<a id="nsEVD.utils.GPD_parsViaLM"></a>

## GPD_parsViaLM

``` python
def GPD_parsViaLM(arr)
```

Estimate Generalized Pareto Distribution (GPD) parameters using L-moments, based on the method by Hosking and Wallis (1987)

##### Parameters

**arr** : array-like Observed data sample.

##### Returns

**np.ndarray** A NumPy array of size 3 containing the estimated GPD parameters: [shape, location, scale].

<a id="nsEVD.utils.GEV_parsViaLM"></a>

## GEV_parsViaLM

``` python
def GEV_parsViaLM(arr)
```

Estimate Generalized Extreme Value (GEV) parameters using L-moments, based on the method by Hosking and Wallis (1987)

##### Parameters

**arr** : array-like Observed data sample.

##### Returns

**np.ndarray** A NumPy array of size 3 containing the estimated GEV parameters: [shape, location, scale].

<a id="nsEVD.utils.plot_trace"></a>

## plot_trace

``` python
def plot_trace(samples, config, fig_size=None, param_names_override=None)
```

Plot MCMC trace plots for each parameter based on config.

##### Parameters

**samples** : np.ndarray MCMC samples of shape (n_iterations, n_parameters)

**config** : list of int Non-stationarity config [loc, scale, shape]

**fig_size** : tuple Optional figure size.

**param_names_override** : list of str Optional custom names for parameters.

<a id="nsEVD.utils.plot_posterior"></a>

## plot_posterior

``` python
def plot_posterior(samples, config, fig_size=None, param_names_override=None)
```

Plot histograms with density curves for each parameter based on config.

##### Parameters

**samples** : np.ndarray MCMC samples of shape (n_iterations, n_parameters)

**config** : list of int Non-stationarity config [loc, scale, shape]

**fig_size** : tuple, optional Optional figure size (width, height). Default is based on number of parameters.

**param_names_override** : list of str, optional Custom parameter names to override default naming from config.

<a id="nsEVD.utils.bayesian_metrics"></a>

## bayesian_metrics

``` python
def bayesian_metrics(samples, data, cov, config, dist)
```

Compute Bayesian model selection criteria (DIC, AIC, BIC) from posterior samples.

This function evaluates the model's performance using Deviance Information Criterion (DIC), Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC) based on the log-likelihoods computed from the posterior samples.

##### Parameters

**samples** : ndarray of shape (n_samples, n_params) Posterior samples of model parameters obtained from MCMC or another Bayesian method.

**data** : array-like Observed data used to compute the likelihood.

**cov** : array-like or None Covariates used in the non-stationary model, if applicable.

**config** : dict Configuration settings for the likelihood computation, e.g., fixed parameters, link functions.

**dist** : str or callable Distribution type used for modeling the data (e.g., "gev", "gumbel"), passed to the likelihood function.

##### Returns

**dict** A dictionary containing the computed values of DIC, AIC, and BIC.

###### Notes

-   DIC is computed using the effective number of parameters (pD = 2 \* (max_ll - mean_ll)).

-   AIC and BIC are computed using the maximum log-likelihood and number of parameters.

-   The log-likelihood is computed using the negative log-likelihood function for each sample.

<a id="nsEVD.utils.gelman_rubin"></a>

## gelman_rubin

``` python
def gelman_rubin(chains: List[np.ndarray])
```

Compute the Gelman-Rubin R-hat statistic for each parameter.

##### Parameters

**chains** : list of np.ndarray List of chains (arrays of shape [n_samples, n_params])

##### Returns

**np.ndarray**

R-hat values for each parameter

<a id="nsEVD._version"></a>

## nsEVD.\_version

Returns the current version of the software.
