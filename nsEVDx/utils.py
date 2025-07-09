import numpy as np
from typing import List, Tuple, Union, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import genextreme, norm, uniform, genpareto
from scipy.stats import rv_continuous
from scipy.optimize import minimize


# Utility functions
def neg_log_likelihood(params, data,dist):
    """
    Compute the negative log-likelihood for given parameters and distribution.
    
    Parameters
    ----------
    params : list or np.ndarray
        Parameters [loc, scale, shape] for the distribution.
    data : array-like
        Observed data points.
    dist : scipy.stats distribution object
        Distribution object (e.g., genpareto or genextreme).
    
    Returns
    -------
    float
        Negative log-likelihood. Returns np.inf if parameters are invalid or
        evaluation fails.
    """
    loc, scale, shape = params

    # Ensure parameters are within valid bounds
    if scale <= 0:  # Scale parameter must be positive
        return np.inf

    # Calculate the log-likelihood safely
    try:
        pdf_values = dist.pdf(data, c=shape, loc=loc, scale=scale)
        # Avoid log(0) by replacing zeros with a very small value
        pdf_values = np.clip(pdf_values, a_min=1e-10, a_max=None)
        log_likelihood = np.sum(np.log(pdf_values))
        return -log_likelihood
    except Exception as e:
        return np.inf  # Return a large value to avoid invalid parameter sets

 
def neg_log_likelihood_ns(
    params: Union[List[float], np.ndarray],
    data: Union[List[float], np.ndarray],
    cov: Union[List[List[float]], np.ndarray],
    config: List[int],
    dist: rv_continuous  # type hint for scipy cont. distribution objects 
                         # like genpareto/genextreme
) -> float:
    """
    Calculate the negative log-likelihood of the non-stationary extreme 
    value distribution.

    Parameters
    ----------
    params : np.ndarray
        Parameter vector ordered according to the config.
    data : list or np.ndarray
        Observed extreme values (e.g., annual maxima).
    cov : list of lists or np.ndarray
        Covariate matrix with shape (n_covariates, n_samples).
    config : list of int
        Non-stationarity configuration [location, scale, shape], where
        0 = stationary, >=1 = number of covariates for non-stationary.
    dist : rv_continuous
        SciPy continuous distribution object (e.g., genextreme or 
                                              genpareto).

    Returns
    -------
    float
        Negative log-likelihood value. Returns np.inf if invalid 
        parameters.
    """
    cov = np.asarray(cov)
    cov  =  np.atleast_2d(cov)
    if cov.ndim > 1:
        n_cov = cov.shape[0]
    else :
        n_cov = 1
    idx = 0
    # Location: linear relationship with covariates
    if config[0] >= 1:
        n_cov_ = int(config[0])
        B = params[idx:idx + n_cov_ + 1]  # B0 + B1*x1 + B2*x2 + ...
        idx += n_cov_ + 1
        mu = B[0] + B[1:] @ cov[0:n_cov_,:]
    else:
        mu = np.full_like(data, fill_value=params[idx])
        idx += 1
    # Scale: exponential relationship with covariates
    if config[1] >= 1:
        n_cov_ = int(config[1])
        A = params[idx:idx + n_cov_+1]
        idx += n_cov_+1
        sigma = np.exp(A[0] + A[1:] @ cov[0:n_cov_,:])
    else:
        sigma = np.full_like(data, fill_value=params[idx])
        idx += 1
    # Shape: linear relationship with covariates
    if config[2] >= 1:
        n_cov_ = int(config[2])
        K = params[idx:idx + n_cov_+1]
        xi = K[0] + K[1:] @ cov[0:n_cov_,:]
    else:
        xi = np.full_like(data, fill_value=params[idx])
    # Ensure parameters are valid
    if np.any(sigma <= 0):
        return np.inf
    try:
        # Evaluate PDF and compute log-likelihood
        pdf_values = dist.pdf(data, c=xi, loc=mu, scale=sigma)
        pdf_values = np.clip(pdf_values, a_min=1e-10, a_max=None)
        log_likelihood = np.sum(np.log(pdf_values))
        return -log_likelihood
    except Exception:
        return np.inf
    

def EVD_parsViaMLE(data,dist, verbose=False):
    """
    Estimate EVD (GEV or GPD) parameters via MLE.

    Parameters
    ----------
    data : array-like
        Observed data.
    dist : scipy.stats distribution object
        genextreme or genpareto distribution.

    Returns
    -------
    np.ndarray
        Estimated parameters [xi (shape), mu (location), sigma (scale)].

    Raises
    ------
    ValueError
        If optimization fails.
    """
    X = data
      # Initial guesses for mu, sigma, xi
    if  dist.name.lower() in ['genpareto','gpd']:
        mu_guess = np.min(X)
        initial_params = [mu_guess, np.std(X-mu_guess), 0.01] 
    elif dist.name.lower() in ['genextreme', 'gev']:
        initial_params = [np.percentile(X,40), np.std(X), 0.01]
    else:
        raise ValueError("Unsupported distribution. Use GEV or GPD.")        
      
    # Minimize the negative log-likelihood
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(X,dist),
        method='Nelder-Mead',
        options={'disp': verbose,'maxiter':2500}  # Enable verbose output for 
                                               # debugging
    )

    if result.success:
        mu_hat, sigma_hat, xi_hat = result.x
        if verbose:
            print(
                f"Estimated parameters: loc={mu_hat}, sigma={sigma_hat},"
                f" xi={xi_hat}")
        return np.array([xi_hat,mu_hat,sigma_hat])
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def comb(n, k):
    """
    Compute the binomial coefficient "n choose k".

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items to choose.

    Returns
    -------
    float
        The binomial coefficient C(n, k).
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    numerator = 1
    for i in range(n, n - k, -1):
        numerator *= i
    denominator = 1
    for i in range(1, k + 1):
        denominator *= i
    return numerator // denominator


def l_moments(data ):
    """
    Compute L-moments from the given data sample.

    Parameters
    ----------
    data : array-like
        Sample data array.
        
    Returns
    -------
    np.ndarray
        Array containing [n, mean, L1, L2, T3, T4], where
        - n: sample size
        - mean: sample mean
        - L1, L2: first and second L-moments
        - T3, T4: L-skewness and L-kurtosis
    """
    n_moments=4
    n = len(data)
    b = np.zeros(n_moments)
    data = np.sort(data)
    mu = data.mean()
    data = data/mu
    
    for r in range(0, n_moments):
        coef = 1/(n*comb(n-1,r))
        summ = 0
        for j in range(r+1,n+1):
            aux = data[j-1]*comb(j-1,r)# here data[j-1] because index for 
            # data starts from 0
            summ += aux
        b[r] = coef*summ

    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    t3 = (6 * b[2] - 6 * b[1] + b[0]) / l2
    t4 = (20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]) / l2
    result = [n,mu,l1, l2, t3, t4] 
    result = np.array([np.round(x,3) for x in result])
    return result 


def GPD_parsViaLM(arr):
    """
    Estimate Generalized Pareto Distribution (GPD) parameters using L-moments.
    baseed on Hosking and Wallis (1987)
    
    Parameters
    ----------
    arr : array-like
        Observed data sample. 
    
    Returns
    -------
    np.ndarray
        A NumPy array of size 3 containing the estimated GPD parameters:
        [shape, location, scale].
    """
    # compute pars by normalising first
    # i.e., useful for index flood procedure
    arr = np.sort(arr)
    pr  = np.zeros(9)
    pr[0] = len(arr)
    pr[1] = np.round(arr.mean(),3)
    l1,l2,t3,t4 = np.round(l_moments(arr/pr[1])[2:],4)
    k = (1-3*t3)/(1 + t3) # shape
    a = (1+k)*(2+k)*l2  # scale
    x =  l1-(a/(1+k))  # location
    pr[2] = l1
    pr[3] = l2
    pr[4] = t3
    pr[5] = t4
    pr[6] = x*pr[1]
    pr[7] = a*pr[1]
    pr[8] = -1*k # Because the formulation used here assumes negative shape
                 # shape parameter compared to the GPD formulation in scipy
    return(np.array([pr[8],pr[6],pr[7]]))

        
def GEV_parsViaLM(arr):
    """
    Estimate Generalized Extreme Value (GEV) parameters using L-moments.
    baseed on Hosking and Wallis (1987)
    
    Parameters
    ----------
    arr : array-like
        Observed data sample. 
    
    Returns
    -------
    np.ndarray
        A NumPy array of size 3 containing the estimated GEV parameters:
        [shape, location, scale].
        
    """
    arr = np.sort(arr)
    pr = np.zeros(9)
    pr[0] = len(arr)
    pr[1] = np.round(arr.mean(), 3)

    l1, l2, t3, t4 = np.round(l_moments(arr / pr[1])[2:], 4)

    c = (2 / (3 + t3)) - (np.log(2) / np.log(3))
    k = 7.8590 * c + 2.9554 * (c ** 2)
    a = l2 * k / ((1 - 2 ** (-k)) * gamma(1 + k))
    x = l1 - (a * (1 - gamma(1 + k)) / k)

    pr[2] = l1
    pr[3] = l2
    pr[4] = t3
    pr[5] = t4
    pr[6] = x*pr[1]
    pr[7] = a*pr[1]
    pr[8] = k
    return(np.array([pr[8],pr[6],pr[7]]))


def plot_trace(samples, config, fig_size=None, param_names_override=None):
    """
    Plot MCMC trace plots for each parameter based on config.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples of shape (n_iterations, n_parameters)
    config : list of int
        Non-stationarity config [loc, scale, shape]
    fig_size : tuple
        Optional figure size.
    param_names_override : list of str
        Optional custom names for parameters.
    """
    # Generate default names based on config
    if param_names_override is None:
        param_names = []
        if config[0] == 0:
            param_names.append("loc")
        else:
            param_names.append("B0")
            param_names.extend([f"B{i+1}" for i in range(config[0])])
        if config[1] == 0:
            param_names.append("scale")
        else:
            param_names.append("a0")
            param_names.extend([f"a{i+1}" for i in range(config[1])])
        if config[2] == 0:
            param_names.append("shape")
        else:
            param_names.append("k0")
            param_names.extend([f"k{i+1}" for i in range(config[2])])
    else:
        param_names = param_names_override

    n_params = len(param_names)
    if fig_size is None:
        fig_size = (10, n_params * 2)

    plt.figure(figsize=fig_size)
    for i in range(n_params):
        plt.subplot(n_params, 1, i + 1)
        plt.plot(samples[:, i], label=param_names[i])
        plt.ylabel(param_names[i], fontsize=14)
        plt.xlabel("Iteration", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def plot_posterior(samples, config, fig_size=None, param_names_override=None):
    """
    Plot histograms with density curves for each parameter based on config.

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples of shape (n_iterations, n_parameters)
    config : list of int
        Non-stationarity config [loc, scale, shape]
    fig_size : tuple, optional
        Optional figure size (width, height). Default is based on number of 
        parameters.
    param_names_override : list of str, optional
        Custom parameter names to override default naming from config.
    """
    # Generate parameter names based on config if no override provided
    if param_names_override is None:
        param_names = []
        if config[0] == 0:
            param_names.append("loc")
        else:
            param_names.append("B0")
            param_names.extend([f"B{i+1}" for i in range(config[0])])
        if config[1] == 0:
            param_names.append("scale")
        else:
            param_names.append("a0")
            param_names.extend([f"a{i+1}" for i in range(config[1])])
        if config[2] == 0:
            param_names.append("shape")
        else:
            param_names.append("k0")
            param_names.extend([f"k{i+1}" for i in range(config[2])])
    else:
        param_names = param_names_override

    n_params = len(param_names)
    if fig_size is None:
        fig_size = (10, n_params * 2)

    plt.figure(figsize=fig_size)
    for i in range(n_params):
        plt.subplot(n_params, 1, i + 1)
        sns.histplot(samples[:, i], kde=True, color='skyblue', bins=20, 
                     stat='density')
        plt.title(f'Distribution of {param_names[i]}', fontsize=14)
        plt.xlabel(param_names[i], fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def bayesian_metrics(samples, data, cov, config, dist):
    """
    Compute Bayesian model selection criteria (DIC, AIC, BIC) from posterior 
    samples.

    This function evaluates the model's performance using Deviance Information 
    Criterion (DIC), Akaike Information Criterion (AIC), and Bayesian 
    Information Criterion (BIC) based on the log-likelihoods computed from the 
    posterior samples.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n_params)
        Posterior samples of model parameters obtained from MCMC or another 
        Bayesian method.
    
    data : array-like
        Observed data used to compute the likelihood.
    
    cov : array-like or None
        Covariates used in the non-stationary model, if applicable.
    
    config : dict
        Configuration settings for the likelihood computation, e.g., fixed 
        parameters, link functions.
    
    dist : str or callable
        Distribution type used for modeling the data (e.g., "gev", "gumbel"),
        passed to the likelihood function.

    Returns
    -------
    dict
        A dictionary containing the computed values of DIC, AIC, and BIC.

    Notes
    -----
    - DIC is computed using the effective number of parameters 
        (pD = 2 * (max_ll - mean_ll)).
    - AIC and BIC are computed using the maximum log-likelihood and number of
        parameters.
    - The log-likelihood is computed using the negative log-likelihood function
        for each sample.
    """
    log_likelihoods = np.array([
        -neg_log_likelihood_ns(p, data, cov, config, dist)
        for p in samples
    ])
    mean_ll = np.mean(log_likelihoods)
    max_ll = np.max(log_likelihoods)
    pD = 2 * (max_ll - mean_ll)
    DIC = -2 * max_ll + 2 * pD
    
    n_params = samples.shape[1]
    AIC = -2 * max_ll + 2 * n_params
    BIC = -2 * max_ll + n_params * np.log(len(data))
    
    # print(f"DIC: {DIC:.2f}")
    # print(f"AIC: {AIC:.2f}")
    # print(f"BIC: {BIC:.2f}")
    return {"DIC": DIC, "AIC": AIC, "BIC": BIC}

def gelman_rubin(chains: List[np.ndarray]):
    """
    Compute the Gelman-Rubin R-hat statistic for each parameter.

    Parameters
    ----------
    chains : list of np.ndarray
        List of chains (arrays of shape [n_samples, n_params])

    Returns
    -------
    np.ndarray
        R-hat values for each parameter
    """
    m = len(chains)
    n = chains[0].shape[0]
    chains = np.array(chains)  # shape (m, n, p)
    p = chains.shape[2]

    # Mean per chain and overall mean
    mean_per_chain = chains.mean(axis=1)
    overall_mean = chains.mean(axis=(0, 1))

    # Between-chain variance
    B = n * np.var(mean_per_chain, axis=0, ddof=1)

    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)

    # Estimate of marginal posterior variance
    var_hat = ((n - 1)/n) * W + (1/n) * B

    # R-hat
    R_hat = np.sqrt(var_hat / W)
    return R_hat
