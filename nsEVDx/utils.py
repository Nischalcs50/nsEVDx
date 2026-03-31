import warnings
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import rv_continuous


# Utility functions
def neg_log_likelihood(params, data, dist):
    """
    Compute the negative log-likelihood of data for given parameters of a
    stationary distribution.

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
    Safe_INF = 10e25
    # Ensure parameters are within valid bounds
    if scale <= 0:  # Scale parameter must be positive
        return Safe_INF

    # Calculate the log-likelihood safely
    try:
        pdf_values = dist.pdf(data, c=shape, loc=loc, scale=scale)
        if np.any(pdf_values <= 0):
            return Safe_INF
        # Avoid log(0) by replacing zeros with a very small value
        # catches float underflow
        pdf_values = np.clip(pdf_values, a_min=1e-300, a_max=None)
        log_likelihood = np.sum(np.log(pdf_values))
        return -log_likelihood
    except Exception:
        return Safe_INF  # Return a large value to avoid invalid parameter sets


def neg_log_likelihood_ns(
    params: Union[List[float], np.ndarray],
    data: Union[List[float], np.ndarray],
    cov: Union[List[List[float]], np.ndarray],
    config: List[int],
    dist: str,  # type hint for scipy cont. distribution objects
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
    Safe_INF = 10e25
    cov = np.asarray(cov)
    cov = np.atleast_2d(cov)
    idx = 0
    # Location: linear relationship with covariates
    if config[0] >= 1:
        n_cov_ = int(config[0])
        B = params[idx : idx + n_cov_ + 1]  # B0 + B1*x1 + B2*x2 + ...
        idx += n_cov_ + 1
        mu = B[0] + B[1:] @ cov[0:n_cov_, :]
    else:
        mu = np.full_like(data, fill_value=params[idx])
        idx += 1
    # Scale: exponential relationship with covariates
    if config[1] >= 1:
        n_cov_ = int(config[1])
        A = params[idx : idx + n_cov_ + 1]
        idx += n_cov_ + 1
        sigma = np.exp(A[0] + A[1:] @ cov[0:n_cov_, :])
    else:
        sigma = np.full_like(data, fill_value=params[idx])
        idx += 1
    # Shape: linear relationship with covariates
    if config[2] >= 1:
        n_cov_ = int(config[2])
        K = params[idx : idx + n_cov_ + 1]
        xi = K[0] + K[1:] @ cov[0:n_cov_, :]
    else:
        xi = np.full_like(data, fill_value=params[idx])
    # Ensure parameters are valid
    if np.any(sigma <= 0):
        return Safe_INF
    
    # Standardized variates
    z = (data - mu) / sigma
    
    if dist.lower() in ["gev", "genextreme"]:
        v = 1.0 - xi * z
        if np.any(v <= 0): return Safe_INF
        
        # Gumbel Limit Handling (xi -> 0)
        eps = 1e-6
        log_v = np.log(v)
        log_lik = np.where(
            np.abs(xi) < eps,
            -np.log(sigma) - z - np.exp(-z),  # Gumbel/Extreme Type I
            -np.log(sigma) + (1.0/xi - 1.0) * log_v - v**(1.0/xi)
        )
        
    elif dist.lower() in ["gpd", "genpareto"]:
        w = 1.0 + xi * z
        if np.any(w <= 0): return Safe_INF
        
        eps = 1e-6
        log_w = np.log(w)
        log_lik = np.where(
            np.abs(xi) < eps,
            -np.log(sigma) - z,  # Exponential Distribution limit
            -np.log(sigma) - (1.0 + 1.0/xi) * log_w
        )
    else:
        raise ValueError(f"Unsupported dist_name: {dist}")

    nll = -np.sum(log_lik)
    return nll if np.isfinite(nll) else Safe_INF


def _grad_nll_gev(
    params: np.ndarray,
    data:   np.ndarray,
    cov:    np.ndarray,
    config: List[int],
) -> np.ndarray:
    """
    Analytical gradient of the GEV negative log-likelihood  ∂NLL/∂θ.

    We use these expressions to compute:
        v_t = 1 − xi·z_t,   z_t = (x_t − μ_t) / σ_t
        log p_t = −log σ_t + (1/xi − 1)·log v_t − v_t^(1/xi)

    Scale link: log σ_t = a_0 + Σ_k a_k·cov_{k,t}   when config[1] ≥ 1.

    Returns ∂NLL/∂θ.  Returns NaN array if any v_t ≤ 0 (outside support).
    """
    cov = np.atleast_2d(cov)
    n   = len(data)
    grad = np.zeros_like(params)
    idx = 0

    # location
    if config[0] >= 1:
        nc = int(config[0])
        B  = params[idx: idx + nc + 1]
        idx_mu = idx
        idx += nc + 1
        mu = B[0] + B[1:] @ cov[:nc, :]
    else:
        mu  = np.full(n, params[idx]); idx_mu = idx; idx += 1

    # scale
    if config[1] >= 1:
        nc    = int(config[1])
        A     = params[idx: idx + nc + 1]
        idx_sig = idx
        idx += nc + 1
        sigma = np.exp(A[0] + A[1:] @ cov[:nc, :])
    else:
        sigma = np.full(n, params[idx]); idx_sig = idx; idx += 1

    # shape 
    if config[2] >= 1:
        nc = int(config[2])
        K  = params[idx: idx + nc + 1]
        idx_xi = idx
        xi = K[0] + K[1:] @ cov[:nc, :]
    else:
        xi = np.full(n, params[idx]); idx_xi = idx

    z = (data - mu) / sigma
    v = 1.0 - xi * z                  
    if np.any(v <= 0):
        return np.full(len(params), np.nan)

    # Gumbel limit: |xi| < eps, approximating to avoid 1/xi singularity
    eps      = 1e-6
    xi_safe  = np.where(np.abs(xi) < eps, np.sign(xi + 1e-30) * eps, xi)
    small_xi = np.abs(xi) < eps
    inv_v = 1.0 / v
    
    # Partial derivatives of LL
    # ---dlogL/dv--------------
    dL_dv = np.where(
        small_xi,
        -inv_v - 1.0,                                    # Gumbel limit
        (1.0/xi_safe - 1.0)*inv_v - (1.0/xi_safe)*v**(1.0/xi_safe - 1.0)
    )

    #---- Location gradient (dv/dmu = xi/sigma) 
    dL_dmu = dL_dv * (xi / sigma)
    
    #---- Scale gradient (dv/dsigma = xi*z/sigma)
    # dL_dsig_base = -1.0/sigma + dL_dv * (xi * z / sigma)
    # because of exponential regression for the scale
    # chain rule: d(log σ) = dsigma/sigma
    dL_da = -1.0 + dL_dv * (xi * z)  
    
    #---- Shape gradient (dv/dxi = -z)
    log_v  = np.log(v)
    dL_dxi = np.where(
        small_xi,
        0.5*z**2 - z,                           # Gumbel limit (approx)
        -log_v/xi_safe**2
        + (1.0/xi_safe - 1.0)*(-z)/v
        + v**(1.0/xi_safe)*log_v/xi_safe**2
        - v**(1.0/xi_safe - 1.0)*(-z)/xi_safe
    )
    
    #----Vectorized assembling
    # (Negative Log-Likelihood Gradient -> Multiply by -1)
    # Location components
    grad[idx_mu] = -np.sum(dL_dmu)
    if config[0] >= 1:
        nc = int(config[0])
        grad[idx_mu + 1 : idx_mu + 1 + nc] = -(dL_dmu @ cov[:nc, :].T)

    # Scale components
    grad[idx_sig] = -np.sum(dL_da)
    if config[1] >= 1:
        nc = int(config[1])
        grad[idx_sig + 1 : idx_sig + 1 + nc] = -(dL_da @ cov[:nc, :].T)

    # Shape components
    grad[idx_xi] = -np.sum(dL_dxi)
    if config[2] >= 1:
        nc = int(config[2])
        grad[idx_xi + 1 : idx_xi + 1 + nc] = -(dL_dxi @ cov[:nc, :].T)

    return grad

    
def _grad_nll_gpd(
    params: np.ndarray,
    data:   np.ndarray,
    cov:    np.ndarray,
    config: List[int],
) -> np.ndarray:
    """
    Analytical gradient of the GPD negative log-likelihood  ∂NLL/∂θ.

    We use theseexpressions to compute:
        w_t = 1 + xi·z_t,   z_t = (x_t − μ_t) / σ_t   (PLUS sign)
        log p_t = −log σ_t − (1 + 1/xi)·log w_t

    Returns ∂NLL/∂θ.  Returns NaN array if any w_t ≤ 0.
    """
    cov = np.atleast_2d(cov)
    n   = len(data)
    idx = 0
    grad = np.zeros_like(params)
    
    # Location
    if config[0] >= 1:
        nc = int(config[0])
        B = params[idx : idx + nc + 1]; idx_mu = idx; idx += nc + 1
        mu = B[0] + B[1:] @ cov[:nc, :]
    else:
        mu = np.full(n, params[idx]); idx_mu = idx; idx += 1

    # Scale (with Log-Link)
    if config[1] >= 1:
        nc = int(config[1])
        A = params[idx : idx + nc + 1]; idx_sig = idx; idx += nc + 1
        sigma = np.exp(A[0] + A[1:] @ cov[:nc, :])
    else:
        sigma = np.full(n, params[idx]); idx_sig = idx; idx += 1

    # Shape
    if config[2] >= 1:
        nc = int(config[2])
        K = params[idx : idx + nc + 1]; idx_xi = idx
        xi = K[0] + K[1:] @ cov[:nc, :]
    else:
        xi = np.full(n, params[idx]); idx_xi = idx

    z = (data - mu) / sigma
    w = 1.0 + xi * z                    # GPD
    if np.any(w <= 0):
        return np.full(len(params), np.nan)

    eps      = 1e-6
    xi_safe  = np.where(np.abs(xi) < eps, np.sign(xi + 1e-30) * eps, xi)
    small_xi = np.abs(xi) < eps
    inv_w = 1.0 / w
    
    # Partial derivatives of LL
    #----dlogL/dw = -(1 + 1/xi) / w
    dL_dw = np.where(small_xi, -inv_w, -(1.0 + 1.0/xi_safe)* inv_w)
    
    #----Location gradient dw/dmu = -xi/sigma
    dL_dmu = dL_dw * (-xi / sigma)
    
    #----Scale gradient dw/dsigma = -xi*z/sigma
    # dL_dsig_base = -1.0/sigma + dL_dw * (-xi * z / sigma)
    # dL/dsigma: includes log-link chain rule (dL/d_sigma * sigma)
    # dL_dsig_base * sigma = 
    #  = (-1/sig + dL/dw * dw/dsig) * sig = -1 + dL/dw * (-xi * z)
    dL_da = -1.0 + dL_dw * (-xi * z)
    
    #----Shape gradient dL/dxi
    log_w = np.log(w)
    dL_dxi = np.where(
        small_xi,
        -0.5 * z**2,
        (1.0 / xi_safe**2) * log_w - (1.0 + 1.0/xi_safe) * z * inv_w
    )
    
    #----Vectorized assembling
    # (Negative Log-Likelihood Gradient -> Multiply by -1)
    # Location Components
    grad[idx_mu] = -np.sum(dL_dmu)
    if config[0] >= 1:
        nc = int(config[0])
        grad[idx_mu+1 : idx_mu+1+nc] = -(dL_dmu @ cov[:nc, :].T)

    # Scale Components
    grad[idx_sig] = -np.sum(dL_da)
    if config[1] >= 1:
        nc = int(config[1])
        grad[idx_sig+1 : idx_sig+1+nc] = -(dL_da @ cov[:nc, :].T)

    # Shape Components
    grad[idx_xi] = -np.sum(dL_dxi)
    if config[2] >= 1:
        nc = int(config[2])
        grad[idx_xi+1 : idx_xi+1+nc] = -(dL_dxi @ cov[:nc, :].T)

    return grad
    
            
def _total_log_prior(params: np.ndarray, prior_specs: list) -> float:
    """
    Compute the total log-prior probability of the parameter vector.

    This method calculates the sum of log-prior probabilities for each
    parameter based on the specified prior distributions in
    prior_specs.
    
    Parameters
    ----------
    params : array-like
        A 1D array of parameter values corresponding to the linear or
        exponential models for location, scale, and shape parameters.
        The number and order of parameters must match the configuration.
    prior_specs : list of tuples
        A list where each entry is a tuple: (prior_type, hyperparameter_dict).
        Example: [('normal', {'loc': 50, 'scale': 10}), ('uniform', {'loc': -0.5, 'scale': 1.0})].
        If None, the function returns 0.0 (equivalent to an improper flat prior).

    Returns
    -------
    float
        The total log-prior probability of the parameter vector.
        Returns -np.inf if any prior evaluates to a non-finite value.

    Notes
    -----
    - Supports 'normal', 'uniform', and 'halfnormal' priors.
    - If no `prior_specs` are provided (i.e., None), returns 0.0 (flat
                                                                  prior).
    - Prior specification format:
        prior_specs = [('normal', {'loc': 0, 'scale': 10}), ...]
    """
    if prior_specs is None:
        return 0.0

    logp = 0.0
    _LOG_SQRT_2PI = 0.9189385332046727
    with np.errstate(over='ignore', invalid='ignore'):
        for i, (ptype, kw) in enumerate(prior_specs):
            val = params[i]
            if ptype == "normal":
                loc = kw["loc"]
                scale = kw["scale"]
                logp += -0.5 * ((val - loc) / scale) ** 2 - np.log(scale)- _LOG_SQRT_2PI
            
            elif ptype == "uniform":
                lo = kw["loc"]
                hi = lo + kw["scale"]
                if not (lo <= val <= hi):
                    return -np.inf
                logp -= np.log(kw['scale'])
                
            elif ptype == "halfnormal":
                scale = kw["scale"]
                if v < 0:
                    return -np.inf
                logp += (-0.5 * (val / scale) ** 2 - np.log(scale)
                         - _LOG_SQRT_2PI + 0.69314718056)
            else:
                return -np.inf   # unknown type — safe fallback
            
            # Sanity Check
            if not np.isfinite(logp):
                return -np.inf
    return logp
    

def _grad_total_log_prior(params: np.ndarray,
                    prior_specs: list,
                    h: float = 1e-4
                    ) -> np.ndarray :
    """
    Compute gradient of log-prior (∂log π(θ)/∂θ) via central difference (h=1e-4).
    
    Parameters:
    -----------
        params      : 1D array of parameter values.
        prior_specs : List of tuples matching params order.
        h           : Step size for finite difference.

    Returns:
    --------
        1D array of gradients for each parameter.
    
    """
    grad = np.zeros(len(params))
    # Initial check: If we are already in an impossible spot, gradient is zero
    lp0 = _total_log_prior(params,prior_specs) 
    if not np.isfinite(lp0):
        return grad

    # Loop through each parameter
    for i in range(len(params)):
        step = np.zeros(len(params))
        step[i] = h
        lp_plus  = _total_log_prior(params + step, prior_specs)
        lp_minus = _total_log_prior(params - step, prior_specs)
        
        #  Updating the gradient if only, lp at both sides are valid
        if np.isfinite(lp_plus) and np.isfinite(lp_minus):
            grad[i] = (lp_plus - lp_minus) / (2.0 * h)
        else:
            grad[i] = 0.0
            
    return grad


def EVD_parsViaMLE(data, dist, verbose=False):
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
    if dist.name.lower() in ["genpareto", "gpd"]:
        mu_guess = np.min(X)
        initial_params = [mu_guess, np.std(X - mu_guess), 0.01]
    elif dist.name.lower() in ["genextreme", "gev"]:
        initial_params = [np.percentile(X, 40), np.std(X), 0.01]
    else:
        raise ValueError("Unsupported distribution. Use GEV or GPD.")

    # Minimize the negative log-likelihood
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(X, dist),
        method="Nelder-Mead",
        options={"disp": verbose, "maxiter": 2500},  # Enable verbose output for
        # debugging
    )

    if result.success:
        mu_hat, sigma_hat, xi_hat = result.x
        if verbose:
            print(f"Estimated parameters: loc={mu_hat}, sigma={sigma_hat}, xi={xi_hat}")
        return np.array([xi_hat, mu_hat, sigma_hat])
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def _comb(n, k):
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


def l_moments(data):
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
    n_moments = 4
    n = len(data)
    b = np.zeros(n_moments)
    data = np.sort(data)
    mu = data.mean()
    data = data / mu

    for r in range(0, n_moments):
        coef = 1 / (n * _comb(n - 1, r))
        summ = 0
        for j in range(r + 1, n + 1):
            aux = data[j - 1] * comb(j - 1, r)  # here data[j-1] because index for
            # data starts from 0
            summ += aux
        b[r] = coef * summ

    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    t3 = (6 * b[2] - 6 * b[1] + b[0]) / l2
    t4 = (20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]) / l2
    result = [n, mu, l1, l2, t3, t4]
    result = np.array([np.round(x, 3) for x in result])
    return result


def GPD_parsViaLM(arr):
    """
    Estimate Generalized Pareto Distribution (GPD) parameters using L-moments
    based on the formulations given in Hosking and Wallis (1987)

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
    pr = np.zeros(9)
    pr[0] = len(arr)
    pr[1] = np.round(arr.mean(), 3)
    l1, l2, t3, t4 = np.round(l_moments(arr / pr[1])[2:], 4)
    k = (1 - 3 * t3) / (1 + t3)  # shape
    a = (1 + k) * (2 + k) * l2  # scale
    x = l1 - (a / (1 + k))  # location
    pr[2] = l1
    pr[3] = l2
    pr[4] = t3
    pr[5] = t4
    pr[6] = x * pr[1]
    pr[7] = a * pr[1]
    pr[8] = -1 * k  # Because the formulation used here assumes negative shape
    # shape parameter compared to the GPD formulation in scipy
    return np.array([pr[8], pr[6], pr[7]])


def GEV_parsViaLM(arr):
    """
    Estimate Generalized Extreme Value (GEV) parameters using L-moments
    based on the formulations given in Hosking and Wallis (1987)

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
    k = 7.8590 * c + 2.9554 * (c**2)
    a = l2 * k / ((1 - 2 ** (-k)) * gamma(1 + k))
    x = l1 - (a * (1 - gamma(1 + k)) / k)

    pr[2] = l1
    pr[3] = l2
    pr[4] = t3
    pr[5] = t4
    pr[6] = x * pr[1]
    pr[7] = a * pr[1]
    pr[8] = k
    return np.array([pr[8], pr[6], pr[7]])


def _build_param_names(config, override=None):
    if override is not None:
        return override
    names = []
    if config[0] == 0:
        names.append("loc")
    else:
        names.append("B0")
        names.extend([f"B{i+1}" for i in range(config[0])])
    if config[1] == 0:
        names.append("scale")
    else:
        names.append("a0")
        names.extend([f"a{i+1}" for i in range(config[1])])
    if config[2] == 0:
        names.append("shape")
    else:
        names.append("k0")
        names.extend([f"k{i+1}" for i in range(config[2])])
    return names


def plot_trace(chains, config, fig_size=None, param_names_override=None):
    """
    Plot MCMC trace plots for each parameter based on config. vector

    Parameters
    ----------
    chains : np.ndarray
        MCMC samples of shape (n_iterations, n_parameters)
    config : list of int
        Non-stationarity config [loc, scale, shape]
    fig_size : tuple
        Optional figure size.
    param_names_override : list of str
        Optional custom names for parameters.
    """
    # 1. Ensure chains is a list of arrays (even if only 1 chain was run)
    if isinstance(chains, np.ndarray):
        chains = [chains]

    # Generate default names based on config
    param_names = _build_param_names(config, param_names_override)
    n_params = len(param_names)
    n_chains = len(chains)

    if fig_size is None:
        fig_size = (10, n_params * 2)

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(n_params, 1, figsize=fig_size,
                             sharex=True)
    for i in range(n_params):
        # Plot each chain separately
        for c in range(n_chains):
            axes[i].plot(chains[c][:, i],
                         label=f"{c+1}",
                         color=colors[c % len(colors)],
                         linewidth=0.7,
                         alpha=0.8)
        axes[i].set_ylabel(param_names[i], fontsize=12)
        axes[i].tick_params(labelsize=12)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].grid(True)
        axes[-1].set_xlabel("Iteration", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,
               loc='center left',
               title ="Chain",
                    fontsize=12,
                    ncol=1,
                    bbox_to_anchor=(0.96, 0.5),
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='black'
                )


    plt.tight_layout()
    plt.show()


def plot_posterior(chains, config, fig_size=None, param_names_override=None):
    """
    Plot histograms with density curves for each parameter based on config.
    vector

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
    if isinstance(chains, list):
        samples = np.vstack(chains)
    else:
        samples = chains

    # Generate parameter names based on config if no override provided
    param_names = _build_param_names(config, param_names_override)
    n_params = len(param_names)

    cols = 2 if n_params >= 4 else 1
    rows = int(np.ceil(n_params / cols))

    if fig_size is None:
        fig_size = (5 * cols, 3 * rows)

    plt.figure(figsize=fig_size)
    for i in range(n_params):
        plt.subplot(rows, cols, i + 1)
        sns.histplot(samples[:, i], kde=True, color="#5DADE2",
                     bins=30, stat="density", alpha=0.6)
        plt.title(f"Posterior: {param_names[i]}", fontsize=13, fontweight='bold')
        plt.ylabel("Density", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(fontsize=10, rotation=25)
        plt.yticks(fontsize=10)
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
    if np.sum(config) == 0:
        # Stationary case
        log_likelihoods = np.array(
            [
                -neg_log_likelihood(p, data, dist)  # stationary function
                for p in samples
            ]
        )
    else:
        # Non-stationary case
        log_likelihoods = np.array(
            [
                -neg_log_likelihood_ns(p, data, cov, config, dist)  # non-stationary
                for p in samples
            ]
        )
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
    n = chains[0].shape[0]
    chains = np.array(chains)  # shape (m, n, p)
    # where, m = len(chains) and p = chains.shape[2]

    # Mean per chain and overall mean
    mean_per_chain = chains.mean(axis=1)
    # Between-chain variance
    B = n * np.var(mean_per_chain, axis=0, ddof=1)
    # Within-chain variance
    W = np.mean(np.var(chains, axis=1, ddof=1), axis=0)
    # Estimate of marginal posterior variance
    var_hat = ((n - 1) / n) * W + (1 / n) * B

    # R-hat
    R_hat = np.sqrt(var_hat / W)
    return R_hat
