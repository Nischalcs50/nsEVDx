import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import kstest
from scipy.optimize import minimize
from scipy import stats
from scipy.special import gamma
from scipy.stats import genextreme, norm, uniform, genpareto, halfnorm
from typing import List, Tuple, Union, Optional
import seaborn as sns
from scipy.stats import rv_continuous
from .utils import (
    neg_log_likelihood,
    neg_log_likelihood_ns,
    EVD_parsViaMLE,
    comb,
    l_moments,
    GPD_parsViaLM,
    GEV_parsViaLM,
    plot_trace,
    plot_posterior,
    bayesian_metrics,
    gelman_rubin
)

# nsEVDx main model code
class NonStationaryEVD:
    def __init__(self, config, data, cov, dist, prior_specs=None,
                 bounds=None):
        """
        Instantiate a object(reffered to as 'sampler') of class 
        NonStationaryEVD.
        
        Parameters
        ----------
        
        config : list of int.
            Non-stationarity configuration for [location, scale, shape].
            For example:
                config  = [0,0,0] indicates [stationary_location, 
                                             stationary_scale ,
                                             stationary_shape].
                config = [1,0,0] indicates [locatin modeled with 1 covariate,
                                            stationary scale,
                                            stationary shape].
                config = [2,1,0] indicates [location modeled with 2 covariates,
                                            scale modeled with 1 covariate,
                                            stationary shape].
                Note: the location and shape parameter are modeled linearly,
                whereas the scale parameter is modeled exponentially 
            
        data : array-like
            Observed extremes in chronlogical order (e.g., annual maxima).
        cov : array-like
            Covariate matrix, shape (n_covariates, n_samples).
        dist : scipy.stats distribution object (genextreme or genpareto).
        prior_specs : list of tuples
            Optional prior specifications for each parameter. Required if 
            performing bayesian sampling.
            Format: [(dist_name, params_dict), ...]
            e.g., [('normal', {'loc': 0, 'scale': 10}), ('uniform', 
                                                  {'loc': 0, 'scale': 5}), ...]
        bounds : List of tuples.
                Optional bounds for each parameter, required if estimating the
                the parameters by frequentist approach
        Returns
        -------
        NonStationaryEVD,
         An instance of the NonStationaryEVD class initialized with the 
         specified configuration, data, covariates, and distribution.
        """
        self.config = config
        self.data = np.asarray(data)
        self.cov = np.atleast_2d(np.asarray(cov))
        self.dist = dist
        self.n_cov = cov.shape[0] if cov.ndim > 1 else 1
        self.prior_specs = prior_specs
        self.bounds = bounds
        assert self.data.shape[0] == self.cov.shape[1],( 
            "Mismatch between number of samples in data and covariates"
        )
        expected_param_count = sum(config) + 3  # or logic based on config
        if prior_specs and len(prior_specs) != expected_param_count:
            raise ValueError(
                "Mismatch between config (expected parameters to estimate)"
                " and prior_specs length"
            )
        if bounds and len(bounds) != expected_param_count:
            raise ValueError(
                "Mismatch between config (expected parameters to estimate)"
                " and bounds provided"
            )
        self.descriptions = self.get_param_description(self.config, self.n_cov)
     
            
    @staticmethod
    def get_param_description(config: List[int], n_cov: int) -> List[str]:
        """
        Returns a list of strings describing each parameter's role in the 
        parameter vector, based on the provided configuration.
    
        Parameters
        ----------
        config : list of int
            Non-stationarity configuration [location, scale, shape].
        n_cov : int
            Total number of covariates available.
    
        Returns
        -------
        list of str
            Descriptions of each parameter in order.
        """
        desc = []
        idx = 0
    
        # Location parameters
        if config[0] >= 1:
            n = int(config[0])
            desc.append('B0 (location intercept)')
            for i in range(1, n + 1):
                desc.append(f'B{i} (location slope for covariate {i})')
            idx += n + 1
        else:
            desc.append('mu (stationary location)')
            idx += 1
    
        # Scale parameters
        if config[1] >= 1:
            n = int(config[1])
            desc.append('a0 (scale intercept)')
            for i in range(1, n + 1):
                desc.append(f'a{i} (scale slope for covariate {i})')
            idx += n + 1
        else:
            desc.append('sigma (scale)')
            idx += 1
    
        # Shape parameters
        if config[2] >= 1:
            n = int(config[2])
            desc.append('k0 (shape intercept)')
            for i in range(1, n + 1):
                desc.append(f'k{i} (shape slope for covariate {i})')
        else:
            desc.append('xi (shape)')
    
        return desc
    
    
    def suggest_priors(self):
        """
        Suggest default prior distributions for model parameters based on the
        current configuration and data statistics.
        
        Returns
        -------
        prior_specs : list of tuples
            List of prior specifications for each parameter in the order 
            expected by the sampler. Each element is a tuple like
            (distribution_name, distribution_parameters_dict).
        """
        sd = np.std(self.data)
        loc = np.percentile(self.data,35)
    
        prior_specs = []
    
        # Location
        if self.config[0] == 0:
            prior_specs.append(('normal', {'loc': loc, 'scale': loc*0.1}))
        else:
            # intercept
            prior_specs.append(('normal', {'loc': loc, 'scale': loc*0.5}))  
            for _ in range(self.config[0]):
                prior_specs.append(('normal', {'loc': 0, 'scale': 0.3}))
    
        # Scale
        if self.config[1] == 0:
            prior_specs.append(('normal', {'loc': sd-0.15 , 'scale': 0.3}))
        else:
            lower = np.log(sd * 0.5)
            upper = np.log(sd * 1.5)
            # intercept on log-scale
            prior_specs.append(('normal', {'loc': lower, 'scale': upper - lower}))  
            for _ in range(self.config[1]):
                prior_specs.append(('normal', {'loc': 0, 'scale': 0.025}))
    
        # Shape
        if self.config[2] == 0:
            prior_specs.append(('normal', {'loc': 0, 'scale': 0.1}))
        else:
            # intercept
            prior_specs.append(('normal', {'loc': 0, 'scale': 0.2})) 
            for _ in range(self.config[2]):
                prior_specs.append(('normal', {'loc': 0, 'scale': 0.025}))
    
        return prior_specs


    def suggest_bounds(self, buffer: float = 0.5) -> List[Tuple[float, float]]:
        """
        Suggests bounds for MLE optimization based on config and distribution
        type.
    
        Parameters
        ----------
        buffer : float
            Fractional buffer around stationary parameter estimates.
    
        Returns
        -------
        bounds : List[Tuple[float, float]]
            List of (lower, upper) tuples for each parameter in order.
        """
        # Step 1: Estimate stationary parameters
        if self.dist.name.lower() in ['genextreme', 'gev']:
            shape, loc, scale = GEV_parsViaLM(self.data)
            log_scale = np.log(scale)
        elif self.dist.name.lower() in ['genpareto', 'gpd']:
            shape, loc, scale = GPD_parsViaLM(self.data)
            log_scale = np.log(scale)
        else:
            raise ValueError("Unsupported distribution. Use GEV or GPD.")
    
        bounds = []
    
        # Location
        if self.config[0] == 0:
            bounds.append((loc * (1 - buffer), loc * (1 + buffer)))
        else:
            bounds.append((loc * (1 - buffer), loc * (1 + buffer)))  # B0
            for _ in range(self.config[0]):
                bounds.append((-0.1, 0.1))  # B_i
    
        # Scale
        if self.config[1] == 0:
            bounds.append((scale * 0.5, scale * 2))
        else:
            bounds.append((log_scale - 0.5, log_scale + 0.5))  # log(a0)
            for _ in range(self.config[1]):
                bounds.append((-0.1, 0.1))  # a_i
    
        # Shape
        if self.config[2] == 0:
            bounds.append((shape - 0.2, shape + 0.2))
        else:
            bounds.append((shape - 0.5, shape + 0.5))  # k0
            for _ in range(self.config[2]):
                bounds.append((-0.1, 0.1))  # k_i
    
        return bounds    

   
    def log_prior(self, params):
        """
        Compute the log prior probability of the parameter vector.
        
        This method calculates the sum of log-prior probabilities for each 
        parameter based on the specified prior distributions in 
        self.prior_specs. 
        The number and type of parameters are determined by the 
        non-stationarity configuration (self.config) 
        provided at initialization.
        
        Parameters
        ----------
        params : array-like
            A 1D array of parameter values corresponding to the linear or 
            exponential models for location, scale, and shape parameters. 
            The number and order of parameters must match the configuration.
        
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
        if self.prior_specs is None:
            return 0.0
    
        logp = 0.0
        param_idx = 0
        cov_counts = self.config  # [n_loc_cov, n_scale_cov, n_shape_cov]
    
        for group in range(3):  # location, scale, shape
            count = cov_counts[group]
            n_params = count + 1 if count > 0 else 1  # intercept + covariates 
                                                    # OR just stationary param
            for _ in range(n_params):
                ptype, kwargs = self.prior_specs[param_idx]
                val = params[param_idx]
    
                # Compute log-prior value
                if ptype == 'normal':
                    prior_val = norm.logpdf(val, **kwargs)
                elif ptype == 'uniform':
                    prior_val = uniform.logpdf(val, **kwargs)
                elif ptype == 'halfnormal':
                    prior_val = halfnorm.logpdf(val, **kwargs)
                else:
                    raise ValueError(f"Unsupported prior type: {ptype}")
    
                if not np.isfinite(prior_val):
                    print(f"[WARNING] Prior {ptype} returned invalid logpdf for"
                          f" param {param_idx}: val={val}, kwargs={kwargs}")
                    return -np.inf
    
                logp += prior_val
                param_idx += 1
    
        return logp

    

    def neg_log_likelihood(self, params):
        """
        Compute the negative log-likelihood for the given parameter vector.

        This method delegates the calculation to the neg_log_likelihood_ns
        function using the class attributes such as data, covariates, model 
        configuration, and distribution type.

        Parameters
        ----------
        params : array-like
            A 1D array of model parameters corresponding to the location, 
            scale, and shape components of the non-stationary distribution.

        Returns
        -------
        float
            The negative log-likelihood value.
        """
        return neg_log_likelihood_ns(
            params, self.data, self.cov, self.config, self.dist
        )
    

    def posterior_log_prob(self, params):
        """
        Compute the log posterior probability for the given parameter vector.

        The posterior is calculated as the sum of the log-prior and the 
        log-likelihood (negated). This is used for Bayesian inference, 
        particularly in MCMC sampling.

        Parameters
        ----------
        params : array-like
            A 1D array of parameter values matching the model configuration.

        Returns
        -------
        float
            The log posterior probability. If the prior is improper or 
            evaluates to -inf, the result will reflect that.
        """
        return -1*self.neg_log_likelihood(params) + self.log_prior(params)
    
    
    def numerical_grad_log_posterior(self, params, h=1e-2):
        """
        Compute the numerical gradient of the log-posterior with respect to 
        parameters.

        This uses the central difference method to approximate the gradient of 
        the log-posterior at the given parameter vector.

        Parameters
        ----------
        params : array-like
            A 1D array of parameter values at which to evaluate the gradient.

        h : float or array-like, optional
            The step size for finite difference approximation. Can be a scalar
            or an array of the same shape as `params` for per-parameter step 
            sizes.
            Default is 1e-2.

        Returns
        -------
        grad : ndarray
            A 1D array containing the approximate gradient of the log-posterior 
            with respect to each parameter.
        """        
        grad = np.zeros_like(params)
        h_vec = h if hasattr(h, '__len__') else np.full_like(params, h, 
                                                             dtype=float)
    
        for i in range(len(params)):
            step = np.zeros_like(params)
            step[i] = h_vec[i]
            f_plus = self.posterior_log_prob(params + step)
            f_minus = self.posterior_log_prob(params - step)
            grad[i] = (f_plus - f_minus) / (2 * h_vec[i])
    
        return grad
    

    def MH_Mala(
        self,
        num_samples: int,
        initial_params: Union[list[float], np.ndarray],
        step_sizes: list[float],
        T: float = 1.0  # Optional temperature scaling factor, default 1 
                        # (no scaling)
    ) -> tuple[np.ndarray, float]:
        """
        Perform MALA sampling to generate samples from the posterior 
        distribution.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        initial_params : Union[list[float], np.ndarray]
            Initial parameter vector to start the Markov chain.
        step_size : float
            Step size epsilon for MALA proposals.
        T : float, optional
            Temperature factor to scale the acceptance ratio.
            Values greater than 1 make acceptance more lenient, values less
            than 1 stricter. Default is 1.
        
        Returns
        -------
        samples : np.ndarray
            Array of shape `(num_samples, n_parameters)` containing sampled 
            parameter vectors.
        acceptance_rate : float
            Fraction of proposals accepted.
        """
        step_sizes = np.array(step_sizes)  # Convert list to NumPy array
        samples = []
        current_params = np.array(initial_params)
        current_log_post = self.posterior_log_prob(current_params)
        total_params = len(current_params)
    
        if self.prior_specs is None:
            self.prior_specs = self.suggest_priors()
    
        accept_count = 0
    
        for _ in range(num_samples):
            # Compute gradient at current position
            grad = self.numerical_grad_log_posterior(current_params)
    
            # Proposal mean for MALA
            proposal_mean = current_params + (step_sizes**2 / 2) * grad
    
            # Draw proposal from normal centered at proposal_mean
            proposal = proposal_mean + step_sizes * np.random.normal(
                size=total_params)
    
            # Compute gradient at proposal for asymmetric correction
            grad_proposal = self.numerical_grad_log_posterior(proposal)
    
            # Compute log proposal densities q(proposal | current) and 
            # q(current | proposal)
            log_q_forward = -np.sum(((proposal - current_params - (
                step_sizes**2 / 2) * grad)**2) / (2 * step_sizes**2))

            log_q_backward = -np.sum(((current_params - proposal - (
                step_sizes**2 / 2) * grad_proposal)**2) / (2 * step_sizes**2))

    
            proposed_log_post = self.posterior_log_prob(proposal)
    
            # Log acceptance ratio
            log_alpha = (proposed_log_post + log_q_backward) - (
                current_log_post - log_q_forward)
            log_alpha = log_alpha / T  # Scale by temperature factor
    
            if log_alpha > 0 or np.log(np.random.rand()) < log_alpha:
                current_params = proposal
                current_log_post = proposed_log_post
                accept_count += 1
    
            samples.append(current_params.copy())
    
        acceptance_rate = accept_count / num_samples
    
        return np.array(samples), acceptance_rate



    def MH_RandWalk(
        self,
        num_samples: int,
        initial_params: Union[list[float], np.ndarray],
        proposal_widths: Union[list[float], np.ndarray],
        T: float
    ) -> tuple[np.ndarray, float]:
        """
        Perform Metropolis-Hastings sampling to generate samples from the 
        posterior distribution.
    
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        initial_params : Union[list[float], np.ndarray]
            Initial parameter vector to start the Markov chain.
        proposal_widths : Union[list[float], np.ndarray]
            Standard deviations for the Gaussian proposal distribution 
            (random walk).
            Length must match the number of parameters.
        T : float
            Temperature factor to scale the acceptance ratio.
            Values greater than 1 make the acceptance more lenient, values less
            than 1 make it stricter.
    
        Raises
        ------
        ValueError
            If the length of `proposal_widths` does not match the number of
            parameters to sample.
    
        Returns
        -------
        np.ndarray
            Array of shape `(num_samples, n_parameters)` containing the sampled
            parameter vectors.
        """
        samples = []
        current_params = np.array(initial_params)
        current_log_post = self.posterior_log_prob(current_params)
        total_params = sum(self.config) + 3
        if len(proposal_widths) != total_params:
            raise ValueError(
                "Length of proposal_widths must match number of parameters"
            )
            
        if self.prior_specs is None:
            self.prior_specs = self.suggest_priors()
        
        accept_count  = 0
        for _ in range(num_samples):
            proposal = current_params + np.random.normal(0, 
                                                         proposal_widths,
                                                         size=total_params)

            proposed_log_post = self.posterior_log_prob(proposal)

            log_alpha = proposed_log_post - current_log_post
            log_alpha = log_alpha / T
            if log_alpha > 0:
                accept = True
            else:
                u = np.log(np.random.rand())  # log uniform random in (-inf, 0]
                accept = u < log_alpha
            
            if accept:
                current_params = proposal
                current_log_post = proposed_log_post
                accept_count += 1

            samples.append(current_params)
            
        acceptance_rate = accept_count / num_samples

        return np.array(samples),acceptance_rate
    
    
    def hamiltonian(self, params, momentum,T):
        """
        Compute the Hamiltonian (total energy) of the system for HMC sampling.
    
        The Hamiltonian is the sum of the potential energy and kinetic energy. 
        In this context:
        - Potential energy is defined as the negative log-posterior (scaled by
           T),which encourages high-probability regions of parameter space.
        - Kinetic energy is computed as 0.5 * sum(momentum^2), assuming a 
          standard Gaussian momentum distribution.
    
        Parameters
        ----------
        params : array-like
            Current position in parameter space (model parameters).
        
        momentum : array-like
            Auxiliary momentum variables, typically sampled from a standard 
            normal distribution.
        
        T : float
            Temperature scaling factor. T=1 corresponds to standard HMC; 
            higher values flatten the posterior (tempering).
    
        Returns
        -------
        float
            The total Hamiltonian energy (scaled potential + kinetic energy).
        """
        potential_energy = -self.posterior_log_prob(params)  
        # assuming negative log posterior = potential energy
        kinetic_energy = 0.5 * np.sum(momentum ** 2)
        return T*potential_energy + kinetic_energy


    def MH_Hmc(
        self,
        num_samples: int,
        initial_params: Union[list[float], np.ndarray],
        step_size: float = 0.1,
        num_leapfrog_steps: int = 10,
        T: float = 1.0  # Optional temperature scaling
    ) -> tuple[np.ndarray, float]:
        """
        Perform HMC sampling to generate samples from the posterior 
        distribution.
    
        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        initial_params : Union[list[float], np.ndarray]
            Initial parameter vector to start the Markov chain.
        step_size : float
            Step size (epsilon) for the leapfrog integrator.
        num_leapfrog_steps : int
            Number of leapfrog steps per iteration.
        T : float, optional
            Temperature scaling factor for log-acceptance ratio.
    
        Returns
        -------
        samples : np.ndarray
            Array of shape (num_samples, n_parameters) containing parameter
            vectors.
        acceptance_rate : float
            Fraction of proposals accepted.
        """
        dim = len(initial_params)
        samples = np.zeros((num_samples, dim))
        accepted = 0
    
        current_params = initial_params.copy()
        current_log_post = self.posterior_log_prob(current_params)
    
        for i in range(num_samples):
            # Draw momentum
            current_momentum = np.random.normal(0, 1, dim)
            proposed_params = current_params.copy()
            proposed_momentum = current_momentum.copy()
    
            # Leapfrog integration
            grad = self.numerical_grad_log_posterior(proposed_params)
            proposed_momentum += 0.5 * step_size * grad
    
            for _ in range(num_leapfrog_steps):
                proposed_params += step_size * proposed_momentum
                if _ != num_leapfrog_steps - 1:
                    grad = self.numerical_grad_log_posterior(proposed_params)
                    proposed_momentum += step_size * grad
    
            grad = self.numerical_grad_log_posterior(proposed_params)
            proposed_momentum += 0.5 * step_size * grad
            proposed_momentum *= -1  # Negate momentum for symmetry
    
            # Hamiltonian
            # def H(params, momentum):
            #     log_post = self.posterior_log_prob(params)
            #     kinetic = 0.5 * np.sum(momentum**2)
            #     return -T * log_post + kinetic
    
            current_H = self.hamiltonian(current_params, current_momentum,T)
            proposed_H = self.hamiltonian(proposed_params, proposed_momentum,T)
    
            log_alpha = - (proposed_H - current_H)
            if np.log(np.random.rand()) < log_alpha:
                current_params = proposed_params
                current_log_post = self.posterior_log_prob(current_params)
                accepted += 1
    
            samples[i, :] = current_params
    
        acceptance_rate = accepted / num_samples
        return samples, acceptance_rate

     
    def frequentist_nsEVD(
        self,
        initial_params: Union[List[float], np.ndarray],
        max_retries: int = 10
    ) -> tuple[np.ndarray, float]:
        """
        Estimate non-stationary EVD parameters via MLE with retries.
    
        Parameters
        ----------
        initial_params : array-like
            Initial guess for parameters.
        max_retries : int
            Number of retry attempts with modified initial guess.
    
        Returns
        -------
        params : array-like
            Estimated parameters.
        """
        retry = 0
        params = np.array(initial_params)
        
        if self.bounds is None:
            self.bounds = self.suggest_bounds()
            
        while retry < max_retries:
            res = minimize(self.neg_log_likelihood, 
                           params,
                           method='L-BFGS-B', 
                           bounds=self.bounds)
            if res.success:
                print(f"Optimization succeeded after {retry+1} attempt(s)")
                return res.x
            else:
                print(
                    f"Optimization failed at attempt {retry+1}: {res.message}"
                )
                params += np.random.normal(0, 0.01, size=len(params))
                
                retry += 1
                
            # Fallback to Nelder-Mead
        print("Optimization failed after max retries, trying fallback"
              " (Nelder-Mead)...")
        for _ in range(max_retries):
            res = minimize(self.neg_log_likelihood,
                           params,
                           method='Nelder-Mead')
            if res.success:
                print("Fallback optimization (Nelder-Mead) succeeded.")
                return res.x
            params += np.random.normal(0, 0.01, size=len(params))
        
        raise RuntimeError("Optimization failed after max retries and fallback"
                           ".")
    
        
        
         
    @staticmethod      
    def ns_EVDrvs(
        dist: rv_continuous,
        params: Union[List[float], np.ndarray],
        cov: np.ndarray,
        config: List[int],
        size: int 
    ) -> np.ndarray:
        """
        Generate non-stationary GEV or GPD random samples.
    
        Parameters
        ----------
        dist_name : rv_continuous
                    SciPy continuous distribution object (e.g., genextreme or 
                                                  genpareto).
        params : list
            Flattened parameter list according to config.
        cov : np.ndarray
            Covariate matrix, shape (n_covariates, n_samples).
        config : list of int
            Non-stationarity config [loc, scale, shape].
        size : int
            Number of random samples to generate.
    
        Returns
        -------
        np.ndarray
            Generated non-stationary random variates.
        """
        cov = np.atleast_2d(cov)
        n_samples = cov.shape[1]
        
        if size != n_samples:
            raise ValueError(f"Provided 'size' ({size}) must match number of "
                             "samples in covariate matrix ({n_samples})")
        idx = 0
    
        # Location
        if config[0] >= 1:
            n = config[0]
            B = params[idx:idx + n + 1]
            loc = B[0] + B[1:] @ cov[:n, :]
            idx += n + 1
        else:
            loc = np.full(n_samples, params[idx])
            idx += 1
    
        # Scale
        if config[1] >= 1:
            n = config[1]
            A = params[idx:idx + n + 1]
            scale = np.exp(A[0] + A[1:] @ cov[:n, :])
            idx += n + 1
        else:
            scale = np.full(n_samples, params[idx])
            idx += 1
    
        # Shape
        if config[2] >= 1:
            n = config[2]
            K = params[idx:idx + n + 1]
            shape = K[0] + K[1:] @ cov[:n, :]
        else:
            shape = np.full(n_samples, params[idx])
    
        return dist.rvs(c=shape, loc=loc, scale=scale, size=n_samples)
        
