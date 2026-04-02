from __future__ import annotations

import logging
import warnings
from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import minimize
from scipy.stats import (
    rv_continuous,
)
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from .utils import (
    GEV_parsViaLM,
    GPD_parsViaLM,
    _grad_nll_gev,
    _grad_nll_gpd,
    _grad_total_log_prior,
    _total_log_prior,
    gelman_rubin,
    neg_log_likelihood_ns,
)

logging.basicConfig(
    filename='nsEVDx_run.log',
    filemode='w', # 'w' overwrites each time, 'a' appends to the end
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nsEVDx")

# Helper: acceptance rate warning
def _check_acceptance(rate: float, sampler_name: str) -> None:
    lo, hi = 0.20, 0.70
    if not (lo <= rate <= hi):
        warnings.warn(
            f"[{sampler_name}] Acceptance rate {rate:.1%} is outside the"
            f"recommended range [{lo:.0%}, {hi:.0%}]. "
            "Consider tuning step_size / proposal_widths.",
            stacklevel=2,
        )

# nsEVDx main model code
class NonStationaryEVD:
    def __init__(self, config, data, cov, dist, prior_specs=None, bounds=None):
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
                whereas the scale parameter is modeled exponentially to ensure
                positivity.
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
        self.n_cov = self.cov.shape[0] if self.cov.ndim > 1 else 1
        self.prior_specs = prior_specs
        self.bounds = bounds
        self._is_gev = dist in ("genextreme", "gev")
        self._is_gpd = dist in ("genpareto",  "gpd")
        assert self.data.shape[0] == self.cov.shape[1], (
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
        parameter vector, based on the provided configuration (config. vector).

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
            desc.append("B0 (location intercept)")
            for i in range(1, n + 1):
                desc.append(f"B{i} (location slope for covariate {i})")
            idx += n + 1
        else:
            desc.append("mu (stationary location)")
            idx += 1

        # Scale parameters
        if config[1] >= 1:
            n = int(config[1])
            desc.append("a0 (scale intercept)")
            for i in range(1, n + 1):
                desc.append(f"a{i} (scale slope for covariate {i})")
            idx += n + 1
        else:
            desc.append("sigma (scale)")
            idx += 1

        # Shape parameters
        if config[2] >= 1:
            n = int(config[2])
            desc.append("k0 (shape intercept)")
            for i in range(1, n + 1):
                desc.append(f"k{i} (shape slope for covariate {i})")
        else:
            desc.append("xi (shape)")

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
        loc = np.percentile(self.data, 35)

        prior_specs = []

        # Location
        if self.config[0] == 0:
            prior_specs.append(("normal", {"loc": loc, "scale": loc * 0.1}))
        else:
            # intercept
            prior_specs.append(("normal", {"loc": loc, "scale": loc * 0.5}))
            for _ in range(self.config[0]):
                prior_specs.append(("normal", {"loc": 0, "scale": 0.3}))

        # Scale
        if self.config[1] == 0:
            prior_specs.append(("normal", {"loc": sd - 0.15, "scale": 0.3}))
        else:
            lower = np.log(sd * 0.5)
            upper = np.log(sd * 1.5)
            # intercept on log-scale
            prior_specs.append(("normal", {"loc": lower, "scale": upper - lower}))
            for _ in range(self.config[1]):
                prior_specs.append(("normal", {"loc": 0, "scale": 0.025}))

        # Shape
        if self.config[2] == 0:
            prior_specs.append(("normal", {"loc": 0, "scale": 0.3}))
        else:
            # intercept
            prior_specs.append(("normal", {"loc": 0, "scale": 0.2}))
            for _ in range(self.config[2]):
                prior_specs.append(("normal", {"loc": 0, "scale": 0.025}))

        return prior_specs

    def suggest_bounds(self, buffer: float = 0.5) -> List[Tuple[float, float]]:
        """
        Suggests bounds for MLE optimization based on config. vector
        and distribution.

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
        if self.dist.name.lower() in ["genextreme", "gev"]:
            shape, loc, scale = GEV_parsViaLM(self.data)
            log_scale = np.log(scale)
        elif self.dist.name.lower() in ["genpareto", "gpd"]:
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

    def _log_prior(self, params):
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

        return _total_log_prior(params, self.prior_specs)


    def _grad_log_prior(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log prior probability: ∂log π(θ)/∂θ
        Parameters
        ----------
            params : 1D array of current parameter values.
        Returns
        -------
            1D array of gradients for each prior.
        """
        # Pass the current parameters and the model's specific prior specs
        return _grad_total_log_prior(params, self.prior_specs)


    def _neg_log_likelihood(self, params):
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

    def _posterior_log_prob(self, params):
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
        with np.errstate(over='ignore', invalid='ignore'):
            x = -1 * self._neg_log_likelihood(params) + self._log_prior(params)
        return x


    def _grad_log_posterior(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the full gradient of the log-posterior. The gradient is
        calculated as the sum of the log-likelihood gradient and the
        log-prior gradient:
            ∂log p(θ|x)/∂θ = −∂NLL/∂θ + ∂log π(θ)/∂θ.
        The likelihood gradient uses high-performance analytical derivations for
        GEV and GPD distributions. If an unsupported distribution is used, it
        falls back to a central-difference numerical approximation. The prior
        gradient is always computed numerically.
        Parameters
        ----------
        params : np.ndarray
            1D array of model parameters (float64) at which to evaluate
            the gradient.
        Returns
        -------
        np.ndarray
            A 1D array of the same shape as `params` containing the total
            log-posterior gradient. Returns an array of zeros if the
            parameters fall outside the distribution support (NaN-safe).
        Notes
        -----
        - Uses `np.errstate` to suppress floating-point overflows during
        gradient calculation in extreme regions of the parameter space.
        - Boundary handling: If the analytical gradient returns NaN (e.g.,
         violating the condition 1 + ξ(x-μ)/σ > 0), the method returns
        a zero vector to prevent the HMC sampler from diverging.
        """
        # Likelihood Gradient (∂LL/∂θ)
        if self._is_gev:
            with np.errstate(over='ignore', invalid='ignore'):
                grad_nll = _grad_nll_gev(
                    params, self.data, self.cov, self.config
                )
                grad_ll = -grad_nll
        elif self._is_gpd:
            with np.errstate(over='ignore', invalid='ignore'):
                grad_nll = _grad_nll_gpd(
                    params, self.data, self.cov, self.config
                )
                grad_ll = -grad_nll
        else:
            # Fallback if the distribution isn't GEV or GPD
            grad_ll = self._numerical_grad_log_posterior(params)
            grad_nll = -grad_ll

        # Nan-check (If we hit an impossible v <= 0 region, return 0)
        if np.any(np.isnan(grad_nll)):
            return np.zeros_like(params)

        # Prior Gradient (∂logπ/∂θ)
        grad_prior = self._grad_log_prior(params)

        # Total Gradient
        return grad_ll + grad_prior

    def _numerical_grad_log_posterior(self, params, h=1e-3):
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
        h_vec = h if hasattr(h, "__len__") else np.full_like(params, h, dtype=float)

        for i in range(len(params)):
            step = np.zeros_like(params)
            step[i] = h_vec[i]
            f_plus = -1* self._neg_log_likelihood(params + step)
            f_minus = -1* self._neg_log_likelihood(params - step)
            if f_plus > -1e20 and f_minus > -1e20:
                grad[i] = (f_plus - f_minus) / (2 * h_vec[i])
            else:
                grad[i] = 0.0
        return grad


    def _Mala_1chain(
        self,
        num_samples: int,
        initial_params: Union[list[float], np.ndarray],
        step_sizes: list[float],
        T: float,  # Optional temperature scaling factor, default 1
        burn_in: int,
        chain_id: int,
        show_progress: bool,
    ) -> tuple[np.ndarray, float]:
        """
        Samples from the posterior distribution of the parameters by
        implementing MCMC simulation based on Metropolis-Adjusted Langevin
        algorithm.

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
        total_params = sum(self.config) + 3
        if len(step_sizes) != total_params:
            raise ValueError(
                "Length of step sizes must match number of parameters"
            )

        if self.prior_specs is None:
            self.prior_specs = self.suggest_priors()

        step_sizes = np.array(step_sizes)  # Convert list to NumPy array
        total_samples = num_samples + burn_in
        samples = np.empty((total_samples, total_params))
        current_params = np.array(initial_params)
        current_log_post = self._posterior_log_prob(current_params)
        total_params = len(current_params)

        accept_count = 0

        pbar = tqdm(
            range(total_samples),
            desc=f"MALA Chain {chain_id+1}",
            disable=not show_progress,
            position=0,
            leave=True,
            ascii=True,
            unit="sample",
            dynamic_ncols=True
        )

        for i in pbar:
            # Compute gradient at current position
            grad = self._grad_log_posterior(current_params)

            # Proposal mean for MALA
            proposal_mean = current_params + (step_sizes**2 / 2) * grad

            # Draw proposal from normal centered at proposal_mean
            proposal = proposal_mean + step_sizes * np.random.normal(size=total_params)

            # Compute gradient at proposal for asymmetric correction
            grad_proposal = self._grad_log_posterior(proposal)

            # Compute log proposal densities q(proposal | current) and
            # q(current | proposal)
            log_q_forward = -np.sum(
                ((proposal - current_params - (step_sizes**2 / 2) * grad) ** 2)
                / (2 * step_sizes**2)
            )

            log_q_backward = -np.sum(
                ((current_params - proposal - (step_sizes**2 / 2) * grad_proposal) ** 2)
                / (2 * step_sizes**2)
            )

            proposed_log_post = self._posterior_log_prob(proposal)

            # Log acceptance ratio
            log_alpha = (proposed_log_post + log_q_backward) - (
                current_log_post - log_q_forward
            )
            log_alpha = log_alpha / T  # Scale by temperature factor

            if log_alpha > 0 or np.log(np.random.rand()) < log_alpha:
                current_params = proposal
                current_log_post = proposed_log_post
                accept_count += 1

            samples[i] = current_params.copy()

        pbar.close()

        acceptance_rate = accept_count / num_samples
        _check_acceptance(acceptance_rate, "MH_MALA")

        return samples[burn_in:,:], acceptance_rate

    def MH_Mala(
        self,
        num_samples: int,
        initial_params: Union[List[float], np.ndarray],
        step_sizes: Union[List[float], np.ndarray],
        T: float = 1.0,
        burn_in: int = 0,
        num_chains: int = 1,
        show_progress: bool = True,
        n_jobs: int = 1,
    ) -> Union[Tuple[np.ndarray, float], Tuple[List[np.ndarray],
                                               List[float], np.ndarray]]:
        """
        Metropolis-Adjusted Langevin Algorithm (MALA) sampler.

        Parameters
        ----------
        num_samples : int
            Total iterations per chain (including burnin).
        initial_params : array-like
            Starting parameter vector.
        step_sizes : array-like
            Per-parameter step sizes (epsilon).
        T : float
            Temperature scaling factor.
        burnin : int
            Number of initial samples to discard per chain.
        num_chains : int
            Number of independent chains.
        show_progress : bool
            Display tqdm progress bars.
        n_jobs : int
            Parallel jobs via joblib.

        Returns
        -------
        Same convention as MH_RandWalk.
        """
        initial_params = np.asarray(initial_params, dtype=float)
        step_sizes = np.asarray(step_sizes, dtype=float)

        def _run(cid):
            jitter = np.random.normal(0, step_sizes * 0.05) if cid > 0 else 0
            return self._Mala_1chain(
                num_samples, initial_params + jitter,
                step_sizes, T, burn_in, cid,
                show_progress=(show_progress and (n_jobs == 1 or cid == 0)),
            )

        if num_chains == 1:
            return _run(0)

        if _JOBLIB_AVAILABLE and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run)(cid) for cid in range(num_chains)
            )
        else:
            results = [_run(cid) for cid in range(num_chains)]

        chains, rates = zip(*results)
        chains_list = list(chains)
        rates_list = list(rates)

        if num_chains >= 2:
            r_hat = gelman_rubin(chains)
            if show_progress:
                max_r = np.max(r_hat)
                print("\n" + "="*55)
                print("MCMC CONVERGENCE REPORT")
                print("-" * 55)
                print(f"Average Acceptance: {np.mean(rates_list)*100:.2f}%")
                if np.any(r_hat > 1.1):
                    print(f"WARNING: Some chains may not have converged"
                          f" (R-hat, {max_r:.3f} > 1.1).")
                else:
                    print("Convergence Check: PASSED (r_hat < 1.1)")
                    print("-"*55 + "\n")
                for name, r in zip(self.descriptions, r_hat):
                    print(f"{name:<30} R-hat: {r:.4f}")
                print("=" * 55)
            return chains_list, rates_list, r_hat

        return list(chains), list(rates)

    def _RandWalk_1chain(
        self,
        num_samples: int,
        initial_params: Union[list[float], np.ndarray],
        proposal_widths: Union[list[float], np.ndarray],
        T: float,
        burn_in: int,
        chain_id: int,
        show_progress: bool,
    ) -> tuple[np.ndarray, float]:
        """
        Perform Metropolis-Hastings sampling by implementing the random walk
        approach to generate samples from the posterior distribution.

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
        total_params = sum(self.config) + 3
        if len(proposal_widths) != total_params:
            raise ValueError(
                "Length of proposal_widths must match number of parameters"
            )

        if self.prior_specs is None:
            self.prior_specs = self.suggest_priors()

        total_samples = num_samples + burn_in
        samples = np.empty((total_samples, total_params))
        current_params = np.array(initial_params)
        current_log_post = self._posterior_log_prob(current_params)
        accept_count = 0

        pbar = tqdm(
            range(total_samples),
            desc=f"RandWalk Chain {chain_id+1}",
            disable=not show_progress,
            position=0,
            leave=True,
            ascii=True,
            unit="sample",
            dynamic_ncols=True
        )

        for i in pbar:
            proposal = current_params + np.random.normal(
                0, proposal_widths, size=total_params
            )

            proposed_log_post = self._posterior_log_prob(proposal)

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

            samples[i] = current_params

        pbar.close()

        acceptance_rate = accept_count / num_samples
        _check_acceptance(acceptance_rate, "MH_RandWalk")

        return samples[burn_in:,:], acceptance_rate

    def MH_RandWalk(
        self,
        num_samples: int,
        initial_params: Union[List[float], np.ndarray],
        proposal_widths: Union[List[float], np.ndarray],
        T: float = 1.0,
        burn_in: int = 0,
        num_chains: int = 1,
        show_progress: bool = True,
        n_jobs: int = 1,
    ) -> Union[Tuple[np.ndarray, float], Tuple[List[np.ndarray],
                                               List[float], np.ndarray]]:
        """
        Metropolis-Hastings Random-Walk sampler.

        Parameters
        ----------
        num_samples : int
            Total iterations per chain (excluding burnin).
        initial_params : array-like
            Starting parameter vector (same start used for all chains +
            small jitter for chains > 1).
        proposal_widths : array-like
            Per-parameter proposal standard deviations.
        T : float
            Temperature (default 1.0 = no tempering).
        burnin : int
            Number of initial samples to discard per chain.
        num_chains : int
            Number of independent chains.  R-hat is automatically computed
            when num_chains >= 2.
        show_progress : bool
            Display tqdm progress bars (default True).
        n_jobs : int
            Parallel jobs via joblib (-1 = all cores).  Requires joblib.

        Returns
        -------
        If num_chains == 1:
            (samples [n_post, n_params], acceptance_rate)
        If num_chains  > 1:
            (list_of_chains, list_of_acceptance_rates)
            Use run_chains() for automatic R-hat reporting.
        """
        initial_params = np.asarray(initial_params, dtype=float)
        proposal_widths = np.asarray(proposal_widths, dtype=float)

        def _run(cid):
            jitter = np.random.normal(0, proposal_widths * 0.05) if cid > 0 else 0
            return self._RandWalk_1chain(
                num_samples, initial_params + jitter,
                proposal_widths, T, burn_in, cid,
                show_progress=(show_progress and (n_jobs == 1 or cid == 0)),
            )

        if num_chains == 1:
            return _run(0)

        if _JOBLIB_AVAILABLE and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run)(cid) for cid in range(num_chains)
            )
        else:
            results = [_run(cid) for cid in range(num_chains)]

        chains, rates = zip(*results)
        chains_list = list(chains)
        rates_list = list(rates)

        if num_chains >= 2:
            r_hat = gelman_rubin(chains)
            if show_progress:
                max_r = np.max(r_hat)
                print("\n" + "="*55)
                print("MCMC CONVERGENCE REPORT")
                print("-" * 55)
                print(f"Average Acceptance: {np.mean(rates_list)*100:.2f}%")
                if np.any(r_hat > 1.1):
                    print(f"WARNING: Some chains may not have converged"
                          f" (R-hat, {max_r:.3f} > 1.1).")
                else:
                    print("Convergence Check: PASSED (r_hat < 1.1)")
                    print("-"*55 + "\n")
                for name, r in zip(self.descriptions, r_hat):
                    print(f"{name:<30} R-hat: {r:.4f}")
                print("=" * 55)
            return chains_list, rates_list, r_hat

        return chains_list, rates_list


    def MH_Hmc(
         self,
         num_samples: int,
         initial_params: Union[List[float], np.ndarray],
         step_size: float = 0.01,
         num_leapfrog_steps: int = 10,
         burn_in: int = 1000,
         num_chains: int = 1,
         show_progress: bool = True,
         n_jobs: int = 1,
         T: float = 1.0,
     ) -> Union[Tuple[np.ndarray, float], Tuple[List[np.ndarray],
                                                List[float], np.ndarray]]:
         """
         Hamiltonian Monte Carlo (HMC) sampler. Wrapper around the HMC egine class
         to run multi-chain HMC sampling.

        Parameters
        ----------
         num_samples : int
             Total iterations per chain (excluding burnin).
         initial_params : array-like
             Starting parameter vector.
         step_size : float
             Leapfrog step size epsilon.
         num_leapfrog_steps : int
             Number of leapfrog steps per proposal.
         burn_in : int
             Number of initial samples to discard per chain.
         num_chains : int
             Number of independent chains.
         show_progress : bool
             Display tqdm progress bars.
         n_jobs : int
             Parallel jobs via joblib.
         T : float
             Temperature scaling factor.
         Returns
         -------
         dict
             A dictionary containing:
             - 'chains': List of sample arrays [chain, iteration, parameter].
             - 'r_hats': Gelman-Rubin convergence values for each parameter.
             - 'acceptance_rates': List of acceptance rates per chain.
             - 'step_sizes': List of final step sizes per chain.
             - 'divergences': List of divergence counts per chain.
         """
         # Import locally to avoid circular dependency issues
         from .hmc_engine import HMCEngine
         # Initialize the engine with this model instance
         engine = HMCEngine(model=self,
                            grad_method="analytical")
         # Delegate execution to the engine's multi-chain runner
         return engine._run_chains(
             sampler="hmc",
             num_samples=num_samples,
             initial_params=initial_params,
             step_size=step_size,
             n_leapfrog=num_leapfrog_steps,
             burn_in=burn_in,
             num_chains=num_chains,
             n_jobs=n_jobs,
             show_progress=show_progress,
             T=T)

    def frequentist_nsEVD(self,
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
            res = minimize(
                self._neg_log_likelihood, params,
                method="L-BFGS-B", bounds=self.bounds
            )
            if res.success:
                logger.info("Optimization succeeded after %d attempt(s)",retry + 1)
                return res.x
            else:
                logger.warning("Optimization failed at attempt %d: %s",
                               retry + 1, res.message)
                params += np.random.normal(0, 0.01, size=len(params))

                retry += 1

            # Fallback to Nelder-Mead
        logger.warning("Optimization failed after max retries,"
                       " trying fallback (Nelder-Mead)...")
        for _ in range(max_retries):
            res = minimize(self._neg_log_likelihood, params, method="Nelder-Mead")
            if res.success:
                logger.info("Fallback optimization (Nelder-Mead) succeeded.")
                return res.x
            params += np.random.normal(0, 0.01, size=len(params))

        raise RuntimeError("Optimization failed after max retries and fallback.")

    @staticmethod
    def ns_EVDrvs(
        dist: rv_continuous,
        params: Union[List[float], np.ndarray],
        cov: np.ndarray,
        config: List[int],
        size: int,
    ) -> np.ndarray:
        """
        Generate non-stationary GEV or GPD random samples.
        Parameters
        ----------
        dist : rv_continuous
            SciPy continuous distribution object (e.g., genextreme or genpareto).
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
            raise ValueError(
                f"Provided 'size' ({size}) must match number of "
                "samples in covariate matrix ({n_samples})"
            )
        idx = 0

        # Location
        if config[0] >= 1:
            n = config[0]
            B = params[idx : idx + n + 1]
            loc = B[0] + B[1:] @ cov[:n, :]
            idx += n + 1
        else:
            loc = np.full(n_samples, params[idx])
            idx += 1

        # Scale
        if config[1] >= 1:
            n = config[1]
            A = params[idx : idx + n + 1]
            scale = np.exp(A[0] + A[1:] @ cov[:n, :])
            idx += n + 1
        else:
            scale = np.full(n_samples, params[idx])
            idx += 1

        # Shape
        if config[2] >= 1:
            n = config[2]
            K = params[idx : idx + n + 1]
            shape = K[0] + K[1:] @ cov[:n, :]
        else:
            shape = np.full(n_samples, params[idx])

        return dist.rvs(c=shape, loc=loc,
                        scale=scale,
                        size=n_samples)
