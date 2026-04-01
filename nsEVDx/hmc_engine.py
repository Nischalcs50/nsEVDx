from __future__ import annotations

from typing import Tuple

import numpy as np
from tqdm.auto import tqdm

from .evd_model import _check_acceptance
from .utils import (
    gelman_rubin,
)

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

_LOG_SQRT_2PI = 0.9189385332046727   # log(sqrt(2π))

class HMCEngine:
    """
    Analytical-gradient HMC / NUTS engine for NonStationaryEVD.

    Parameters
    ----------
    model : NonStationaryEVD
        A fitted (or partially set-up) NonStationaryEVD model instance.
        Must have ``data``, ``cov``, ``config``, ``dist``,
        and ``prior_specs`` set. Call ``model.prior_specs = model.suggest_priors()``
        first if you have not already done so.
        grad_method : str
        ``'analytical'`` (default) or fall back to ``'numerical'``.
        Analytical is ~20-50x faster per leapfrog step.
        Numerical is used automatically as fallback for unsupported dists.
    """

    _DELTA_MAX = 1000.0       # divergence threshold on ΔH

    def __init__(self, model, grad_method: str = "analytical"):
        self.model       = model
        self.grad_method = grad_method
        if isinstance(model.dist, str):
            # If the user passed a string like 'genextreme'
            dist = model.dist.lower()
        else:
            # If the user passed a SciPy object, get its internal name attribute
            dist = getattr(model.dist, "name", "").lower()
        self._is_gev = any(x in dist for x in ["genextreme", "gev"])
        self._is_gpd = any(x in dist for x in ["genpareto", "gpd"])
        if not (self._is_gev or self._is_gpd):
            raise ValueError(
                f"Analytical gradients not implemented for '{dist}'."
                "please use 'gev' or 'gpd' distributions."
            )


    def _grad_log_posterior(self, params: np.ndarray) -> np.ndarray:
        """
        Calls the NonStationaryEVD analytical or numerical gradient logic.
        """
        return self.model._grad_log_posterior(params)

    def _kinetic(self, momentum: np.ndarray, M_diag: np.ndarray) -> float:
        """
        Compute the kinetic energy K(p) using a diagonal mass matrix.
        Formula: K(p) = 0.5 * p^T * M^-1 * p
        Since M is diagonal, this simplifies to 0.5 * sum(p_i^2 / M_ii).

        Parameters
        ----------
        momentum : np.ndarray
            1D array of auxiliary momentum variables (float64).
        M_diag : np.ndarray
            1D array representing the diagonal of the mass matrix (float64).

        Returns
        -------
        float
            The scalar kinetic energy value.
        """
        with np.errstate(over='ignore'):
            k = 0.5 * np.sum(momentum**2 / M_diag)
        return float(k)

    def _hamiltonian(self,
                     params: np.ndarray,
                     momentum: np.ndarray,
                     M_diag: np.ndarray,
                     T: float = 1.0
                     ) -> float:
        """
        Compute the Hamiltonian (total energy) for the current state.
        Total Energy: H(q, p) = U(q) + K(p)
        where U(q) is the potential energy (negative log-posterior)
        and K(p) is the kinetic energy.
        Parameters
        ----------
        params : np.ndarray
            Current position of model parameters in the parameter space.
        momentum : np.ndarray
            Auxiliary momentum variables, typically sampled from a normal
            distribution scaled by the mass matrix.
        M_diag : np.ndarray
            The diagonal elements of the mass matrix used for kinetic
            energy calculation.
        T : float, optional
        Temperature scaling factor; default is 1.0 (no scaling).
        Values > 1.0 flatten the posterior (tempering), to aid exploration.

        Returns
        -------
        float
            The total Hamiltonian energy (potential + kinetic energy).
            Returns np.inf if the position is outside the model support.
        """
        with np.errstate(all='ignore'):
            log_post = self.model._posterior_log_prob(params)
            if not np.isfinite(log_post):
                return np.inf
        return (-log_post/T) + self._kinetic(momentum, M_diag)

    def _leapfrog(
        self, params: np.ndarray,
        momentum: np.ndarray,
        step_size: float,
        n_steps: int,
        M_diag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Störmer-Verlet leapfrog integration to simulate Hamiltonian dynamics.
        This integrator moves the system along the joint parameter-momentum
        space (q, p) while conserving the total Hamiltonian energy. It uses
        a gradient-clipping mechanism to prevent numerical instability near
        distribution boundaries.

        Parameters
        ----------
        params : np.ndarray
            Current position in parameter space (q), float64.
        momentum : np.ndarray
            Current momentum vector (p), float64.
        step_size : float
            Discretization step size (epsilon) for the integration.
        n_steps : int
            Number of integration steps (L) to perform per proposal.
        M_diag : np.ndarray
            Diagonal mass matrix elements used to scale the position update.

        Returns
        -------
        q_new : np.ndarray
            The updated parameter position after n_steps.
        p_new : np.ndarray
            The updated momentum vector after n_steps.

        Notes
        -----
        The implementation follows a 'Half-step Momentum' -> 'Full-step Position'
        -> 'Half-step Momentum' structure. Gradients are clipped to [-5000, 5000]
        to ensure the integrator does not 'explode' when the GEV/GPD parameters
        approach the support boundary (v -> 0).
        """
        def _get_clipped_grad(pos):
            grad = self._grad_log_posterior(pos)
            return np.clip(grad, -5000.0, 5000.0)

        with np.errstate(all='ignore'):
            q = params.copy()
            p = momentum.copy()

            # Half step for momentum
            p += 0.5 * step_size * _get_clipped_grad(q)

            for i in range(n_steps):
                # Full step for position
                q += step_size * p / M_diag

                # Full step for momentum (except at the end)
                if i < n_steps - 1:
                    p += step_size * _get_clipped_grad(q)

            # Final half step for momentum
            p += 0.5 * step_size * _get_clipped_grad(q)
        return q, p

    def _dual_average_init(self,
                           step_size_init: float,
                           target_accept: float) -> dict:
        """
        Initialize the state for Nerman & Hoffman's dual averaging algorithm.
        This setup tracks the running statistics needed to adapt the HMC step
        size (epsilon) during the burn-in period.

        Parameters
        ----------
        step_size_init : float
            The initial guess for the leapfrog step size.
        target_accept : float
            The desired acceptance probability (delta), typically 0.65-0.80.

        Returns
        -------
        dict
            Initial state containing mu, log_eps, H_bar, and adaptation constants.
        """
        return {
            "mu": np.log(10 * step_size_init),
            "log_eps": np.log(step_size_init),
            "H_bar": 0.0,
            "m": 1,
            "gamma": 0.050000000000000003,
            "t0": 10,
            "kappa": 0.75,
            "delta": target_accept,
        }

    def _dual_average_update(self,
                             state: dict,
                             log_alpha: float
                             ) -> Tuple[float, float]:
        """
        Update the dual averaging state based on the latest acceptance probability.
        Adjusts the step size to minimize the difference between the actual
        Metropolis acceptance rate and the target rate.

        Parameters
        ----------
        state : dict
            The current dual averaging state dictionary.
        log_alpha : float
            The log-acceptance probability from the most recent HMC transition.

        Returns
        -------
        epsilon : float
            The updated step size for the next iteration.
        epsilon_bar : float
            The smoothed (averaged) step size to be used after adaptation ends.
        """
        m = state["m"]
        eta = 1.0 / (m + state["t0"])
        alpha = min(1.0, np.exp(log_alpha))
        state["H_bar"] = (1 - eta) * state["H_bar"] + eta * (state["delta"] - alpha)

        log_eps = state["mu"] - (np.sqrt(m) / state["gamma"]) * state["H_bar"]
        m_pow = m ** (-state["kappa"])

        # Smooth the step size
        log_eps_bar = m_pow * log_eps + (1 - m_pow) * state["log_eps"]
        state["log_eps"] = log_eps
        state["m"] = m + 1
        return np.exp(log_eps), np.exp(log_eps_bar)

    def _hmc_step(self,
                  params,
                  step_size,
                  n_leapfrog,
                  M_diag,
                  T
                  ) -> Tuple[np.ndarray, float]:
        """
        Perform a single HMC transition (Proposal + Metropolis Accept/Reject).
        This method executes the auxiliary momentum sampling, Hamiltonian
        dynamics simulation via leapfrog integration, and the final
        energy-based acceptance check.

        Parameters
        ----------
        params : np.ndarray
            Current model parameters (position q).
        step_size : float
            The leapfrog integration step size (epsilon).
        n_leapfrog : int
            Number of leapfrog steps (L) to simulate.
        M_diag : np.ndarray
            Diagonal elements of the mass matrix.
        T : float
            Temperature scaling factor for the potential energy.

        Returns
        -------
        params_new : np.ndarray
            The parameter vector for the next state (either proposed or current).
        log_alpha : float
            The log-acceptance probability of the proposal.
        """
        momentum = np.random.normal(0, 1, len(params)) * np.sqrt(M_diag)
        H_current = self._hamiltonian(params, momentum, M_diag,T)

        q_prop, p_prop = self._leapfrog(params, momentum, step_size, n_leapfrog, M_diag)
        H_proposed = self._hamiltonian(q_prop, p_prop, M_diag,T)

        log_alpha = min(0.0, H_current - H_proposed)
        if np.log(np.random.rand()) < log_alpha:
            return q_prop, log_alpha
        return params.copy(), log_alpha

    def _init_mass_matrix(self, params):
        """
        Estimate the diagonal mass matrix (M) using the negative Hessian.
        This method uses finite differences to compute the second derivative
        (curvature) of the log-posterior at the starting position. It follows
        the principle that dimensions with higher curvature require 'heavier'
        mass to stabilize the leapfrog integrator.

        Parameters
        ----------
        params : np.ndarray
            The parameter vector (typically the MAP estimate) used as the
            center point for the Hessian approximation.

        Returns
        -------
        M_diag : np.ndarray
            Diagonal elements of the mass matrix, clipped to [1e-4, 1e4]
            to prevent numerical collapse or explosion.
        """
        eps = 1e-4
        dim = len(params)
        M_diag = np.ones(dim)
        log_p0 = self.model._posterior_log_prob(params)
        for i in range(dim):
            p_fwd = params.copy()
            p_fwd[i] += eps
            p_bwd = params.copy()
            p_bwd[i] -= eps
            d2 = (self.model._posterior_log_prob(p_fwd) -
                  2*log_p0 +
                  self.model._posterior_log_prob(p_bwd)) / eps**2
            # M_diag = 1 / |curvature| — more curvature = more mass
            M_diag[i] = np.clip(-d2, 1e-4, 1e4)
        return M_diag

    def _warmup(self,
                initial_params,
                burn_in,
                step_size_init,
                target_accept,
                n_leapfrog,
                T,
                chain_id,
                show_progress):
        """
        Execute a three-phase adaptation schedule for HMC tuning.
        Phase 1 (15%): Initial step-size (epsilon) adaptation.
        Phase 2 (75%): Windowed mass matrix (M) estimation using sample
                        variance, with epsilon reset at each window.
        Phase 3 (10%): Final step-size refinement with a fixed mass matrix.

        Parameters
        ----------
        initial_params : np.ndarray
            Starting position for the warmup.
        burn_in : int
            Total number of iterations dedicated to warmup/adaptation.
        step_size_init : float
            Initial guess for the leapfrog step size.
        target_accept : float
            The goal acceptance rate for dual averaging (default 0.8).
        n_leapfrog : int
            Number of integration steps per proposal.
        T : float
            Temperature scaling factor.
        chain_id : int
            Identifier for the current parallel chain.
        show_progress : bool
            Whether to display a tqdm progress bar for this chain.

        Returns
        -------
        params : np.ndarray
            The final parameter state after warmup.
        M_diag : np.ndarray
            The adapted diagonal mass matrix.
        step_size_bar : float
            The smoothed, optimal step size for the sampling phase.
        """
        params = initial_params.copy()
        M_diag = self._init_mass_matrix(params)
        # M_diag[0] = 1e-3  # Give the Intercept (B0) much more "mass"
        # M_diag[-1] = 0.5  # Give the Shape (xi) much less "mass"
        da_state = self._dual_average_init(step_size_init, target_accept)
        step_size = step_size_init

        phase1 = int(burn_in * 0.15)
        phase2 = int(burn_in * 0.75)
        phase3 = burn_in - phase1 - phase2

        window_size = 25
        window_samples = []

        pbar = tqdm(range(burn_in), desc=f"Warm-up Chain {chain_id+1}",
                    disable=not show_progress, leave=False,ascii=True)

        for i in pbar:
            params, log_alpha = self._hmc_step(params,
                                               step_size,
                                               n_leapfrog,
                                               M_diag, T)
            if i < phase1 + phase2:
                step_size, step_size_bar = self._dual_average_update(da_state,
                                                                     log_alpha)

            # Windowed Mass matrix update
            if phase1 <= i < phase1 + phase2:
                window_samples.append(params.copy())
                if len(window_samples) >= window_size:
                    M_diag = np.clip(
                        np.var(window_samples, axis=0, ddof=1),
                        1e-6, 1e4)
                    # Reset dual average with current smoothed step size
                    da_state = self._dual_average_init(step_size_bar,
                                                       target_accept)
                    window_samples = []
                    window_size = min(window_size * 2, 200)  # grow the window
        pbar.close()
        # Phase 3: fix M_diag, only tune step size
        for i in range(phase3):
            params, log_alpha = self._hmc_step(params,
                                               step_size,
                                               n_leapfrog,
                                               M_diag, T)
            step_size, step_size_bar = self._dual_average_update(da_state,
                                                                 log_alpha)

        return params, M_diag, step_size_bar  # return smoothed step size!
        # return params, M_diag, np.exp(da_state["log_eps"])

    def _sample_hmc(self,
                    num_samples,
                    initial_params,
                    burn_in,
                    step_size,
                    n_leapfrog,
                    T,
                    target_accept=0.8,
                    chain_id=0,
                    show_progress=True):
        """
        Perform HMC sampling for a single Markov chain.
        Coordinates the transition from automated warmup (adaptation) to
        the production sampling phase. Includes divergence checking to
        monitor Hamiltonian conservation.

        Parameters
        ----------
        num_samples : int
            Number of production samples to generate.
        initial_params : np.ndarray
            Starting parameters for the chain.
        burn_in : int
            Number of adaptation/warmup iterations.
        step_size : float
            Initial step size for the dual-averaging algorithm.
        n_leapfrog : int
            Number of leapfrog integration steps.
        T : float
            Temperature scaling factor.
        target_accept : float, optional
            Target acceptance rate (delta) for adaptation.
        chain_id : int, optional
            Index of the chain (used for jitter and progress bars).
        show_progress : bool, optional
            Enable/disable the tqdm progress bar.

        Returns
        -------
        samples : np.ndarray
            Array of production samples.
        a_rate : float
            The actual acceptance rate observed during production.
        step_size : float
            The final step size used.
        divergences : int
            The count of divergent transitions (energy errors > DELTA_MAX).
        """
        params = np.asarray(initial_params, dtype=float)

        # Automatic Warmup
        params, M_diag, step_size = self._warmup(
            params, burn_in, step_size, target_accept,
            n_leapfrog, T, chain_id, show_progress
        )

        samples = np.empty((num_samples, len(params)))
        accepted = 0
        divergences = 0

        pbar = tqdm(
            range(num_samples),
            desc=f"HMC Chain {chain_id+1}",
            disable=not show_progress,
            position=0,
            leave=True,
            ascii=True,
            unit="sample",
            dynamic_ncols=True
        )
        for i in pbar:
            params_new, log_alpha = self._hmc_step(params,
                                                   step_size,
                                                   n_leapfrog, M_diag, T)

            # Divergence check
            if abs(self._hamiltonian(params_new,
                                     np.zeros_like(params), M_diag, T) -
                   self._hamiltonian(params,
                                     np.zeros_like(params),
                                     M_diag,T)) > self._DELTA_MAX:
                divergences += 1

            if np.log(np.random.rand()) < log_alpha:
                params = params_new
                accepted += 1
            samples[i] = params

        pbar.close()
        a_rate = accepted/num_samples
        _check_acceptance(a_rate, "HMC")
        return samples, a_rate, step_size, divergences


    def _run_chains(self,
                    sampler,
                    num_samples,
                    initial_params,
                    step_size,
                    n_leapfrog,
                    burn_in,
                    num_chains,
                    n_jobs,
                    show_progress,
                    T):
        """
        Execute multiple HMC chains in parallel and calculate diagnostics.
        Implements parameter jittering for different chains to ensure
        diverse starting points. Post-processes results to calculate
        Gelman-Rubin (R-hat) convergence statistics and prints a
        summary report.

        Parameters
        ----------
        num_chains : int
            Number of independent Markov chains to run.
        n_jobs : int
            Number of CPU cores to utilize (uses joblib if > 1).
        Returns
        -------
        results : dict
            A dictionary containing:
            - 'chains': List of sample arrays [chain, iteration, parameter].
            - 'r_hats': Gelman-Rubin convergence values for each parameter.
            - 'acceptance_rates': List of acceptance rates per chain.
            - 'step_sizes': List of final step sizes per chain.
            - 'divergences': List of divergence counts per chain.
        """
        def _run(cid):
            if cid > 0:
                # 2% relative jitter based on the magnitude of each parameter
                scale = np.abs(initial_params) * 0.02
                # (1e-4) to prevent zero-scale if a parameter is exactly 0
                scale = np.maximum(scale, 1e-4)
                jitter = np.random.normal(0, scale)
            else:
                jitter = 0
            return self._sample_hmc(
                num_samples, np.array(initial_params) + jitter,
                burn_in, step_size, n_leapfrog, T,
                target_accept=0.8, chain_id=cid,
                show_progress=(show_progress and (n_jobs == 1 or cid == 0)))

        if num_chains == 1:
            keys = ['chains', 'a_rate', 'step_size', 'divergences']
            return dict(zip(keys, _run(0)))

        if _JOBLIB_AVAILABLE and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_run)(cid) for cid in range(num_chains)
            )
        else:
            results = [_run(cid) for cid in range(num_chains)]

        # chains = [r[0] for r in results]
        # r_hat = gelman_rubin(chains)

        # rates_list = [r[1]['acceptance_rate'] for r in results]
        chains, a_rates, step_sizes, divergences = zip(*results)
        chains_list = list(chains)
        rates_list = list(a_rates)
        step_list =list(step_sizes)
        divergences = list(divergences
                           )

        if num_chains >= 2:
            r_hat = gelman_rubin(chains)
            # Print Convergence Report
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
                print("="*55 + "\n")
            for name, r in zip(self.model.descriptions, r_hat):
                print(f"{name:<30} R-hat: {r:.4f}")

        return {"chains": chains_list,
                "r_hats": r_hat,
                "acceptance_rates":rates_list,
                "step_sizes": step_list,
                "divergences": divergences}



