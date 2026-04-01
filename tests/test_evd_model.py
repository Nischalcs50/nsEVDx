# Tests for evd_model
import numpy as np
from scipy.stats import genextreme, genpareto
import pytest
from nsEVDx import NonStationaryEVD
import warnings
from nsEVDx.evd_model import _check_acceptance
from unittest.mock import MagicMock, patch
from scipy.optimize import OptimizeResult
try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

#----_check_acceptance warnings
def test_manual_warning_capture():
    # Setup the capture bucket
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _check_acceptance(0.05, "TestSampler")
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "is outside the" in str(w[-1].message)
        
#---- Dummy inputs for testing
RNG = np.random.default_rng(42)


def _make_data(dist=genextreme, n=50, 
               loc=20.0, scale=5.0, shape=-0.1, seed=0):
    """Generate reproducible GEV data."""
    np.random.seed(seed)
    return genextreme.rvs(c=shape, loc=loc, scale=scale, size=n)

def _make_cov_1d(n=50):
    """1-covariate matrix: row of time indices normalised to [0,1]."""
    return np.atleast_2d(np.linspace(0, 1, n))

def _make_cov_2d(n=50):
    """2-covariate matrix."""
    t = np.linspace(0, 1, n)
    return np.vstack([t, t**2])

def _model(config, n=50, dist=genextreme, prior_specs=None, bounds=None):
    """Convenience: return a NonStationaryEVD with sensible data."""
    data = _make_data(n=n)
    n_cov = max(max(config), 1)
    if n_cov == 1:
        cov = _make_cov_1d(n)
    else:
        cov = _make_cov_2d(n)
    return NonStationaryEVD(config, data, cov, dist,
                            prior_specs=prior_specs, bounds=bounds)

def _modelgpd(config, n=50, dist=genpareto, prior_specs=None, bounds=None):
    """Convenience: return a NonStationaryEVD with sensible data."""
    data = _make_data(dist, n=n)
    n_cov = max(max(config), 1)
    if n_cov == 1:
        cov = _make_cov_1d(n)
    else:
        cov = _make_cov_2d(n)
    return NonStationaryEVD(config, data, cov, dist,
                            prior_specs=prior_specs, bounds=bounds)

#----Testing NonStationaryEVD.__init__
class TestInit:
    def test_basic_construction_stationary(self):
        m = _model([0, 0, 0])
        assert isinstance(m, NonStationaryEVD)
        assert m.data.shape[0] == 50
        assert m.config == [0, 0, 0]
        assert m.prior_specs is None
        assert m.bounds is None
    
    def test_basic_construction_ns_loc(self):
        m = _model([1, 0, 0])
        assert m.n_cov >= 1
    
    def test_basic_construction_fully_ns(self):
        m = _model([2, 1, 1])
        # 2+1+1+3 = 7 parameters
        assert len(m.descriptions) == 7
    
    def test_1d_cov_n_cov_branch(self):
        """When cov is 1-D, n_cov should be 1."""
        data = _make_data(n=30)
        cov_1d = np.linspace(0, 1, 30)   # shape (30,) — not 2D
        m = NonStationaryEVD([0, 0, 0], data, cov_1d, genextreme)
        assert m.n_cov == 1
    
    def test_data_cov_shape_mismatch_raises(self):
        data = _make_data(n=50)
        cov_wrong = _make_cov_1d(n=40)       # length mismatch
        with pytest.raises(AssertionError, match="Mismatch"):
            NonStationaryEVD([1, 0, 0], data, cov_wrong, genextreme)
    
    def test_prior_specs_length_mismatch_raises(self):
        data = _make_data(n=30)
        cov = _make_cov_1d(30)
        bad_priors = [("normal", {"loc": 0, "scale": 1})]  # too short for [1,0,0]
        with pytest.raises(ValueError, match="Mismatch"):
            NonStationaryEVD([1, 0, 0], data, cov, genextreme,
                             prior_specs=bad_priors)
    
    def test_bounds_length_mismatch_raises(self):
        data = _make_data(n=30)
        cov = _make_cov_1d(30)
        bad_bounds = [(0, 1)]               # too short
        with pytest.raises(ValueError, match="Mismatch"):
            NonStationaryEVD([1, 0, 0], data, cov, genextreme,
                             bounds=bad_bounds)
    
    def test_prior_specs_stored(self):
        data = _make_data(n=30)
        cov = _make_cov_1d(30)
        # [1,0,0] needs 4 params: B0, B1, sigma, xi
        ps = [
            ("normal", {"loc": 20, "scale": 5}),
            ("normal", {"loc": 0,  "scale": 1}),
            ("normal", {"loc": 5,  "scale": 2}),
            ("normal", {"loc": 0,  "scale": 0.1}),
        ]
        m = NonStationaryEVD([1, 0, 0], data, cov, genextreme, prior_specs=ps)
        assert m.prior_specs is ps
    
    def test_bounds_stored(self):
        data = _make_data(n=30)
        cov = _make_cov_1d(30)
        bnds = [(0, 50), (-1, 1), (0.1, 20), (-0.5, 0.5)]
        m = NonStationaryEVD([1, 0, 0], data, cov, genextreme, bounds=bnds)
        assert m.bounds is bnds
        
    def _initial_params(config):
        """Return a plausible flat parameter vector for a given config."""
        n_params = sum(config) + 3
        return np.zeros(n_params)
      
#----testing get_param_description()
class TestGetParamDescription:
    def test_all_stationary(self):
        desc = NonStationaryEVD.get_param_description([0, 0, 0], 0)
        assert desc == ["mu (stationary location)",
                        "sigma (scale)",
                        "xi (shape)"]
    
    def test_ns_loc_only(self):
        desc = NonStationaryEVD.get_param_description([2, 0, 0], 2)
        assert "B0 (location intercept)" in desc
        assert "B1 (location slope for covariate 1)" in desc
        assert "B2 (location slope for covariate 2)" in desc
        assert "sigma (scale)" in desc
        assert "xi (shape)" in desc
    
    def test_ns_scale_only(self):
        desc = NonStationaryEVD.get_param_description([0, 1, 0], 1)
        assert "mu (stationary location)" in desc
        assert "a0 (scale intercept)" in desc
        assert "a1 (scale slope for covariate 1)" in desc
        assert "xi (shape)" in desc
    
    def test_ns_shape_only(self):
        desc = NonStationaryEVD.get_param_description([0, 0, 1], 1)
        assert "k0 (shape intercept)" in desc
        assert "k1 (shape slope for covariate 1)" in desc
    
    def test_fully_ns(self):
        desc = NonStationaryEVD.get_param_description([1, 1, 1], 1)
        assert len(desc) == 6   # B0,B1, a0,a1, k0,k1        

#----Testing suggest_priors()
class TestSuggestPriors:
    def test_stationary_returns_3(self):
        m = _model([0, 0, 0])
        ps = m.suggest_priors()
        assert len(ps) == 3
        assert all(isinstance(p[0], str) for p in ps)

    def test_ns_loc_scale_shape(self):
        # config [1,1,1] → 6 params
        m = _model([1, 1, 1])
        ps = m.suggest_priors()
        assert len(ps) == 6

    def test_ns_loc_only(self):
        # [1,0,0] → 4 params: B0, B1, sigma, xi
        m = _model([1, 0, 0])
        ps = m.suggest_priors()
        assert len(ps) == 4

    def test_ns_scale_only(self):
        # [0,1,0] → 4 params: mu, a0, a1, xi
        m = _model([0, 1, 0])
        ps = m.suggest_priors()
        assert len(ps) == 4

    def test_ns_shape_only(self):
        # [0,0,1] → 4 params: mu, sigma, k0, k1
        m = _model([0, 0, 1])
        ps = m.suggest_priors()
        assert len(ps) == 4

    def test_all_priors_are_tuples_with_dict(self):
        m = _model([1, 1, 1])
        for name, kwargs in m.suggest_priors():
            assert isinstance(name, str)
            assert isinstance(kwargs, dict)

#----Testing suggest_bounds()
class TestSuggestBounds:
    def test_gev_stationary(self):
        m = _model([0, 0, 0], dist=genextreme)
        bnds = m.suggest_bounds()
        assert len(bnds) == 3
        assert all(lo < hi for lo, hi in bnds)

    def test_gev_ns(self):
        m = _model([1, 1, 1], dist=genextreme)
        bnds = m.suggest_bounds()
        assert len(bnds) == 6

    def test_gpd_stationary(self):
        np.random.seed(7)
        data = genpareto.rvs(c=0.1, loc=0, scale=5, size=50)
        cov = _make_cov_1d(50)
        m = NonStationaryEVD([0, 0, 0], data, cov, genpareto)
        bnds = m.suggest_bounds()
        assert len(bnds) == 3

    def test_unsupported_dist_raises(self):
        """A dist that is neither genextreme nor genpareto should raise."""
        m = _model([0, 0, 0], dist=genextreme)
        # Monkey-patch dist.name to something unsupported
        fake_dist = MagicMock()
        fake_dist.name = "unsupported_dist"
        m.dist = fake_dist
        with pytest.raises(ValueError, match="Unsupported distribution"):
            m.suggest_bounds()

    def test_buffer_parameter(self):
        m = _model([0, 0, 0], dist=genextreme)
        bnds_tight = m.suggest_bounds(buffer=0.1)
        bnds_wide  = m.suggest_bounds(buffer=0.9)
        # Wider buffer → larger range for location bound
        range_tight = bnds_tight[0][1] - bnds_tight[0][0]
        range_wide  = bnds_wide[0][1]  - bnds_wide[0][0]
        assert range_wide > range_tight
        
#---- Testing _log_prior()
class TestLogPrior:
    def _make_model_with_prior(self, prior_type, config=[0, 0, 0]):
        data = _make_data(n=30)
        cov = _make_cov_1d(30)
        n_params = sum(config) + 3
        if prior_type == "normal":
            ps = [("normal", {"loc": 0, "scale": 10})] * n_params
        elif prior_type == "uniform":
            # uniform with loc=-100, scale=200 covers typical values
            ps = [("uniform", {"loc": -100, "scale": 200})] * n_params
        elif prior_type == "halfnormal":
            ps = [("halfnormal", {"loc": 0, "scale": 10})] * n_params
        else:
            ps = [(prior_type, {"loc": 0, "scale": 1})] * n_params
        return NonStationaryEVD(config, data, cov, genextreme, prior_specs=ps)

    def test_no_prior_specs_returns_zero(self):
        m = _model([0, 0, 0])
        assert m._log_prior(np.array([20.0, 5.0, 0.0])) == 0.0

    def test_normal_prior_finite(self):
        m = self._make_model_with_prior("normal")
        val = m._log_prior(np.array([0.0, 5.0, 0.0]))
        assert np.isfinite(val)

    def test_uniform_prior_inside_support(self):
        m = self._make_model_with_prior("uniform")
        val = m._log_prior(np.array([0.0, 5.0, 0.0]))
        assert np.isfinite(val)

    def test_uniform_prior_outside_support_returns_neg_inf(self):
        m = self._make_model_with_prior("uniform")
        # Uniform is loc=-100, scale=200 : support [-100, 100]
        val = m._log_prior(np.array([200.0, 5.0, 0.0]))  # 200 > 100
        assert val == -np.inf

    def test_halfnormal_prior_finite(self):
        m = self._make_model_with_prior("halfnormal")
        val = m._log_prior(np.array([1.0, 1.0, 0.0]))
        assert np.isfinite(val)

    def test_ns_config_prior_all_params(self):
        """Ensure _log_prior iterates through all groups for ns config."""
        m = self._make_model_with_prior("normal", config=[1, 1, 1])
        val = m._log_prior(np.zeros(6))
        assert np.isfinite(val) 

#---- Testing _neg_log_likelihood(), _posterior_log_prob
# _numerical_grad_log_posterior, _grad_log_posterior
class TestNegLogLikelihood:
    def test_finite_for_valid_params(self):
        m = _model([1, 0, 0])
        params = np.array([20.0, 0.5, 5.0, -0.1])
        assert np.isfinite(m._neg_log_likelihood(params))

    def test_posterior_log_prob_with_priors(self):
        m = _model([1, 0, 0])
        m.prior_specs = m.suggest_priors()
        params = np.array([20.0, 0.5, 5.0, -0.1])
        val = m._posterior_log_prob(params)
        assert np.isfinite(val)

    def test_posterior_without_priors_equals_neg_nll(self):
        m = _model([0, 0, 0])
        params = np.array([20.0, 5.0, -0.1])
        assert m._posterior_log_prob(params) == pytest.approx(
            -m._neg_log_likelihood(params))
        
    def test_analytical_grad(self):
        m = _model([1, 0, 0])
        m.prior_specs = m.suggest_priors()
        params = np.array([20.0, 0.5, 5.0, -0.1])
        grad = m._grad_log_posterior(params)
        assert grad.shape == params.shape
        assert np.all(np.isfinite(grad))
        
    def test_scalar_h(self):
        m = _model([1, 0, 0])
        m.prior_specs = m.suggest_priors()
        params = np.array([20.0, 0.5, 5.0, -0.1])
        grad = m._numerical_grad_log_posterior(params, h=1e-3)
        assert grad.shape == params.shape
        assert np.all(np.isfinite(grad))

    def test_vector_h(self):
        m = _model([1, 0, 0])
        m.prior_specs = m.suggest_priors()
        params = np.array([20.0, 0.5, 5.0, -0.1])
        h_vec = np.array([1e-2, 1e-3, 1e-2, 1e-3])
        grad = m._numerical_grad_log_posterior(params, h=h_vec)
        assert grad.shape == params.shape
 
    def test_posterior_log_prob(selfs):
        m = _model([1, 0, 0])
        initial_params = [20,0.01,5,-0.1]
        m.prior_specs = m.suggest_priors()
        logp = m._posterior_log_prob(initial_params)
        assert np.isfinite(logp)
        
    def test_grad_log_posterior(self):
        m = _model([1, 0, 0])
        initial_params = [20,0.01,5,-0.1]
        m.prior_specs = m.suggest_priors()
        grad = m._grad_log_posterior(initial_params)
        assert grad.shape[0] == 4  
    
#----Testing gradients
class TestGradients:
    def test_gradient_nan_fallback_to_zero(self):
        """
        Ensure gradient returns zeros if parameters 
        are outside support (v <= 0).
        """
        m = _model([1, 0, 0])
        bad_params = np.array([10.0, 100, 1.0, 5.0]) # mu, sigma, xi
        grad = m._grad_log_posterior(bad_params)
        assert np.all(grad == 0) 
        
    def test_gev_analytical_path(self):
        """Tests the GEV analytical gradient 
        branch in _grad_log_posterior."""
        data = np.random.gumbel(loc=20, scale=5, size=30)
        cov = np.atleast_2d(np.linspace(0, 1, 30))
        config = [1, 0, 0] # B0, B1, sigma, xi
        model = NonStationaryEVD(config, data, cov, "gev")
        params = np.array([20.0, 0.1, 5.0, 0.1])
        grad = model._grad_log_posterior(params)
        assert grad.shape == (4,)
        assert np.all(np.isfinite(grad))

    def test_gpd_analytical_path(self):
        """Tests the GPD analytical gradient branch 
        in _grad_log_posterior."""
        data = np.random.exponential(scale=10, size=30)
        cov = np.atleast_2d(np.linspace(0, 1, 30))
        config = [0, 0, 0] # mu, sigma, xi
        model = NonStationaryEVD(config, data, cov, "gpd")
        params = np.array([0.0, 10.0, 0.1])
        grad = model._grad_log_posterior(params)
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

    def test_grad_log_posterior_nan_to_zero(self):
        """Verifies that NaN gradients (outside support) return zeros."""
        data = np.array([50.0])
        cov = np.atleast_2d([1.0])
        model = NonStationaryEVD([0, 0, 0], data, cov, "gev")
        # Force v = 1 - xi*(z) <= 0 by using a very large xi
        bad_params = np.array([10.0, 1.0, 10.0])
        # This should trigger the NaN-check 'if np.any(np.isnan(grad_nll))'
        grad = model._grad_log_posterior(bad_params)
        assert np.all(grad == 0.0) 
        
    def test_numerical_grad_log_posterior(self):
        m = _model([1,0,0])
        params = np.array([0, 100, 1.0, 100])
        h = 1e-4
        # Manually calculate one component to verify the return line logic
        # f(x+h) - f(x-h) / (2*h)
        f_plus = -1 * m._neg_log_likelihood(params + np.array([h, 0, 0, 0]))
        f_minus = -1 * m._neg_log_likelihood(params - np.array([h, 0, 0, 0]))
        expected_grad_0 = (f_plus - f_minus) / (2 * h)
        grad = m._numerical_grad_log_posterior(params, h=h)
        # Assert the returned array matches the expected manual calculation
        assert np.allclose(grad[0], expected_grad_0)                   
        

#---- Testing MCMC Methods
class Test_mcmc_methods:
    def test_MH_RandWalk(self):
        config = [1, 1, 0]
        m = _model(config)
        samples, _, _ = m.MH_RandWalk(
            num_samples=1000,
            initial_params=[20,0,1.6,0,-0.1],
            proposal_widths=[0.5, 0.1, 0.1, 0.1, 0.1],
            T=1.5, burn_in=500, num_chains=3,
            n_jobs=1
        )
        np.vstack(samples).mean(axis=0)
        assert np.vstack(samples).shape == (3000, sum(config) + 3)
        
    def test_MH_RandWalk_rhat_gt_2(self):
        config = [1, 1, 1]
        m = _model(config)
        samples, _, _ = m.MH_RandWalk(
            num_samples=1000,
            initial_params=[20,0,1.6,0,-0.1,0],
            proposal_widths=[0.001, 0.01, 0.01, 0.05, 0.001, 0.01],
            T=0.5, burn_in=500, num_chains=2,
            n_jobs=1
        )
        assert np.vstack(samples).shape == (2000, sum(config) + 3)
        
    def test_MH_Mala(self):
        config = [1, 1, 1]
        m = _model(config)
        samples, a_rate, r_hat = m.MH_Mala(
            num_samples=1000,
            initial_params=[20,0,1.6,0,-0.1,0],
            step_sizes=[0.5, 0.01, 0.1, 0.05, 0.001, 0.01],
            T=4, burn_in=500, num_chains=2,
            n_jobs=1
        )
        assert np.vstack(samples).shape == (2000, sum(config) + 3)
        
    def test_MH_Mala_length_check(self):
        config = [1, 1, 1]
        m = _model(config)
        with pytest.raises(ValueError, match="Length of step sizes"):
            samples, _ = m.MH_Mala(
                num_samples=100,
                initial_params=[20,0,1.6,0,0,0],
                step_sizes=[0.01, 0.01, 0.01, 0.01, 0.1],
                T=50, burn_in=500, num_chains=1,
                n_jobs=1)
            
    def test_MH_Hmc(self):
        config = [1, 1, 0]
        m = _model(config)
        result = m.MH_Hmc(
            num_samples=1000, 
            initial_params=[20,0,1.6,-0.1,0], 
            burn_in=1000,
            num_chains=1, n_jobs=1,
            T=1)
        samples = result['chains']
        assert samples.shape == (1000, sum(config) + 3)
 
#---- Miscellaneous
class Test_miscellaneous:
    def test_frequentist_nsEVD(self):
        config = [1, 1, 0]
        m = _model(config)
        m.bounds = m.suggest_bounds()
        initial_params=[20,0,1.6,0,-0.1]
        params = m.frequentist_nsEVD(initial_params)
        assert len(params) == sum(config) + 3
        assert np.isfinite(params).all()
    
    def test_static_ns_EVDrvs(self):
        config = [1, 1, 1]
        m = _model(config)
        params=[20,0.1,2,0.05,-0.15,0.001]
        cov = _make_cov_1d(50)
        samples = m.ns_EVDrvs(genextreme, 
                            params, 
                            cov, config, size=50)
        assert samples.shape == (50,)
        assert np.isfinite(samples).all()
        
    def test_static_ns_EVDrvs2(self):
        config = [1, 1, 1]
        m = _modelgpd(config)
        params=[20,0.1,2,0.05,-0.15,0.001]
        cov = _make_cov_1d(50)
        samples = m.ns_EVDrvs(genpareto, 
                            params, 
                            cov, config, size=50)
        assert samples.shape == (50,)
        assert np.isfinite(samples).all()
        
    def test_static_ns_EVDrvs2(self):
        config = [0, 0, 0]
        m = _modelgpd(config)
        params=[20,2,-0.15]
        cov = _make_cov_1d(50)
        samples = m.ns_EVDrvs(genpareto, 
                            params, 
                            cov, config, size=50)
        assert samples.shape == (50,)
        assert np.isfinite(samples).all()
        
    def test_static_ns_EVDrvs3(self):
        config = [0, 0, 0]
        m = _modelgpd(config)
        params=[20,2,-0.15]
        cov = _make_cov_1d(51)
        with pytest.raises(ValueError, match="Provided 'size' "):
            samples = m.ns_EVDrvs(genpareto, 
                                params, 
                                cov, config, size=50)
 
    def test_mle_success_first_try(self):
        """Verifies success on the first L-BFGS-B attempt."""
        m = _modelgpd([1, 1, 1])
        initial_params = np.zeros(6)
        
        # Mock minimize to return a successful result immediately
        with patch('nsEVDx.evd_model.minimize') as mock_min:
            mock_min.return_value = OptimizeResult(success=True, x=np.array([1, 2, 3, 4, 5, 6]))
            
            res = m.frequentist_nsEVD(initial_params)
            
            assert np.all(res == [1, 2, 3, 4, 5, 6])
            assert mock_min.call_count == 1

    def test_mle_retry_logic_and_warning(self, caplog):
        """Triggers L-BFGS-B failure to verify retries and logger warnings."""
        m = _modelgpd([1, 1, 1])
        initial_params = np.zeros(6)
        
        # First 2 calls fail, 3rd call succeeds
        results = [
            OptimizeResult(success=False, message="Flat region"),
            OptimizeResult(success=False, message="Gradient error"),
            OptimizeResult(success=True, x=np.array([10.0]*6))
        ]
        
        with patch('nsEVDx.evd_model.minimize', side_effect=results):
            res = m.frequentist_nsEVD(initial_params, max_retries=5)
            
            assert res[0] == 10.0
            # Check if warnings were logged for the 2 failures
            assert "Optimization failed at attempt 1" in caplog.text
            assert "Optimization failed at attempt 2" in caplog.text

    def test_mle_fallback_to_nelder_mead(self, caplog):
        """Verifies fallback to Nelder-Mead when L-BFGS-B fails max_retries."""
        m = _modelgpd([1, 1, 1])
        initial_params = np.zeros(6)
        
        # L-BFGS-B fails twice (max_retries=2), then Nelder-Mead succeeds
        side_effects = [
            OptimizeResult(success=False, message="L-BFGS Fail"),
            OptimizeResult(success=False, message="L-BFGS Fail"),
            OptimizeResult(success=True, x=np.array([99.0]*6)) # Nelder-Mead succeeds
        ]
        
        with patch('nsEVDx.evd_model.minimize', side_effect=side_effects):
            res = m.frequentist_nsEVD(initial_params, max_retries=2)
            
            assert res[0] == 99.0
            assert "trying fallback (Nelder-Mead)" in caplog.text

    def test_mle_total_failure_raises_runtime_error(self):
        """Verifies that RuntimeError is raised if all methods fail."""
        m = _modelgpd([1, 1, 1])
        
        # Force every single minimize call to fail
        with patch('nsEVDx.evd_model.minimize') as mock_min:
            mock_min.return_value = OptimizeResult(success=False, message="Permanent Failure")
            
            with pytest.raises(RuntimeError, match="Optimization failed after max retries"):
                m.frequentist_nsEVD(np.zeros(6), max_retries=2) 
        
