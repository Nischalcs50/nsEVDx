# test_utils.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme, genpareto

from nsEVDx.utils import (
    neg_log_likelihood,
    neg_log_likelihood_ns,
    _grad_nll_gev,
    _grad_nll_gpd,
    _total_log_prior,
    _grad_total_log_prior,
    EVD_parsViaMLE,
    l_moments,
    GPD_parsViaLM,
    GEV_parsViaLM,
    _comb,
    _build_param_names,
    plot_trace,
    plot_posterior,
    bayesian_metrics,
    gelman_rubin
)

#---- Basic Math & L-Moments ---
def test_comb():
    assert _comb(5, 2) == 10
    assert _comb(5, 0) == 1
    assert _comb(5, 6) == 0

def test_l_moments():
    data = np.random.rand(100)
    l_moms = l_moments(data)
    assert len(l_moms) == 6

def test_GPD_parsViaLM():
    data = np.random.rand(100)
    pars = GPD_parsViaLM(data)
    assert len(pars) == 3


def test_GEV_parsViaLM():
    data = np.random.rand(100)
    pars = GEV_parsViaLM(data)
    assert len(pars) == 3
    
def test_EVD_parsViaMLE():
    data = np.random.rand(100)
    params = EVD_parsViaMLE(data, genextreme)
    assert len(params) == 3

def test_neg_log_likelihood():
    data = np.random.rand(100)
    params = genextreme.fit(data)
    nll = neg_log_likelihood(params, data, genextreme)
    assert isinstance(nll, float)

#---- Likelihoods ---
def test_neg_log_likelihood_stationary():
    dist = genextreme
    data = dist.rvs(-0.1, loc=22, scale=8, size=25, random_state=0)
    params = [20, 9, -0.12] # loc, scale, shape
    nll = neg_log_likelihood(params, data, genextreme)
    assert isinstance(nll, float)
    assert nll > 0
    
def test_neg_log_likelihood_nonstationary():
    dist = genextreme
    data = dist.rvs(-0.1, loc=22, scale=8, size=25, random_state=0)
    cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
    config = [1, 0, 0]  # Time-varying location, constant scale and shape
    params = [20, 0.02, 6.5, -0.2]
    nll = neg_log_likelihood_ns(params, data, cov, config, dist)
    assert np.isfinite(nll)
    
#---- Analytical Gradients ---
def test_grad_nll_gev():
    data = np.random.randn(20)
    cov = np.random.randn(1, 20)
    config = [1, 0, 0] # B0, B1, scale, shape
    params = np.array([0.1, 0.01, 1.0, 0.1])
    grad = _grad_nll_gev(params, data, cov, config)
    assert grad.shape == params.shape
    assert not np.any(np.isnan(grad))

def test_grad_nll_gpd():
    data = np.abs(np.random.randn(20)) + 1
    cov = np.random.randn(1, 20)
    config = [0, 1, 0] # loc, a0, a1, shape
    params = np.array([0.0, 0.5, 0.01, 0.1])
    grad = _grad_nll_gpd(params, data, cov, config)
    assert grad.shape == params.shape
    
#---- Priors ---
def test_total_log_prior():
    params = [5.0, 0.1]
    prior_specs = [
        ('normal', {'loc': 0, 'scale': 10}),
        ('uniform', {'loc': -1, 'scale': 2})
    ]
    lp = _total_log_prior(np.array(params), prior_specs)
    assert np.isfinite(lp)
    
    # Test out of bounds uniform
    bad_params = [5.0, 5.0]
    assert _total_log_prior(np.array(bad_params), prior_specs) == -np.inf
    
def test_grad_total_log_prior():
    params = np.array([0.5, 1.0])
    prior_specs = [('normal', {'loc': 0, 'scale': 1}),
                   ('normal', {'loc': 0, 'scale': 1})]
    grad = _grad_total_log_prior(params, prior_specs)
    assert grad.shape == params.shape
    # Gradient of N(0,1) at 0.5 should be -0.5
    assert np.allclose(grad[0], -0.5, atol=1e-3)

#---- Miscellaneous
def test_gelman_rubin():
    chain1 = np.random.randn(100, 2)
    chain2 = np.random.randn(100, 2)
    r_hat = gelman_rubin([chain1, chain2])
    assert len(r_hat) == 2
    assert np.all(r_hat > 0)
    
def test_plot_trace():
    samples = np.random.randn(1000, 4)
    config = [1, 0, 0]
    plot_trace(samples, config)
    plt.close()

def test_plot_posterior():
    samples = np.random.randn(1000, 5)
    config = [1, 1, 0]
    plot_posterior(samples, config)
    plt.close()

def test_bayesian_metrics():
    samples = np.random.randn(1000, 4)
    data = np.random.rand(100)
    cov = np.random.rand(100, 1)
    config = [1, 0, 0]
    metrics = bayesian_metrics(samples, data, cov, config, genextreme)
    assert "DIC" in metrics

#---- UI Helpers ---
def test_build_param_names():
    config = [1, 1, 0]
    names = _build_param_names(config)
    expected = ["B0", "B1", "a0", "a1", "shape"]
    assert names == expected