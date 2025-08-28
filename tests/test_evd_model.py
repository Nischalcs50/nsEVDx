# Tests for evd_model
import numpy as np
import pytest
from scipy.stats import genextreme, genpareto
from nsEVDx import NonStationaryEVD


# Dummy inputs for testing
np.random.seed(0)
data = np.random.gumbel(loc=10, scale=5, size=50)
cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
config = [1, 0,0]  # Time-varying location, constant scale and shape
dist = genextreme

'''
prior_specs = [('normal',
  {'loc': np.float64(8.7),
   'scale': np.float64(4.35)}),
 ('normal', {'loc': 0, 'scale': 0.025}),
 ('normal', {'loc': np.float64(5.6), 'scale': 1}),
 ('normal', {'loc': 0, 'scale': 0.2})]
'''

initial_params = [0.0,0,1.0, 0.1]

def test_initialization():
    model = NonStationaryEVD(config, data, cov, dist)
    assert isinstance(model, NonStationaryEVD)
    assert model.data.shape[0] == 50
    assert len(model.config) == 3

def test_neg_log_likelihood():
    model = NonStationaryEVD(config, data, cov, dist)
    nll = model.neg_log_likelihood(initial_params)
    assert np.isfinite(nll)

def test_posterior_log_prob():
    model = NonStationaryEVD(config, data, cov, dist)
    model.prior_specs = model.suggest_priors()
    logp = model.posterior_log_prob(initial_params)
    assert np.isfinite(logp)

def test_MH_RandWalk():
    model = NonStationaryEVD(config, data, cov, dist)
    samples,_ = model.MH_RandWalk(
        num_samples=100,
        initial_params=initial_params,
        proposal_widths=[0.01,0.01, 0.01, 0.05],
        T=50
    )
    assert samples.shape == (100, sum(config)+3)
    
def test_MH_Mala():
    model = NonStationaryEVD(config, data, cov, dist)
    samples,_ = model.MH_Mala(
        num_samples=100,
        initial_params=initial_params,
        step_sizes = [0.01,0.01, 0.01, 0.05],
        T=50
    )
    assert samples.shape == (100, sum(config)+3)
    
def test_MH_Mala():
    model = NonStationaryEVD(config, data, cov, dist)
    samples,_ = model.MH_Hmc(
        num_samples=100,
        initial_params=initial_params,
        step_size= 0.05,
        T=50
    )
    assert samples.shape == (100, sum(config)+3)

def test_frequentist_nsEVD():
    model = NonStationaryEVD(config, data, cov, dist)
    model.bounds  = model.suggest_bounds()
    params = model.frequentist_nsEVD(initial_params)
    assert len(params) == sum(config)+3
    assert np.isfinite(params).all()

def test_static_ns_EVDrvs():
    samples = NonStationaryEVD.ns_EVDrvs(dist, initial_params, cov, config, size=50)
    assert samples.shape == (50,)
    assert np.isfinite(samples).all()
