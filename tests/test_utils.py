# test_utils.py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import genextreme

from nsEVDx.utils import (
    EVD_parsViaMLE,
    GEV_parsViaLM,
    GPD_parsViaLM,
    bayesian_metrics,
    comb,
    l_moments,
    neg_log_likelihood,
    neg_log_likelihood_ns,
    plot_posterior,
    plot_trace,
)


def test_comb():
    assert comb(5, 2) == 10


def test_l_moments():
    data = np.random.rand(100)
    l_moms = l_moments(data)
    assert len(l_moms) == 6


def test_EVD_parsViaMLE():
    data = np.random.rand(100)
    params = EVD_parsViaMLE(data, genextreme)
    assert len(params) == 3


def test_neg_log_likelihood():
    data = np.random.rand(100)
    params = genextreme.fit(data)
    nll = neg_log_likelihood(params, data, genextreme)
    assert isinstance(nll, float)


def test_neg_log_likelihood_ns():
    dist = genextreme
    data = dist.rvs(-0.1, loc=22, scale=8, size=25, random_state=0)
    cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
    config = [1, 0, 0]  # Time-varying location, constant scale and shape
    params = [20, 0.02, 6.5, -0.2]
    nll = neg_log_likelihood_ns(params, data, cov, config, dist)
    assert np.isfinite(nll)


def test_GPD_parsViaLM():
    data = np.random.rand(100)
    pars = GPD_parsViaLM(data)
    assert len(pars) == 3


def test_GEV_parsViaLM():
    data = np.random.rand(100)
    pars = GEV_parsViaLM(data)
    assert len(pars) == 3


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
