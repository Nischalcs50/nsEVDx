import numpy as np
import pytest
from scipy.stats import genextreme

from nsEVDx import NonStationaryEVD


def model(config=(1, 0, 0)):
    cov = np.arange(5, dtype=float)[None, :] / 4
    return NonStationaryEVD(config, np.ones(5), cov, genextreme)


def test_predict_params_respects_config():
    result = model().predict_params([10, 2, 3, 0], covariates=[[0, 0.5, 1]])
    assert np.allclose(result["mu"], [10, 11, 12])
    assert np.allclose(result["sigma"], 3)
    assert np.allclose(result["xi"], 0)


def test_risk_and_reliability_are_complements():
    m = model((0, 0, 0))
    risk = m.nonstationary_risk(2, 5, params=[0, 1, 0])
    reliability = m.nonstationary_reliability(2, 5, params=[0, 1, 0])
    assert np.isclose(risk["risk"] + reliability["reliability"], 1)
    assert risk["exceedance_probability"].shape == (5,)


def test_first_exceedance_pmf_and_return_period():
    m = model((0, 0, 0))
    pmf = m.first_exceedance_pmf(2, 5, params=[0, 1, 0])
    period = m.nonstationary_return_period(2, 5, params=[0, 1, 0])
    assert np.isclose(pmf["pmf"].sum() + pmf["no_exceedance"], 1)
    assert np.isclose(period["return_period"], 1 + np.sum(
        np.cumprod(1 - pmf["exceedance_probability"])
    ))


def test_predict_params_defaults_and_validation():
    m = model()
    m.params = [10, 2, 3, 0]
    assert np.allclose(m.predict_params()["mu"], 10 + 2 * m.cov[0])
    with pytest.raises(ValueError):
        m.predict_params([1, 2, 3])
    m2 = model((2, 0, 0))
    with pytest.raises(ValueError):
        m2.predict_params([1, 2, 3, 4, 5], time=[0, 1])


def test_posterior_uncertainty_and_aliases():
    m = model((0, 0, 0))
    draws = [[0, 1, 0], [0.1, 1.1, -0.1]]
    assert "uncertainty" in m.nonstationary_risk(
        2, 5, posterior_samples=draws
    )
    assert "uncertainty" in m.nonstationary_reliability(
        2, 5, posterior_samples=draws
    )
    assert "uncertainty" in m.first_exceedance_pmf(
        2, 5, posterior_samples=draws
    )
    assert "uncertainty" in m.nonstationary_return_period(
        2, 5, posterior_samples=draws
    )
    with pytest.raises(ValueError):
        m.nonstationary_risk(2, 0, params=[0, 1, 0])
    with pytest.raises(ValueError):
        m.nonstationary_risk(2, 5, posterior_samples=[[[0, 1, 0]]])


def test_gpd_and_plot_helpers():
    from scipy.stats import genpareto

    m = NonStationaryEVD([1, 0, 0], np.ones(5),
                         np.arange(5, dtype=float)[None, :], "gpd")
    params = [0, 0.1, 1, 0.1]
    assert m.nonstationary_risk(1, 5, params=params)["risk"] >= 0
    levels = m.plot_return_levels(
        params=params, time=np.arange(5), posterior_samples=[params]
    )
    periods = m.plot_return_periods(
        1, params=params, time=np.arange(5), posterior_samples=[params]
    )
    assert levels["return_levels"].shape == (6, 5)
    assert periods["return_period"].shape == (5,)
    assert genpareto.name == "genpareto"
