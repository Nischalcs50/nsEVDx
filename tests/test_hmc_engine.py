import numpy as np
import pytest
from scipy.stats import chi2, genextreme

from nsEVDx import NonStationaryEVD
from nsEVDx.hmc_engine import HMCEngine


def setup_hmc():
    """Shared setup for HMC testing."""
    data = np.random.gumbel(loc=20, scale=5, size=50)
    cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
    config = [1, 0, 0] # B0, B1, scale, shape
    model = NonStationaryEVD(config, data, cov, 'Genextreme')
    model.prior_specs = model.suggest_priors()
    engine = HMCEngine(model)
    params = np.array([20.0, 0.5, 5.0, 0.1])
    return engine, params

def test_setup_hmc():
    """test for setup conditions"""
    data = np.random.gumbel(loc=20, scale=5, size=50)
    cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
    config = [1, 0, 0] # B0, B1, scale, shape
    model = NonStationaryEVD(config, data, cov, genextreme)
    model.prior_specs = model.suggest_priors()
    engine = HMCEngine(model)
    params = np.array([20.0, 0.5, 5.0, 0.1])
    return engine, params

def test_setup_hmc2():
    """test for value rror"""
    data = np.random.gumbel(loc=20, scale=5, size=50)
    cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
    config = [1, 0, 0] # B0, B1, scale, shape
    with pytest.raises(ValueError, match="Analytical gradients not"):
        HMCEngine(NonStationaryEVD(config, data, cov, chi2))

def test_setup_hmc3():
    """"Verify flags are correct when a
    valid distribution is used"""
    data = np.random.gumbel(loc=20, scale=5, size=50)
    cov = np.vstack([np.ones_like(data), np.linspace(0, 1, len(data))])
    config = [1, 0, 0] # B0, B1, scale, shape
    engine = HMCEngine(NonStationaryEVD(config, data, cov, 'gev'))
    assert engine._is_gev is True
    assert engine._is_gpd is False

def test_hamiltonian_consistency():
    """Verify Hamiltonian calculation and finiteness."""
    engine, params = setup_hmc()
    momentum = np.random.normal(0, 1, len(params))
    M_diag = np.ones(len(params))

    H = engine._hamiltonian(params, momentum, M_diag)
    assert np.isfinite(H)
    # K(p) should be 0.5 * sum(p^2) for M=I
    expected_k = 0.5 * np.sum(momentum**2)
    assert np.isclose(engine._kinetic(momentum, M_diag), expected_k)

def test_leapfrog_conservation():
    """Leapfrog should roughly conserve energy for small step sizes."""
    engine, params = setup_hmc()
    M_diag = np.ones(len(params))
    momentum = np.random.normal(0, 0.1, len(params))

    H_start = engine._hamiltonian(params, momentum, M_diag)

    # Take 10 tiny steps
    q_new, p_new = engine._leapfrog(params,
                                    momentum,
                                    step_size=0.001,
                                    n_steps=10, M_diag=M_diag)
    H_end = engine._hamiltonian(q_new, p_new, M_diag)

    # Energy should be conserved within a reasonable tolerance
    assert np.abs(H_start - H_end) < 0.1

def test_mass_matrix_init():
    """Verify Hessian-based mass matrix is positive and clipped."""
    engine, params = setup_hmc()
    M_diag = engine._init_mass_matrix(params)

    assert M_diag.shape == params.shape
    assert np.all(M_diag >= 1e-4)
    assert np.all(M_diag <= 1e4)

def test_dual_averaging_logic():
    """Test that dual averaging updates step size."""
    engine, _ = setup_hmc()
    state = engine._dual_average_init(0.1, 0.8)

    # Simulate a high acceptance (1.0) -> step size should increase
    eps, _ = engine._dual_average_update(state, log_alpha=0.0)
    assert eps > 0.1

    # Simulate a low acceptance (log(0.1)) -> step size should decrease
    eps_low, _ = engine._dual_average_update(state, log_alpha=np.log(0.1))
    assert eps_low < eps
