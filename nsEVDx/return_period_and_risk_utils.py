"""Tools for explaining and calculating nonstationary extreme-event risk.

The functions in this module use a fitted GEV or GPD model to describe how
the distribution changes over time or with covariates. They can return
ordinary point estimates or propagate Bayesian posterior samples into
credible intervals.
"""

import numpy as np


def _dist(model):
    """Return the SciPy distribution object used by ``model``."""
    if not isinstance(model.dist, str):
        return model.dist
    from scipy.stats import genextreme, genpareto
    return genpareto if model.dist.lower() in ("gpd", "genpareto") else genextreme


def _parameters(model, params):
    """Find parameter values supplied directly or stored on the model."""
    if params is not None:
        return np.asarray(params, dtype=float)
    for name in ("params", "fitted_params", "estimated_params"):
        value = getattr(model, name, None)
        if value is not None:
            return np.asarray(value, dtype=float)
    raise ValueError("Provide fitted parameter values with `params`.")


def _summary(values, quantiles):
    """Summarize posterior values with a median and requested interval."""
    values = np.asarray(values, dtype=float)
    q = np.quantile(values, quantiles, axis=0)
    return {"median": np.median(values, axis=0),
            "lower": q[0], "upper": q[1], "quantiles": tuple(quantiles)}


def _posterior_results(model, method, posterior_samples, kwargs, quantiles):
    """Evaluate one analysis function for every posterior parameter draw."""
    samples = np.asarray(posterior_samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2:
        raise ValueError("posterior_samples must have shape (draws, parameters).")
    results = [method(model, params=draw, **kwargs) for draw in samples]
    return results


def _point_params(params, posterior_samples):
    """Choose explicit parameters or the posterior median for the main result."""
    if params is not None:
        return params
    return np.median(np.asarray(posterior_samples, dtype=float), axis=0)


def predict_params(model, params=None, covariates=None, time=None):
    """Predict the GEV/GPD parameters for each requested time or covariate.

    The returned dictionary contains ``mu`` (location), ``sigma`` (scale),
    and ``xi`` (shape). If no new covariates are supplied, the model's own
    covariates are used.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    params : array-like, optional
        Parameter vector. If omitted, fitted parameters stored on ``model``
        are used.
        Its length must be ``sum(model.config) + 3``.
    model.config : tuple[int, int, int]
        Counts of covariates used by location, scale, and shape. A zero count
        keeps that parameter stationary; positive counts use the corresponding
        leading covariate rows.
    covariates : array-like, optional
        Covariates with shape ``(n_covariates, n_times)``.
    time : array-like, optional
        Time values used as a one-row covariate array. This is valid when
        every nonstationary parameter uses at most one covariate.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``mu``: location parameter for each time step.
        ``sigma``: positive scale parameter for each time step.
        ``xi``: shape parameter for each time step.
    """
    params = _parameters(model, params)
    config = np.asarray(getattr(model, "config", ()), dtype=int)
    if config.shape != (3,) or np.any(config < 0):
        raise ValueError("model.config must contain three nonnegative counts.")
    expected = int(np.sum(config) + 3)
    if params.size != expected:
        raise ValueError(
            f"Expected {expected} parameters for config {config.tolist()}, "
            f"got {params.size}."
        )
    if covariates is None:
        if time is None:
            cov = np.atleast_2d(model.cov)
        else:
            cov = np.atleast_2d(np.asarray(time, dtype=float))
            if np.max(config, initial=0) > 1:
                raise ValueError(
                    "time alone supplies one covariate; provide covariates "
                    "for config entries requiring multiple covariates."
                )
    else:
        cov = np.asarray(covariates, dtype=float)
        if cov.ndim == 1:
            cov = cov[None, :]
    required_covariates = int(np.max(config, initial=0))
    if cov.ndim != 2 or cov.shape[0] < required_covariates:
        raise ValueError(
            f"config requires at least {required_covariates} covariate rows; "
            f"got shape {cov.shape}."
        )
    n = cov.shape[1]
    idx = 0
    values = []
    for i, count in enumerate(config):
        count = int(count)
        if count:
            beta = params[idx:idx + count + 1]
            value = beta[0] + beta[1:] @ cov[:count]
            if i == 1:
                value = np.exp(value)
            idx += count + 1
        else:
            value = np.full(n, params[idx])
            idx += 1
        values.append(np.asarray(value, dtype=float))
    return {"mu": values[0], "sigma": values[1], "xi": values[2]}


def _exceedance_prob(model, design_level, horizon, params=None, covariates=None,
                     time=None):
    """Calculate the probability of exceeding a design level at each time."""
    if horizon is not None:
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("horizon must be positive.")
    if time is None and covariates is None and horizon is not None:
        time = np.arange(horizon, dtype=float)
    p = predict_params(model, params, covariates, time)
    if horizon is not None and len(p["mu"]) != horizon:
        raise ValueError("Covariates/time length must equal horizon.")
    cdf = _dist(model).cdf(design_level, c=p["xi"], loc=p["mu"],
                            scale=p["sigma"])
    return np.clip(1.0 - np.asarray(cdf, dtype=float), 0.0, 1.0), p


def nonstationary_reliability(model, design_level, horizon, params=None,
                              covariates=None, time=None,
                              posterior_samples=None, quantiles=(0.05, 0.95)):
    """Reliability represents the probability that a structure, system,
    design level (not necessarily a POT threshold) survives an
    entire design life without exceedance.

    Reliability is the product of the year-by-year non-exceedance
    probabilities. The result also includes the complementary risk and,
    when posterior samples are supplied, credible intervals for both values.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    design_level : float
        Fixed design level being evaluated.
    horizon : int
        Number of future time steps in the design life.
    params : array-like, optional
        Parameter vector used for the point estimate.
    covariates, time : array-like, optional
        Future covariates or time values.
    posterior_samples : array-like, optional
        Posterior draws with shape ``(n_draws, n_parameters)``.
    quantiles : tuple[float, float], optional
        Lower and upper posterior interval probabilities.

    Returns
    -------
    dict[str, object]
        ``reliability``: float probability of no exceedance over the
        horizon.
        ``risk``: float probability of at least one exceedance.
        ``exceedance_probability``: NumPy array containing ``p_t`` for each
        time step.
        ``uncertainty``: optional dict with ``reliability`` and ``risk``
        posterior summaries; each summary contains ``median``, ``lower``,
        ``upper``, and ``quantiles``.
    """
    if posterior_samples is not None:
        draws = _posterior_results(
            model, nonstationary_reliability, posterior_samples,
            {"design_level": design_level, "horizon": horizon,
             "covariates": covariates, "time": time}, quantiles)
        point = nonstationary_reliability(
            model, design_level, horizon, _point_params(params, posterior_samples),
            covariates, time)
        point["uncertainty"] = {
            "reliability": _summary([d["reliability"] for d in draws], quantiles),
            "risk": _summary([d["risk"] for d in draws], quantiles),
        }
        return point
    p, _ = _exceedance_prob(model, design_level, horizon, params, covariates, time)
    reliability = float(np.prod(1.0 - p))
    return {"reliability": reliability, "risk": 1.0 - reliability,
            "exceedance_probability": p}


def nonstationary_risk(model, design_level, horizon, params=None, covariates=None,
                       time=None, posterior_samples=None,
                       quantiles=(0.05, 0.95)):
    """Estimate the chance of at least one exceedance during a design life.

    This is the complement of nonstationary reliability. Posterior samples
    can be supplied to quantify uncertainty in the risk estimate.

    The design-life risk follows the nonstationary exceedance framework of
    Salas and Obeysekera (2014), where risk over ``n`` time steps is
    ``1 - prod_t(1 - p_t)``.

    For a GPD model, these results describe the conditional excess above the
    design_level only. Because the Poisson exceedance rate is not modeled, they
    do not represent the full frequency of design_level exceedances.

    References
    ----------
    Salas, J. D., & Obeysekera, J. (2014). Revisiting the concepts of return
    period and risk for nonstationary hydrologic extreme events. Journal of
    Hydrologic Engineering, 19(3), 554-568.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    design_level : float
        Fixed design level being evaluated.
    horizon : int
        Number of future time steps in the design life.
    params : array-like, optional
        Parameter vector used for the point estimate.
    covariates, time : array-like, optional
        Future covariates or time values.
    posterior_samples : array-like, optional
        Posterior draws with shape ``(n_draws, n_parameters)``.
    quantiles : tuple[float, float], optional
        Lower and upper posterior interval probabilities.

    Returns
    -------
    dict[str, object]
        ``risk``: float probability of at least one exceedance over the
        horizon.
        ``reliability``: float probability of no exceedance.
        ``exceedance_probability``: NumPy array containing ``p_t`` for each
        time step.
        ``uncertainty``: optional dict with posterior summaries for
        ``risk`` and ``reliability``.
    """
    if posterior_samples is not None:
        result = nonstationary_reliability(
            model, design_level, horizon, params, covariates, time,
            posterior_samples, quantiles)
        return {"risk": result["risk"], "reliability": result["reliability"],
                "exceedance_probability": result["exceedance_probability"],
                "uncertainty": result["uncertainty"]}
    result = nonstationary_reliability(model, design_level, horizon, params,
                                       covariates, time)
    return {"risk": result["risk"], "reliability": result["reliability"],
            "exceedance_probability": result["exceedance_probability"]}


def first_exceedance_pmf(model, design_level, horizon, params=None,
                         covariates=None, time=None, posterior_samples=None,
                         quantiles=(0.05, 0.95)):
    """Calculate when the first design-level exceedance is expected to occur.

    The returned PMF gives the probability that the first exceedance occurs
    in each year, along with the probability that no exceedance occurs within
    the requested horizon.

    For a GPD model, this uses the conditional excess distribution only;
    it does not include a separate Poisson rate for how often exceedances
    occur.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    design_level : float
        Fixed design level being evaluated.
    horizon : int
        Number of time steps to evaluate.
    params : array-like, optional
        Parameter vector used for the point estimate.
    covariates, time : array-like, optional
        Future covariates or time values.
    posterior_samples : array-like, optional
        Posterior draws with shape ``(n_draws, n_parameters)``.
    quantiles : tuple[float, float], optional
        Lower and upper posterior interval probabilities.

    Returns
    -------
    dict[str, object]
        ``time``: integer NumPy array numbered from 1 through ``horizon``.
        ``pmf``: NumPy array where element ``x - 1`` is ``P(X=x)``.
        ``no_exceedance``: float probability of no exceedance in the horizon.
        ``exceedance_probability``: NumPy array containing ``p_t``.
        ``uncertainty``: optional dict with posterior summaries for ``pmf``
        and ``no_exceedance``.
    """
    if posterior_samples is not None:
        draws = _posterior_results(
            model, first_exceedance_pmf, posterior_samples,
            {"design_level": design_level, "horizon": horizon,
             "covariates": covariates, "time": time}, quantiles)
        point = first_exceedance_pmf(
            model, design_level, horizon, _point_params(params, posterior_samples),
            covariates, time)
        point["uncertainty"] = {
            "pmf": _summary([d["pmf"] for d in draws], quantiles),
            "no_exceedance": _summary(
                [d["no_exceedance"] for d in draws], quantiles),
        }
        return point
    p, _ = _exceedance_prob(model, design_level, horizon, params, covariates, time)
    survival_before = np.concatenate(([1.0], np.cumprod(1.0 - p[:-1])))
    pmf = p * survival_before
    return {"time": np.arange(1, len(p) + 1), "pmf": pmf,
            "no_exceedance": float(np.prod(1.0 - p)),
            "exceedance_probability": p}


def nonstationary_return_period(model, design_level, horizon=None, params=None,
                                covariates=None, time=None,
                                posterior_samples=None, quantiles=(0.05, 0.95)):
    """Estimate the expected waiting time to the first exceedance.

    This is the Salas and Obeysekera nonstationary return period, ``E(X)``.
    It is different from calculating a separate return level for each year.
    Posterior samples produce a median and credible interval for the waiting
    time.

    For a GPD model, this is based on the conditional excess distribution
    only. Since no Poisson exceedance rate is modeled, it is not a full POT
    return period.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    design_level : float
        Fixed design level being evaluated.
    horizon : int, optional
        Number of time steps used to approximate the expected waiting time.
        Defaults to 10,000.
    params : array-like, optional
        Parameter vector used for the point estimate.
    covariates, time : array-like, optional
        Future covariates or time values.
    posterior_samples : array-like, optional
        Posterior draws with shape ``(n_draws, n_parameters)``.
    quantiles : tuple[float, float], optional
        Lower and upper posterior interval probabilities.

    Returns
    -------
    dict[str, object]
        ``return_period``: float expected waiting time ``E(X)`` in time
        steps.
        ``pmf``: NumPy array of first-exceedance probabilities.
        ``time``: integer NumPy array numbered from 1 through the horizon.
        ``no_exceedance``: float probability of no exceedance in the horizon.
        ``exceedance_probability``: NumPy array containing ``p_t``.
        ``uncertainty``: optional dict with posterior summaries for
        ``return_period`` and ``no_exceedance``.
    """
    if posterior_samples is not None:
        draws = _posterior_results(
            model, nonstationary_return_period, posterior_samples,
            {"design_level": design_level, "horizon": horizon,
             "covariates": covariates, "time": time}, quantiles)
        point = nonstationary_return_period(
            model, design_level, horizon,
            _point_params(params, posterior_samples), covariates, time)
        point["uncertainty"] = {
            "return_period": _summary(
                [d["return_period"] for d in draws], quantiles),
            "no_exceedance": _summary(
                [d["no_exceedance"] for d in draws], quantiles),
        }
        return point
    if horizon is None:
        horizon = 10000
    result = first_exceedance_pmf(model, design_level, horizon, params,
                                  covariates, time)
    survival = np.cumprod(1.0 - result["exceedance_probability"])
    return {"return_period": float(1.0 + np.sum(survival)),
            "pmf": result["pmf"], "time": result["time"],
            "no_exceedance": result["no_exceedance"],
            "exceedance_probability": result["exceedance_probability"]}


def plot_return_levels(model, return_periods=(2, 5, 10, 25, 50, 100),
                       params=None, covariates=None, time=None, ax=None,
                       posterior_samples=None, quantiles=(0.05, 0.95)):
    """Plot return levels that may change over time or with covariates.

    Each line represents one requested return period. If posterior samples
    are provided, translucent bands show the selected credible interval for
    the corresponding return levels.

    For a GPD model, the plotted levels are conditional excess levels and do
    not include a Poisson exceedance rate.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    return_periods : array-like, optional
        Return periods to plot.
    params : array-like, optional
        Parameter vector used for the plotted point estimates.
    covariates, time : array-like, optional
        Future covariates or time values.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new axes is created when omitted.
    posterior_samples : array-like, optional
        Posterior draws with shape ``(n_draws, n_parameters)``.
    quantiles : tuple[float, float], optional
        Lower and upper posterior interval probabilities.

    Returns
    -------
    dict[str, object]
        ``time``: NumPy array of plotted time values.
        ``return_periods``: NumPy array of requested return periods.
        ``return_levels``: NumPy array with shape
        ``(n_return_periods, n_times)`` containing ``z_T(t)``.
        ``ax``: Matplotlib ``Axes`` object containing the plot.
        ``uncertainty``: optional dict containing
        ``return_levels`` posterior summaries with ``median``, ``lower``,
        ``upper``, and ``quantiles`` arrays.
    """
    import matplotlib.pyplot as plt

    p = predict_params(model, params, covariates, time)
    periods = np.asarray(return_periods, dtype=float)
    if np.any(periods <= 1):
        raise ValueError("return_periods must be greater than one.")
    levels = np.asarray([
        _dist(model).ppf(1.0 - 1.0 / period, c=p["xi"], loc=p["mu"],
                    scale=p["sigma"])
        for period in periods
    ])
    level_uncertainty = None
    if posterior_samples is not None:
        draws = np.asarray(posterior_samples, dtype=float)
        if draws.ndim == 1:
            draws = draws[None, :]
        if draws.ndim != 2:
            raise ValueError(
                "posterior_samples must have shape (draws, parameters)."
            )
        level_draws = []
        for draw in draws:
            pp = predict_params(model, draw, covariates, time)
            level_draws.append(np.asarray([
                _dist(model).ppf(1.0 - 1.0 / period, c=pp["xi"],
                                 loc=pp["mu"], scale=pp["sigma"])
                for period in periods
            ]))
        level_uncertainty = _summary(
            level_draws, quantiles)
    if ax is None:
        _, ax = plt.subplots()
    x = np.arange(len(p["mu"])) if time is None else np.asarray(time)
    ax.plot(x, levels.T)
    if level_uncertainty is not None:
        for lower, upper in zip(level_uncertainty["lower"],
                                level_uncertainty["upper"]):
            ax.fill_between(x, lower, upper, alpha=0.15)
    ax.set_xlabel("Time")
    ax.set_ylabel("Return level")
    ax.legend([f"T={t:g}" for t in periods])
    result = {"time": x, "return_periods": periods, "return_levels": levels,
              "ax": ax}
    if level_uncertainty is not None:
        result["uncertainty"] = {"return_levels": level_uncertainty}
    return result


def plot_return_periods(model, design_level, params=None, covariates=None,
                        time=None, ax=None, posterior_samples=None,
                        quantiles=(0.05, 0.95)):
    """Plot the time-varying return period for one fixed design level.

    The plotted period is ``1 / p_t``, where ``p_t`` is the modelled
    probability of exceeding ``design_level`` at time ``t``. For a GPD model,
    this is conditional on an exceedance above the design_level because no
    Poisson exceedance rate is modeled.

    Parameters
    ----------
    model : NonStationaryEVD
        Fitted nonstationary extreme-value model.
    design_level : float
        Fixed level whose return period is plotted.
    params : array-like, optional
        Parameter vector used for the point estimate.
    covariates, time : array-like, optional
        Future covariates or time values.
    ax : matplotlib.axes.Axes, optional
        Existing axes. A new axes is created when omitted.
    posterior_samples : array-like, optional
        Posterior draws with shape ``(n_draws, n_parameters)``.
    quantiles : tuple[float, float], optional
        Lower and upper posterior interval probabilities.

    Returns
    -------
    dict[str, object]
        ``time``: NumPy array of plotted time values.
        ``design_level``: float design level used for the plot.
        ``return_period``: NumPy array of ``1 / p_t`` values.
        ``ax``: Matplotlib ``Axes`` object containing the plot.
        ``uncertainty``: optional posterior summaries for
        ``return_period``.
    """
    import matplotlib.pyplot as plt

    p, _ = _exceedance_prob(model, design_level, None, params, covariates, time)
    periods = 1.0 / np.maximum(p, np.finfo(float).tiny)
    period_uncertainty = None
    if posterior_samples is not None:
        draws = np.asarray(posterior_samples, dtype=float)
        if draws.ndim == 1:
            draws = draws[None, :]
        if draws.ndim != 2:
            raise ValueError(
                "posterior_samples must have shape (draws, parameters)."
            )
        period_draws = []
        for draw in draws:
            draw_p, _ = _exceedance_prob(
                model, design_level, None, draw, covariates, time
            )
            period_draws.append(1.0 / np.maximum(
                draw_p, np.finfo(float).tiny
            ))
        period_uncertainty = _summary(period_draws, quantiles)

    if ax is None:
        _, ax = plt.subplots()
    x = np.arange(len(periods)) if time is None else np.asarray(time)
    ax.plot(x, periods, label=f"design_level={design_level:g}")
    if period_uncertainty is not None:
        ax.fill_between(
            x, period_uncertainty["lower"], period_uncertainty["upper"],
            alpha=0.2, label="credible interval"
        )
    ax.set_xlabel("Time")
    ax.set_ylabel("Return period")
    ax.legend()
    result = {"time": x, "design_level": float(design_level),
              "return_period": periods, "ax": ax}
    if period_uncertainty is not None:
        result["uncertainty"] = {"return_period": period_uncertainty}
    return result
