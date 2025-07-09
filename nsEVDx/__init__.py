from .evd_model import NonStationaryEVD
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

__all__ = [
    "NonStationaryEVD",
    "neg_log_likelihood",
    "neg_log_likelihood_ns",
    "EVD_parsViaMLE",
    "comb",
    "l_moments",
    "GPD_parsViaLM",
    "GEV_parsViaLM",
    "plot_trace",
    "plot_posterior",
    "bayesian_metrics",
    "gelman_rubin"
]


from ._version import get_versions
__version__ = get_versions()["version"]
del get_versions
