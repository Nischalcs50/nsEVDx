Theoretical Background
======================

Generalized Extreme Value Distribution
--------------------------------------

The GEV distribution unifies the three types of extreme value distributions. 
The cumulative distribution function (CDF) of a GEV random variable :math:`X` 
with location :math:`\mu`, scale :math:`\sigma > 0`, and shape :math:`\xi` is:

**CDF:**

.. math::

   F(x; \mu, \sigma, \xi) = \exp\left\{-\left[1 - \xi\left(\frac{x - \mu}{\sigma}\right)\right]^{1/\xi}\right\} \quad \text{for } \xi \neq 0

.. math::

   F(x; \mu, \sigma, 0) = \exp\left\{-\exp\left[-\left(\frac{x - \mu}{\sigma}\right)\right]\right\} \quad \text{for } \xi = 0

The probability density function (PDF) of GEV is given by:

.. math::

   f(x; \mu, \sigma, \xi)
   =
   \frac{1}{\sigma}
   \left[
      1 - \xi\left(\frac{x-\mu}{\sigma}\right)
   \right]^{1/\xi - 1}
   \exp\left\{
      -
      \left[
         1 - \xi\left(\frac{x-\mu}{\sigma}\right)
      \right]^{1/\xi}
   \right\},
   \qquad \xi \neq 0

with support

.. math::

   1 - \xi\left(\frac{x-\mu}{\sigma}\right) > 0.

For :math:`\xi = 0` (Gumbel case),

.. math::

   f(x; \mu, \sigma, 0)
   =
   \frac{1}{\sigma}
   \exp\left[
      -
      \left(\frac{x-\mu}{\sigma}\right)
   \right]
   \exp\left\{
      -
      \exp\left[
         -
         \left(\frac{x-\mu}{\sigma}\right)
      \right]
   \right\}.

Generalized Pareto Distribution
-------------------------------

The CDF of exceedances :math:`Y = X - \mu > 0` over threshold :math:`\mu` following a GPD, with scale :math:`\sigma > 0` and shape :math:`\xi`:

**CDF:**

.. math::

   F(y; \mu, \sigma, \xi) = 1 - \left(1 + \xi\frac{y - \mu}{\sigma}\right)^{-1/\xi} \quad \text{for } \xi \neq 0

.. math::

   F(y; \mu, \sigma, 0) = 1 - \exp\left(-\frac{y - \mu}{\sigma}\right) \quad \text{for } \xi = 0

The PDF of GPD is given by:

.. math::

   f(y; \mu, \sigma, \xi)
   =
   \frac{1}{\sigma}
   \left(
      1 + \xi\frac{y-\mu}{\sigma}
   \right)^{-1/\xi - 1},
   \qquad \xi \neq 0

with support

.. math::

   1 + \xi\frac{y-\mu}{\sigma} > 0.

For :math:`\xi = 0` (Exponential case),

.. math::

   f(y; \mu, \sigma, 0)
   =
   \frac{1}{\sigma}
   \exp\left(
      -
      \frac{y-\mu}{\sigma}
   \right).

Non-stationary Framework
------------------------

In a non-stationary framework, parameters are modeled as functions of covariates:

**Location (linear):**

.. math::

   \mu(t) = \beta_0 + \beta_1 Z_1(t) + \beta_2 Z_2(t) + \dots

**Scale (exponential):**

.. math::

   \sigma(t) = \exp(\alpha_0 + \alpha_1 Z_1(t) + \alpha_2 Z_2(t) + \dots)

**Shape (linear):**

.. math::

   \xi(t) = \kappa_0 + \kappa_1 Z_1(t) + \kappa_2 Z_2(t) + \dots

Where :math:`Z(t)` is a dynamic covariate that changes with time and affects the extreme value distributions.


Non-Stationarity Configuration via Config Vector
-------------------------------------------------

In ``nsEVDx``, non-stationarity is controlled via a configuration vector:

.. math::

   \text{config} = [a, b, c]

Each element in the configuration specifies the number of covariates for the **location** (:math:`\mu`), **scale** (:math:`\sigma`), and **shape** (:math:`\xi`) parameters:

* A value of **0** indicates stationarity.
* Values **> 0** indicate non-stationary modeling using the corresponding number of covariates.

This framework allows flexible, parsimonious modeling of non-stationary extreme value distributions, including covariates only where supported by data.


Log-Likelihood
--------------

The primary computational core in ``nsEVDx`` is the evaluation of the generalized
log-likelihood  suitable for both stationary and non-stationary three-parameter
extreme value distributions (EVDs).
The log-likelihood form the foundation of parameter estimation, frequentist inference,
and Bayesian inference for both stationary and non-stationary EVDs.

For a sample of observations
:math:`\{x_1, x_2, \ldots, x_n\}`, the log-likelihood is defined as

.. math::

   \ell(\theta)
   =
   \sum_{i=1}^{n}
   \log f(x_i;\theta),

where :math:`f(\cdot)` denotes the PDF of the selected EVD and
:math:`\theta` represents the full parameter vector.

For non-stationary models, :math:`\theta` contains all regression
coefficients associated with the location, scale, and shape parameters,

.. math::

   \mu(t), \qquad \sigma(t), \qquad \xi(t),

which may vary as functions of user-specified covariates.



Analytical Gradient
-------------------

To improve computational efficiency and numerical stability,
``nsEVDx`` provides analytical gradients of the log-likelihood with
respect to all model parameters whenever possible.

The gradient vector is defined as

.. math::

   \nabla_{\theta}\ell(\theta)
   =
   \left(
   \frac{\partial \ell}{\partial \theta_1},
   \frac{\partial \ell}{\partial \theta_2},
   \ldots,
   \frac{\partial \ell}{\partial \theta_p}
   \right)^T,

where :math:`p` is the number of model parameters.

Analytical gradients offer several advantages:

* Faster optimization and sampling.
* Improved numerical accuracy.
* Reduced sensitivity to finite-difference step sizes.
* Better scaling for high-dimensional non-stationary models.

Numerical Gradient Fallback
---------------------------

As a robustness feature, ``nsEVDx`` also provides numerical
finite-difference approximations of the gradient. This fallback option
can be used when analytical derivatives are unavailable or for validating
gradient implementations.

For a parameter :math:`\theta_j`, the central-difference approximation is

.. math::

   \frac{\partial \ell}{\partial \theta_j}
   \approx
   \frac{
   \ell(\theta_j + h)
   -
   \ell(\theta_j - h)
   }{2h},

where :math:`h` is a small perturbation.

Although numerical gradients are generally slower and less accurate than
their analytical counterparts, they provide a robust fallback mechanism
and facilitate validation of derivative implementations.

MCMC Applications
----------------------

Both the log-likelihood and its gradient are designed for direct use in
Markov Chain Monte Carlo (MCMC) algorithms and related Bayesian inference
methods.

Examples include:

* Random-Walk Metropolis-Hastings (RWH).
* Metropolis-Adjusted Langevin Algorithms (MALA).
* Hamiltonian Monte Carlo (HMC).


When analytical gradients are available they are used by default for
maximum computational efficiency. Numerical gradients remain available
as a fallback option, ensuring compatibility across all stationary and
non-stationary GEV and GPD model configurations.
``


