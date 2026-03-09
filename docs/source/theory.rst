Theoretical Background
======================

Generalized Extreme Value Distribution
--------------------------------------

The GEV distribution unifies the three types of extreme value distributions. The cumulative distribution function (CDF) of a GEV random variable $X$ with location $\mu$, scale $\sigma > 0$, and shape $\xi$ is:

**CDF:**

.. math::

   F(x; \mu, \sigma, \xi) = \exp\left\{-\left[1 + \xi\left(\frac{x - \mu}{\sigma}\right)\right]^{-1/\xi}\right\} \quad \text{for } \xi \neq 0

.. math::

   F(x; \mu, \sigma, 0) = \exp\left\{-\exp\left[-\left(\frac{x - \mu}{\sigma}\right)\right]\right\} \quad \text{for } \xi = 0


Generalized Pareto Distribution
-------------------------------

The CDF of exceedances $Y = X - \mu > 0$ over threshold $\mu$ following a GPD, with scale $\sigma > 0$ and shape $\xi$:

**CDF:**

.. math::

   F(y; \mu, \sigma, \xi) = 1 - \left(1 + \xi\frac{y - \mu}{\sigma}\right)^{-1/\xi} \quad \text{for } \xi \neq 0

.. math::

   F(y; \mu, \sigma, 0) = 1 - \exp\left(-\frac{y - \mu}{\sigma}\right) \quad \text{for } \xi = 0


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

Where $Z(t)$ is a dynamic covariate that changes with time and affects the extreme value distributions.


Non-Stationarity Configuration
------------------------------

In ``nsEVDx``, non-stationarity is controlled via a configuration vector:

.. math::

   \text{config} = [a, b, c]

Each element in the configuration specifies the number of covariates for the **location** ($\mu$), **scale** ($\sigma$), and **shape** ($\xi$) parameters:

* A value of **0** indicates stationarity.
* Values **> 0** indicate non-stationary modeling using the corresponding number of covariates.

This framework allows flexible, parsimonious modeling of non-stationary extreme value distributions, including covariates only where supported by data.