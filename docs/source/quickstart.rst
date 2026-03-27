Quickstart Guide
================
.. code-block:: python

   import nsEVDx as ns
   import numpy as np
   from scipy.stats import genextreme
   import matplotlib.pyplot as plt

   ## GIVEN, NON_STATIONARY TIME_SERIES OF EXTREMES
   data = np.array([30.16, 36.29, 49.58, 22.45, 40.75, 42.99, 21.95, 42.8 , 46.04,
         40.07, 15.61, 56.11, 31.87, 25.28, 33.38, 17.82, 41.89, 53.22,
         45.11, 33.3 , 34.23, 44.41, 26.72, 38.47, 29.79, 33.27, 25.33,
         34.62, 44.28, 48.06, 43.9 , 31.94, 61.49, 37.04, 39.72, 46.52,
         44.4 , 45.66, 34.03, 47.3 , 29.83, 43.57, 39.65, 35.54, 42.74,
         43.57, 43.12, 34.17, 45.5 , 33.04])
   plt.plot(data)

   cov = np.array(range(50)) # Assuming a covariate that increases linearly 

   config = [1, 0, 0] # means location parameter is non-stationary and scale and shape parameters are stationary
   # See Usage.md or https://nischalcs50.github.io/nsEVDx/ for more details on config vector
   # checking the parameters corresponding to the config
   print(ns.NonStationaryEVD.get_param_description(config=config, n_cov=1)) # checking the parameters corresponding to the config

   ## SETTING PRIORS
   # Prior: normal for regression coefficients of location parameter, half-normal for scale, normal for shape
   prior_specs = [
    ("normal", {"loc": 30, "scale": 15}),
    ("normal", {"loc": 0, "scale": 0.25}),
    ("halfnormal", {"loc": 5, "scale": 5}),
    ("normal", {"loc": 0, "scale": 0.4}),
   ]
   sampler = ns.NonStationaryEVD(
    config, data, cov, dist=genextreme, prior_specs=prior_specs
   )
   print(sampler.descriptions)

   ## RUNNING BATESIAN ALGORITHM
   # fitting a non-stationary GEV model to the data using MALA algorithm
   initial_params = [30, 0, 7, 0.1]
   samples, a_rate, r_hat = sampler.MH_Mala(
      num_samples=3000, initial_params=initial_params,
      step_sizes=[1,0.075,0.75,5e-2], T=5,
      burn_in=500, num_chains=3, n_jobs=3
   )

   ## PRINT RESULTS
   print(f"acceptance_rate : {a_rate}")
   print(f"r_hat : {r_hat}")
   np.set_printoptions(suppress=True, precision=6)
   sample_all_chains = np.vstack(samples)  # shape (num_chains*num_samples, num_params)
   sample_mean = sample_all_chains.mean(axis=0)
   print(f"Sample mean : {sample_mean}")

   ## PLOT CONVERGENCE & POSTERIORS
   ns.plot_trace(samples, config, fig_size=(7, 10))
   ns.plot_posterior(samples, config, fig_size=(7, 8))
