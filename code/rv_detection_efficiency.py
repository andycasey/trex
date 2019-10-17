
import os
import itertools # dat feel
import operator
import functools
import numpy as np
import h5py as h5
import pickle
import yaml
import warnings
import george
from astropy.constants import G
from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from scipy import special, integrate
from tqdm import tqdm
from collections import OrderedDict


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import twobody
import utils

# Load in sources.
data = h5.File("../data/sources.hdf5", "r")
sources = data["sources"]

# Load in results.
results = h5.File("../results/rc.7/results-5482.h5", "r")


# Simulate a binary population.
np.random.seed(42)

# Assume that we observe each system at a uniformly random time.
# From https://www.cosmos.esa.int/web/gaia/dr2
observing_start, observing_end = (Time('2014-07-25T10:30'), Time('2016-05-23T11:35')) 
observing_span = (observing_end - observing_start).to(u.day) # ~ 668 days

#T = observing_start + np.random.uniform(0, 1, N_rv_obs) * (observing_end - observing_start)

# Now let us make draws from what we really think are representative.
def draw_periods(N, log_P_mu=4.8, log_P_sigma=2.3, log_P_min=-2, log_P_max=12):

    # orbital period Duquennoy and Mayor 1991 distribution
    P = np.empty(N)
    P_min, P_max = (10**log_P_min, 10**log_P_max)

    for i in range(N):
        while True:
            x = np.random.lognormal(log_P_mu, log_P_sigma)
            if P_max >= x >= P_min:
                P[i] = x
                break
    return P


def mass_ratios(M_1, q_min=0.1, q_max=1):
    N = len(M_1)
    q = np.empty(N)
    for i, M in enumerate(M_1):
        q[i] = np.random.uniform(q_min / M.value, q_max)
    return q

def salpeter_imf(N, alpha=2.35, M_min=0.1, M_max=100):
    # salpeter imf
    log_M_limits = np.log([M_min, M_max])

    max_ll = M_min**(1.0 - alpha)

    M = []
    while len(M) < N:
        Mi = np.exp(np.random.uniform(*log_M_limits))

        ln = Mi**(1 - alpha)
        if np.random.uniform(0, max_ll) < ln:
            M.append(Mi)

    return np.array(M)


N_rv_simulations = 1000

log_P_min, log_P_max = (-2, 12) # log_10(P / day)
q_min, q_max = (0.1, 1)
M_min, M_max = (0.1, 100) # sol masses

# Draw periods, masses, and q values.        
N = 1000

P = draw_periods(N, log_P_min=log_P_min, log_P_max=log_P_max) * u.day
M_1 = salpeter_imf(N, M_min=M_min, M_max=M_max) * u.solMass
q = mass_ratios(M_1, q_min=q_min, q_max=q_max)
M_2 = q * M_1
e = np.zeros(N)
i = np.arccos(np.random.uniform(0, 1, size=N))

# Compute the semi-major axis given the orbital period and total mass.
M_total = M_1 + M_2
a = np.cbrt(G * (M_1 + M_2) * (P/(2 * np.pi))**2)

# Compute the radial velocity semi-amplitude.
K = 2 * np.pi * a * np.sin(i) / (P * np.sqrt(1 - e**2))

# Calculate the detection efficiency on a grid of:
# colour, apparent magnitude, absolute magnitude, and number of RV observations.
rv_nb_transits_bins = np.logspace(0, np.log2(50), 10, base=2).astype(int)
rv_nb_transits_bins[0] = 0

label_names_and_bins = OrderedDict([
    ("bp_rp", 15),
    ("absolute_rp_mag", 15),
    ("phot_rp_mean_mag", 2),
    ("rv_nb_transits", rv_nb_transits_bins)
])

X = np.array([sources[ln][()] for ln in label_names_and_bins.keys()]).T
finite = np.all(np.isfinite(X), axis=1)

H, edges = np.histogramdd(X[finite], bins=tuple(label_names_and_bins.values()))
centroids = [e[:-1] + 0.5 * np.diff(e) for e in edges]



# For fun let's plot the rv_nb_transits histogram.
fig, ax = plt.subplots()
ax.hist(X.T[-1], bins=np.arange(0, 200), log=True)
ax.set_xlabel(r"{number of radial velocity transits}")
ax.set_ylabel(r"{number of sources}")
fig.tight_layout()


fig, ax = plt.subplots()
x = P.to(u.day).value
y = K.to(u.km/u.s).value
ax.scatter(x, y)
ax.loglog()


# Now let's calculate the mu_single and sigma_single at each bin point.
def get_gp(results, model_name, parameter_name):

    X = results[f"models/{model_name}/gp_model/{parameter_name}/X"][()]
    Y = results[f"models/{model_name}/gp_model/{parameter_name}/Y"][()]

    attrs = results[f"models/{model_name}/gp_model/{parameter_name}"].attrs

    metric = np.var(X, axis=0)
    kernel = george.kernels.ExpSquaredKernel(metric=metric, ndim=metric.size)
    gp = george.GP(kernel,
                   mean=np.mean(Y), fit_mean=True,
                   white_noise=np.log(np.std(Y)), fit_white_noise=True)
    for p in gp.parameter_names:
        gp.set_parameter(p, attrs[p])

    gp.compute(X)

    return (gp, Y)


gp_mu_single, y = get_gp(results, "rv", "mu_single")
gp_sigma_single, _ = get_gp(results, "rv", "sigma_single")

grid = np.meshgrid(*centroids[:-1])
t = np.array(grid).reshape((3, -1)).T

gp_kwds = dict(y=y, t=t, return_var=True, return_cov=False)

p_mu_single, var_mu_single = gp_mu_single.predict(**gp_kwds)
p_sigma_single, var_sigma_single = gp_sigma_single.predict(**gp_kwds)

observing_span_in_days = observing_span.to(u.day).value

def simulate_rv_jitter(K, P, rv_nb_transits, p_mu_single, var_mu_single, p_sigma_single, var_sigma_single):

    T = np.random.uniform(0, observing_span_in_days, rv_nb_transits)
    phi = np.random.uniform(0, 2 * np.pi)

    v = K * np.sin(2 * np.pi * T / P + phi)
    mu = np.random.normal(p_mu_single, np.sqrt(var_mu_single), rv_nb_transits)
    sigma = np.random.normal(p_sigma_single, np.sqrt(var_sigma_single), rv_nb_transits)

    # TODO THIS IS WRONG
    mu = np.abs(mu)
    sigma = np.abs(sigma)

    noise = np.random.normal(mu, sigma)

    return np.std(v + noise)

def jitter_to_radial_velocity_error(jitter, rv_nb_transits):
    return np.sqrt((2 * rv_nb_transits)/np.pi * jitter**2 - 0.11**2)



G = t.shape[0]
# v_stds has shape (number_of_bins_in mag and color, number of rv_bins, number of fake binaries)
v_stds = np.empty((G, rv_nb_transits_bins.size, N))

KPs = np.array([
    K.to(u.km/u.s).value,
    P.to(u.day).value
]).T

for i in tqdm(range(G)):
    kwds = dict(p_mu_single=p_mu_single[i], var_mu_single=var_mu_single[i],
                p_sigma_single=p_sigma_single[i], var_sigma_single=var_sigma_single[i])

    for j in range(v_stds.shape[1]):
        if rv_nb_transits_bins[j] < 2:
            v_stds[i, j, :] = np.nan
            continue

        kwds.update(rv_nb_transits=rv_nb_transits_bins[j])

        for k, (K_, P_) in enumerate(KPs):
            v_stds[i, j, k] = simulate_rv_jitter(K=K_, P=P_, **kwds)
    
# OK, now when would we actually have detected it.
#rv_error = jitter_to_radial_velocity_error(v_stds, rv_nb_transits_bins.reshape((1, -1, 1)))

def p_single(rv_jitter, rv_nb_transits):

    rv_error = jitter_to_radial_velocity_error(rv_jitter, rv_nb_transits.reshape((1, -1, 1)))

    p_single = np.empty_like(rv_jitter)
    p_single[rv_error >= 20] = np.nan
    p_single[~np.isfinite(rv_error)] = np.nan

    # Let's do something dumb and just say if it's > mu + 3 * sigma 
    # and ignore the variance in estimating both of those quantities
    detectable = (p_mu_single + 3 * p_sigma_single).reshape((-1, 1, 1))

    return rv_jitter >= detectable


p = np.sum(p_single(v_stds, rv_nb_transits_bins), axis=(1, 2))
p = p / (v_stds.shape[1] * v_stds.shape[2])

fig, ax = plt.subplots()
ax.imshow(np.mean(p.reshape((15, 15, -1)), axis=2))




 