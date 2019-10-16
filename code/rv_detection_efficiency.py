
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
    ("bp_rp", 10),
    ("phot_rp_mean_mag", 10),
    ("absolute_rp_mag", 10),
    ("rv_nb_transits", rv_nb_transits_bins)
])

X = np.array([sources[ln][()] for ln in label_names_and_bins.keys()]).T
finite = np.all(np.isfinite(X), axis=1)

H, edges = np.histogramdd(X[finite], bins=tuple(label_names_and_bins.values()))

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


