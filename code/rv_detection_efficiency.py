
import os
import itertools # dat feel
import operator
import functools
import multiprocessing as mp
import numpy as np
import h5py as h5
import pickle
import yaml
import warnings

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
import binaries

# You can't tell me what to do python
warnings.filterwarnings("ignore", category=RuntimeWarning)

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
observing_span_in_days = (observing_end - observing_start).to(u.day).value # ~ 668 days


# Now let us make draws from what we really think are representative.
log_P_min, log_P_max = (-2, 12) # log_10(P / day)
log_P_min, log_P_max = (-2, 4)
print("Warning: using un-representative log_P boundaries")

q_min, q_max = (0.1, 1)
M_min, M_max = (0.1, 100) # sol masses

# Draw periods, masses, and q values.        
N = 10

P = binaries.draw_periods(N, log_P_min=log_P_min, log_P_max=log_P_max) * u.day
M_1 = binaries.salpeter_imf(N, M_min=M_min, M_max=M_max) * u.solMass
q = binaries.mass_ratios(M_1, q_min=q_min, q_max=q_max)
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
rv_nb_transits_centroids = np.logspace(0, np.log2(50), 10, base=2).astype(int)
rv_nb_transits_centroids[0] = 0

Q = 30
label_names_and_bins = OrderedDict([
    ("bp_rp", np.linspace(-0.5, 5, Q)),
    ("absolute_rp_mag", np.linspace(-10, 10, Q)),
    ("phot_rp_mean_mag", np.linspace(4, 14, Q)),
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

grid = np.array(np.meshgrid(*centroids)).reshape((3, -1)).T

gp_kwds = dict(y=y, t=grid, return_var=True, return_cov=False)

p_mu_single, var_mu_single = gp_mu_single.predict(**gp_kwds)
p_sigma_single, var_sigma_single = gp_sigma_single.predict(**gp_kwds)


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


# v_stds has shape (number_of_bins_in mag and color, number of rv_bins, number of fake binaries)
N_grid = grid.shape[0]
v_stds = np.empty((N_grid, rv_nb_transits_centroids.size, N))


KPs = np.array([
    K.to(u.km/u.s).value,
    P.to(u.day).value
]).T


gp_predictions = np.vstack([p_mu_single, var_mu_single, p_sigma_single, var_sigma_single]).T

v_stds = np.ones((N_grid, rv_nb_transits_centroids.size, N)) * np.nan


print("Opening the pool for business!")


def iters():
    for i, j, k in itertools.product(*(map(range, v_stds.shape))):
        args = (*KPs[k], rv_nb_transits_centroids[j], *gp_predictions[i])
        yield (i, j, k, args)


def w(args):
    i, j, k = args[:3]
    return (i, j, k, simulate_rv_jitter(*args[3]))


processes, chunk_size = 100, 100
with mp.Pool(processes) as pool:
    for i, j, k, v_std in tqdm(pool.imap_unordered(w, iters(), chunk_size), total=v_stds.size):
        v_stds[i, j, k] = v_std

# OK, now when would we actually have detected it.
def p_single(rv_jitter, rv_nb_transits):

    rv_error = jitter_to_radial_velocity_error(rv_jitter, rv_nb_transits.reshape((1, -1, 1)))

    p_single = np.empty_like(rv_jitter)
    p_single[rv_error >= 20] = np.nan
    p_single[~np.isfinite(rv_error)] = np.nan

    # Let's do something dumb and just say if it's > mu + 3 * sigma 
    # and ignore the variance in estimating both of those quantities
    detectable = (p_mu_single + 3 * p_sigma_single).reshape((-1, 1, 1))

    return rv_jitter >= detectable



p = np.sum(p_single(v_stds, rv_nb_transits_centroids), axis=(1, 2)) \
  / (rv_nb_transits_centroids.size * N)

# Now plot as a H-R diagram.
def _plot_mean_p_single(bp_rp, absolute_rp_mag, phot_rp_mean_mag, p, 
                        vmin=None, vmax=None, **kwargs):

    x = bp_rp
    y = absolute_rp_mag

    imshow_kwds = dict(vmin=vmin, vmax=vmax,
                       aspect=np.ptp(x)/np.ptp(y),
                       extent=(np.min(x), np.max(x), np.max(y), np.min(y)),
                       cmap="inferno",
                       interpolation="none")
    imshow_kwds.update(kwargs)

    H = np.mean(p.reshape((x.size, y.size, -1)), axis=-1)

    fig, ax = plt.subplots()
    image = ax.imshow(H, **imshow_kwds)

    colorbar = plt.colorbar(image, ax=ax)

    return fig


bp_rp, absolute_rp_mag, phot_rp_mean_mag = centroids


fig = _plot_mean_p_single(bp_rp, absolute_rp_mag,phot_rp_mean_mag, p)




 