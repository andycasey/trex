import h5py as h5

import itertools
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.table import Table
from astropy import (coordinates as coord, units as u)
from astropy.coordinates.matrix_utilities import (matrix_product, rotation_matrix)
from tqdm import tqdm
from scipy import (optimize as op)
from astropy import constants

from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import LineCollection


from corner import corner

import sys
sys.path.insert(0, "../code")
from mpl_utils import mpl_style
from de_utils import (approximate_ruwe, astrometric_excess_noise, salpeter_imf)

import twobody

plt.style.use(mpl_style)

np.random.seed(0)


def worker(j, k, **kwargs):
    ruwe, _ = approximate_ruwe(**kwargs)
    #aen, _ = astrometric_excess_noise(**kw)
    return (j, k, ruwe, np.nan)


population = "hot jupiters"


N = 10_000 # simulations per distance trial
M = 1 # time draws per simulation

populations = [
    "hot jupiters",
    "main-sequence binaries"
]

assert population in populations


# The number of astrometric matched observations in DR2 ranges from 5 to 136.
dr2_mean_astrometric_matched_observations = 28 # number of observations per source
#dr2_draw_astrometric_matched_observations = lambda _=None: int(np.clip(np.random.normal(28, 11), 5, 128))

if population == "main-sequence binaries":
    # We are using the same Ps, qs, etc at each distance trial.
    P = np.random.lognormal(5.03, 2.28, N) * u.day # Raghavan et al. (2010)
    q = np.random.uniform(0.1, 1, N)
    cos_i = np.random.uniform(0, 1, N)
    i = np.arccos(cos_i) * u.rad

    M_1 = salpeter_imf(N, 2.35, 0.1, 100) * u.solMass
    M_2 = q * M_1

    # Assume main-sequence systems.
    f_1 = M_1.value**3.5
    f_2 = M_2.value**3.5

    ylabel = r"$\textrm{detection efficiency of main-sequence binary systems}$"

elif population == "hot jupiters":
    M_1 = salpeter_imf(N, 2.35, 0.1, 100) * u.solMass
    f_1 = 1.0 * np.ones(N)

    M_2 = 10* np.ones(N) * constants.M_jup.to(u.solMass)
    f_2 = 0.0 * np.ones(N)

    cos_i = np.random.uniform(0, 1, N)
    i = np.arccos(cos_i) * u.rad

    P = np.random.uniform(5, 10, N) * u.day

    ylabel = r"$\textrm{detection efficiency of hot jupiters}$"


# Assume that we observe each system at a uniformly random time.
# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start, dr2_obs_end = (Time('2014-07-25T10:30'), Time('2016-05-23T11:35'))  #DR 2

# Let us assume that anything with RUWE above this threshold is a binary
ruwe_binarity_threshold = 3

t = obs_start + np.random.uniform(0, 1, dr2_mean_astrometric_matched_observations) * (dr2_obs_end - obs_start)

fiducial_distance = 1 * u.pc
fiducial_ruwe_dr2 = np.zeros((N, M), dtype=float)
for j in tqdm(range(N)):

    args = (P[j], M_1[j], M_2[j], fiducial_distance)
    kwds = dict(f1=f_1[j], f2=f_2[j], i=i[j])

    for k in range(M):
        fiducial_ruwe_dr2[j, k], meta = approximate_ruwe(t, *args, **kwds)


distances = np.linspace(1, 1000, 10000) * u.pc
D = distances.size

# Just store a detection completeness at each distance.
detection_efficiency_dr2 = np.zeros(D, dtype=float)
for j, distance in tqdm(enumerate(distances), total=D):
    ruwe_dr2 = fiducial_ruwe_dr2 * (fiducial_distance / distance)
    detection_efficiency_dr2[j] = np.sum(ruwe_dr2 >= ruwe_binarity_threshold)/ruwe_dr2.size
    

fig, ax = plt.subplots()
ax.plot(distances, detection_efficiency_dr2, c="k", ms=0)
#ax.plot(distances, detection_efficiency_edr3, lw=1, linestyle=":", c="#666666", ms=0, zorder=-1)
ax.set_xscale("log")
ax.set_xlabel(r"$\textrm{distance}$ $/$ $\textrm{pc}$")
ax.set_ylabel(ylabel)

fig.tight_layout()

raise a



# Gaia DR2 3425096028968232832
ra, dec = (92.95448746, 22.82574178)
gaia_ruwe = 1.45116





# From the paper (their preferred values on distance, etc).
distance = 4.230 * u.kpc
eccentricity = 0.03 # pm 0.01
period = 78.9 * u.day # \pm 0.3 * u.day
m1 = 68 * u.solMass # (+11, -13) from their paper (abstract)
m2 = 8.2 * u.solMass # (+0.9, -1.2) from their paper
f1, f2 = (0, 1)
i = 15 * u.deg # bottom right of page 2 of their paper

omega = 0 * u.deg # doesn't matter. thanks, central limit thereom!
radial_velocity = 0 * u.km/u.s # doesn't matter here

# From Gaia, unless otherwise defined:
origin_kwds = dict(ra=92.95448 * u.deg,
                   dec=22.82574 * u.deg,
                   distance=distance,
                   pm_ra_cosdec=-0.0672 * u.mas/u.yr,
                   pm_dec=-1.88867* u.mas/u.yr,
                   radial_velocity=radial_velocity)

origin = coord.ICRS(**origin_kwds)
kwds = dict(i=i, omega=omega, origin=origin)

kwds.update(P=period,
            m1=m1,
            f1=f1,
            m2=m2,
            f2=f2,
            distance=origin.distance) # ignored by astrometric_excess_noise but needed for approximating functions





# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start = Time('2014-07-25T10:30')
obs_end = Time('2016-05-23T11:35')
observing_span = obs_end - obs_start


#gost = Table.read("gost_21.3.1_652094_2019-12-05-02-26-18.csv")
#t = Time(gost["ObservationTimeAtGaia[UTC]"])




# Assume observed at uniformly random times.
t = obs_start + np.linspace(0, 1, dr2_mean_astrometric_matched_observations) * (obs_end - obs_start)

kwds.update(t=t)
#kwds.update(pm_ra_cosdec=0*u.mas/u.yr, pm_dec=0*u.mas/u.yr)
#simulate_orbit(**kwds)





lb1_approximate_ruwe, meta_ruwe = approximate_ruwe(**kwds)
lb1_aen, meta_aen = astrometric_excess_noise(**kwds)

print(f"Reported RUWE by Gaia DR2 = {gaia_ruwe:.2f}")
print(f"Approximated RUWE = {lb1_approximate_ruwe:.2f}")
print(f"Estimated AEN = {lb1_aen:.2f}")


"""
Do Approximate Bayesian Computation.

Free parameters:
- inclination angle (0, 90)
- m1 (1, 70)
- m2 (0.7, 70)
- distance (1, 5)
"""

processes = 50
#n_repeats = 1

"""
# Based on numbers from the paper.

cost_factor = 1
i_bins = np.linspace(0, 90, cost_factor * 30) # u.deg
m1_bins = np.linspace(1, 70, cost_factor * 70) # u.solMass
m2_bins = np.linspace(1, 15, cost_factor * 30) # u.solMass
distance_bins = np.linspace(2, 5, cost_factor * 30) # u.kpc
"""

SET_DISTANCE = True
if SET_DISTANCE:

    np.random.seed(42)


    sig = 3

    n_bins = 20
    n_draws = 1

    m_bh = (5.37, 1.58)
    m_comp = (0.77, 0.14)
    i_bins = np.linspace(0, 90, n_bins)
    m1_bins = np.linspace(m_bh[0] - sig * m_bh[1], m_bh[0] + sig * m_bh[1], n_bins)
    m2_bins = np.linspace(m_comp[0] - sig * m_comp[1], m_comp[0] + sig * m_comp[1], n_bins)

    distance_draws = np.random.normal(2.14, 0.35, n_draws)
    distance_draws = [1.6]

    pool = mp.Pool(processes=processes)

    results = []

    grid = np.array(list(itertools.product(i_bins, m1_bins, m2_bins)))

    grid_ruwe = np.zeros((grid.shape[0], n_draws))
    grid_aen = np.zeros((grid.shape[0], n_draws))


    for j, (i, m1, m2) in enumerate(tqdm(grid, desc="Hunting")):

        for k, distance in enumerate(distance_draws):

            okw = origin_kwds.copy()
            okw.update(distance=distance << u.kpc)

            origin = coord.ICRS(**okw)
            kwds.update(i=i << u.deg,
                        m1=m1 << u.solMass,
                        m2=m2 << u.solMass,
                        origin=origin,
                        distance=origin.distance)

            if processes == 1:
                grid_ruwe[j, k], _ = approximate_ruwe(**kwds)
                grid_aen[j, k], _ = astrometric_excess_noise(**kwds)

            else:
                results.append(pool.apply_async(worker, (j, k), kwds))


    if processes > 1:

        for each in tqdm(results, desc="Collecting"):
            j, k, ruwe, aen = each.get(timeout=1)
            grid_ruwe[j, k] = ruwe
            grid_aen[j, k] = aen


        pool.close()
        pool.join()


    #grid = grid[:, :-1]

    # Take mean from repeats?
    # TODO]
    target_ruwe = gaia_ruwe
    ruwe_tolerance = 0.05
    mask = np.abs(target_ruwe - grid_ruwe) < ruwe_tolerance

    X = grid[mask[:, 0]]


    fig = corner(X, labels=(r"$i$", r"$M_1$", r"$M_2$"),
                 range=list(zip(np.min(grid, axis=0), np.max(grid, axis=0))))
    #fig.savefig("abc.pdf", dpi=300)
    fig.savefig("fixed_distance_1.6.png", dpi=300)

    raise a


else:

    sig = 3

    n_bins = 20

    m_bh = (5.37, 1.58)
    m_comp = (0.77, 0.14)
    i_bins = np.linspace(0, 90, n_bins)
    m1_bins = np.linspace(1, 10, n_bins)
    m2_bins = np.linspace(0.5, 1.5, n_bins)
    #m2_bins = np.linspace(m_comp[0] - sig * m_comp[1], m_comp[0] + sig * m_comp[1], n_bins)
    distance_bins = np.linspace(1.6, 2.5, n_bins)

    pool = mp.Pool(processes=processes)

    results = []

    grid = np.array(list(itertools.product(i_bins, m1_bins, m2_bins, distance_bins)))

    grid_ruwe = np.zeros((grid.shape[0], 1))
    grid_aen = np.zeros((grid.shape[0], 1))


    for j, (i, m1, m2, distance) in enumerate(tqdm(grid, desc="Hunting")):

        okw = origin_kwds.copy()
        okw.update(distance=distance << u.kpc)

        origin = coord.ICRS(**okw)
        kwds.update(i=i << u.deg,
                    m1=m1 << u.solMass,
                    m2=m2 << u.solMass,
                    origin=origin,
                    distance=origin.distance)

        if processes == 1:
            grid_ruwe[j], _ = approximate_ruwe(**kwds)
            grid_aen[j], _ = astrometric_excess_noise(**kwds)

        else:
            results.append(pool.apply_async(worker, (j, 0, ), kwds))


    if processes > 1:

        for each in tqdm(results, desc="Collecting"):
            j, k, ruwe, aen = each.get(timeout=1)
            grid_ruwe[j, k] = ruwe
            grid_aen[j, k] = aen


        pool.close()
        pool.join()


    grid_ruwe = grid_ruwe[:, 0]
    grid_aen = grid_aen[:, 0]

    #grid = grid[:, :-1]

    # Take mean from repeats?
    # TODO:
    target_ruwe = gaia_ruwe
    ruwe_tolerance = 0.05
    mask = np.abs(target_ruwe - grid_ruwe) < ruwe_tolerance

    X = grid[mask]


    fig = corner(X, labels=(r"$i$", r"$M_{BH}$", r"$M_{comp}$", r"$d$"),
                 range=list(zip(np.min(grid, axis=0), np.max(grid, axis=0))))
    #fig.savefig("abc.pdf", dpi=300)
    

raise a



