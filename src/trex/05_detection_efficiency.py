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
from scipy.stats import binned_statistic_2d
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


def worker(j, k, other_args, **kwds):
    ruwe, _ = approximate_ruwe(*other_args, **kwds)
    #aen, _ = astrometric_excess_noise(**kw)
    return (j, k, ruwe, np.nan)



N = 10_000 # simulations per distance trial
M = 1 # time draws per simulation

# Gaia Data Release:
# 2: Gaia DR 2
# 3: Gaia EDR 3
# 3.1: Gaia DR 3
gaia_data_release = 2
#population = "black hole companions"
population = "hot jupiters"

# Let us assume that anything with RUWE above this threshold is a binary
ruwe_binarity_threshold = 1.5

populations = [
    "hot jupiters",
    "main-sequence binaries",
    "black hole companions"
]

assert population in populations




def get_gaia_observing_span(data_release):

    obs_start = Time('2014-07-25T10:30') # From https://www.cosmos.esa.int/web/gaia/dr2

    if data_release == 2:
        # From https://www.cosmos.esa.int/web/gaia/dr2
        obs_end = Time('2016-05-23T11:35')
        median_astrometric_n_good_obs_al = 225

    elif data_release == 3: # EDR 3
        obs_end = Time('2017-05-25T10:30') # '34 months of data'
        dr2_obs_start, dr2_obs_end, dr2_median_astrometric_n_good_obs_al = get_gaia_observing_span(2)

        factor = (obs_end - obs_start) / (dr2_obs_end - dr2_obs_start)
        median_astrometric_n_good_obs_al = int(factor * dr2_median_astrometric_n_good_obs_al)

    elif data_release == 3.1:
        obs_end = Time('2018-05-25T10:30') # guess
        dr2_obs_start, dr2_obs_end, dr2_median_astrometric_n_good_obs_al = get_gaia_observing_span(2)

        factor = (obs_end - obs_start) / (dr2_obs_end - dr2_obs_start)
        median_astrometric_n_good_obs_al = int(factor * dr2_median_astrometric_n_good_obs_al)

    else:
        raise Exception("Ask Anthony Brown!")


    return (obs_start, obs_end, median_astrometric_n_good_obs_al)



def draw_main_sequence_binary_population(N):
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

    args = (P, M_1, M_2, f_1, f_2, i)
    ylabel = r"$\textrm{detection efficiency of main-sequence binary systems}$"

    return (args, ylabel)


def grid_main_sequence_binary_population(N_per_axis, N_simulation):

    P = np.logspace(1, 7, N_per_axis)
    q = np.linspace(0, 1, N_per_axis)

    cos_i = np.linspace(0, 1, N_simulation)
    i = np.arccos(cos_i)
    M_1 = salpeter_imf(N_simulation, 2.35, 0.1, 100)
    
    params = np.array(list(itertools.product(P, q, M_1, i)))

    P, q, M_1, i = params.T
    M_2 = q * M_1

    f_1 = M_1**3.5
    f_2 = M_2**3.5

    P = P << u.day
    M_1 = M_1 << u.solMass
    M_2 = M_2 << u.solMass
    i = i << u.rad

    args = (P, M_1, M_2, f_1, f_2, i)
    return (args, None)



def draw_hot_jupiter_host_population(N):

    M_1 = salpeter_imf(N, 2.35, 0.1, 100) * u.solMass
    f_1 = 1.0 * np.ones(N)

    M_2 = 10* np.ones(N) * constants.M_jup.to(u.solMass)
    f_2 = 0.0 * np.ones(N)

    q = M_2 / M_1

    cos_i = np.random.uniform(0, 1, N)
    i = np.arccos(cos_i) * u.rad

    P = np.random.uniform(5, 10, N) * u.day

    args = (P, M_1, M_2, f_1, f_2, i)
    ylabel = r"$\textrm{detection efficiency of hot jupiters}$"

    return (args, ylabel)


def draw_black_hole_companion_population(N):

    P = 30 * np.ones(N) * u.day
    M_1 = salpeter_imf(N, 2.35, 0.1, 100) * u.solMass
    f_1 = 1.0 * np.ones(N)

    M_2 = np.random.uniform(3, 5, N) * u.solMass
    f_2 = 0.0 * np.ones(N)

    q = M_2 / M_1

    cos_i = np.random.uniform(0, 1, N)
    i = np.arccos(cos_i) * u.rad

    args = (P, M_1, M_2, f_1, f_2, i)
    ylabel = r"$\textrm{detection efficiency of black hole companions}$"    

    return (args, ylabel)



# Assume that we observe each system at a uniformly random time.
obs_start, obs_end, median_astrometric_n_good_obs_al = get_gaia_observing_span(gaia_data_release)
t = obs_start + np.random.uniform(0, 1, median_astrometric_n_good_obs_al) * (obs_end - obs_start)
distances = np.linspace(1, 5000, 10000) * u.pc

def simulate_detection_efficiency(P, M_1, M_2, f1, f2, i, t, distances, **kwargs):

    processes = kwargs.pop("processes", 50)
    with mp.Pool(processes=processes) as pool:

        results = []

        N = P.size
        M = 1 # TODO

        fiducial_distance = 1 * u.pc
        fiducial_ruwe = np.zeros((N, M), dtype=float)

        kwds = dict()
        kwds.update(kwargs)

        #for j in tqdm(range(N)):
        for j, (P_, M_1_, M_2_, f1_, f2_, i_) in tqdm(enumerate(zip(*(P, M_1, M_2, f1, f2, i))), desc="Pooling"):
            #args = (t, P[j], M_1[j], M_2[j], fiducial_distance)
            #kwds.update(f1=f1[j], f2=f2[j], i=i[j])
            args = (t, P_, M_1_, M_2_, fiducial_distance)
            kwds.update(f1=f1_, f2=f2_, i=i_)

            for k in range(M):
                results.append(pool.apply_async(worker, (j, k, args), kwds))


        for each in tqdm(results, desc="Collecting"):
            j, k, ruwe, aen = each.get(timeout=1)
            fiducial_ruwe[j, k] = ruwe

        D = distances.size

        # Just store a detection completeness at each distance.
        detection_efficiency = np.zeros(D, dtype=float)
        for j, distance in tqdm(enumerate(distances), total=D):
            ruwe = fiducial_ruwe * (fiducial_distance / distance)
            detection_efficiency[j] = np.sum(ruwe >= ruwe_binarity_threshold)/ruwe.size
    


    return (detection_efficiency, fiducial_ruwe, fiducial_distance)


population_functions = {
    "main_sequence": draw_main_sequence_binary_population,
    "black_hole": draw_black_hole_companion_population,
    "hot_jupiter": draw_hot_jupiter_host_population
}

for desc, population_function in population_functions.items():
    print(f"Running {desc} with {population_function}")

    args, ylabel = population_function(N)

    detection_efficiency, fiducial_ruwe, fiducial_distance = simulate_detection_efficiency(*args, t, distances)

    # optimistic modelling and EDR3
    edr3_obs_start, edr3_obs_end, edr3_median_astrometric_n_good_obs_al = get_gaia_observing_span(3)
    edr3_t = edr3_obs_start + np.random.uniform(0, 1, edr3_median_astrometric_n_good_obs_al) * (edr3_obs_end - edr3_obs_start)
    edr3_detection_efficiency, *_ = simulate_detection_efficiency(*args, edr3_t, distances,
                                                                  intrinsic_ra_error=0.06,
                                                                  intrinsic_dec_error=0.06)


    fig, ax = plt.subplots()
    ax.plot(distances, detection_efficiency, c="k", ms=0)
    ax.plot(distances, edr3_detection_efficiency, c="#666666", linestyle=":", ms=0, zorder=-1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\textrm{distance}$ $/$ $\textrm{pc}$")
    ax.set_ylabel(ylabel)

    fig.tight_layout()

    fig.savefig(f"detection_efficiency_{desc}.pdf", dpi=300)


    # 
    """
    args, ylabel = draw_main_sequence_binary_population(1_000_000)
    P, M_1, M_2, f_1, f_2, i = args

    detection_efficiency, fiducial_ruwe, fiducial_distance = simulate_detection_efficiency(*args, t, distances)

    e = 0

    q = M_2/M_1
    a = ((P/(2 * np.pi))**2 * constants.G * (M_1 + M_2))**(1/3)
    K = ((2 * np.pi * a * np.sin(i))/(P * np.sqrt(1 - e**2))).to(u.km/u.s)

    # Bin it into K, q and do contours.
    bins = (
        np.logspace(1, 6, 30),
        np.linspace(0, 1, 30)
    )

    ruwe = fiducial_ruwe.flatten() / 100.0

    B = (ruwe >= ruwe_binarity_threshold).astype(float)

    binned_statistic_args = (P.to(u.day).value, q.value, B)
    binned_statistic_kwds = dict(bins=bins)


    sum_, xedges, yedges, binnumber = binned_statistic_2d(*binned_statistic_args, 
                                                           statistic="sum",
                                                           **binned_statistic_kwds)

    count, xedges, yedges, binnumber = binned_statistic_2d(*binned_statistic_args, 
                                                           statistic="count",
                                                           **binned_statistic_kwds)
    
    H = sum_/count


    extent = (bins[0][0], bins[0][-1], bins[1][-1], bins[1][0])

    fig, ax = plt.subplots()
    ax.imshow(H.T, extent=extent)
    ax.set_xscale("log")
    ax.set_ylim(ax.get_ylim()[::-1])

    X1, Y = np.meshgrid(xedges, yedges)
    X = np.tile(np.linspace(xedges[0], xedges[-1], 30), 30).reshape((30, 30))

    levels = np.linspace(0, 1, 5)
    fig, ax = plt.subplots()
    contour = ax.contour(X[:-1, :-1], Y[:-1, :-1], H.T[:, ::-1], levels)
    ax.set_xscale("log")
    ax.clabel(contour, inline=1, fontsize=10)
    """
    N_per_axis = 30
    N_simulation = 50
    args, _ = grid_main_sequence_binary_population(N_per_axis, N_simulation)

    P, M_1, M_2, f_1, f_2, i = args

    detection_efficiency, fiducial_ruwe, fiducial_distance = simulate_detection_efficiency(*args, t, distances)

    unique_P = np.unique(P.to(u.day).value)
    unique_q = np.unique(q.value)

    bins = (unique_P, unique_q)

    H = (ruwe >= ruwe_binarity_threshold).astype(float)

    binned_statistic_args = (unique_P, unique_q, H)
    binned_statistic_kwds = dict(bins=bins)

    sum_, xe, ye, bin_number = binned_statistic_2d(*binned_statistic_args,
                                                   statistic="sum",
                                                   **binned_statistic_kwds)

    count_, xe, ye, bin_number = binned_statistic_2d(*binned_statistic_args,
                                                     statistic="count",
                                                     **binned_statistic_kwds)

    Q = sum_/count_
    extent = (bins[0][0], bins[0][-1], bins[1][-1], bins[1][0])

    fig, ax = plt.subplots()
    ax.imshow(Q.T, extent=extent)
    ax.set_xscale("log")
    


    raise a

    
    raise a

