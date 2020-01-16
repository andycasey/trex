import h5py as h5

import itertools
import numpy as np
import pickle 
import os
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
from matplotlib.ticker import MaxNLocator


#from corner import corner

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
    q = np.linspace(0.1, 1, N_per_axis)

    cos_i = np.linspace(0, 1, N_simulation)
    i = np.arccos(cos_i)
    #cos_i = np.array([1.0])
    #i = np.arccos(cos_i)
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

def grid_black_hole_companion_population(N_per_axis):

    P = np.logspace(0.5, 2, N_per_axis)
    M_star = salpeter_imf(N_per_axis, 2.35, 0.1, 100)
    M_bh = np.random.uniform(3, 5, N_per_axis) 
    
    v = np.array([M_bh, M_star])
    M_1 = np.max(v, axis=0)
    M_2 = np.min(v, axis=0)
    
    cos_i = np.random.uniform(0, 1, N_per_axis)
    i = np.arccos(cos_i)

    params = np.array(list(itertools.product(P, M_1, M_2, i)))

    P, M_1, M_2, i = params.T

    # always assume the more massive one is the bh
    f_1 = np.zeros(M_1.size)
    f_2 = np.ones(M_1.size)

    P = P << u.day
    M_1 = M_1 << u.solMass
    M_2 = M_2 << u.solMass
    i = i << u.rad

    args = (P, M_1, M_2, f_1, f_2, i)
    return (args, None)





# Assume that we observe each system at a uniformly random time.
obs_start, obs_end, median_astrometric_n_good_obs_al = get_gaia_observing_span(gaia_data_release)
t = obs_start + np.random.uniform(0, 1, median_astrometric_n_good_obs_al) * (obs_end - obs_start)
distances = np.linspace(1, 5000, 10000) * u.pc

def simulate_ruwe_at_fiducial_distance(P, M_1, M_2, f1, f2, i, t, **kwargs):

    processes = kwargs.pop("processes", 50)
    with mp.Pool(processes=processes) as pool:

        results = []

        N = P.size
        M = 1 # TODO

        fiducial_distance = 1 * u.pc
        fiducial_ruwe = np.zeros((N, M), dtype=float)

        kwds = dict()
        kwds.update(kwargs)

        for j, (P_, M_1_, M_2_, f1_, f2_, i_) in tqdm(enumerate(zip(*(P, M_1, M_2, f1, f2, i))), 
                                                      desc="Pooling", total=P.size):

            args = (t, P_, M_1_, M_2_, fiducial_distance)
            kwds.update(f1=f1_, f2=f2_, i=i_)

            for k in range(M):
                results.append(pool.apply_async(worker, (j, k, args), kwds))


        for each in tqdm(results, desc="Collecting"):
            j, k, ruwe, aen = each.get(timeout=5)
            fiducial_ruwe[j, k] = ruwe

        """
        D = distances.size

        # Just store a detection completeness at each distance.
        detection_efficiency = np.zeros(D, dtype=float)
        for j, distance in tqdm(enumerate(distances), total=D):
            ruwe = fiducial_ruwe * (fiducial_distance / distance)
            detection_efficiency[j] = np.sum(ruwe >= ruwe_binarity_threshold)/ruwe.size
        """


    return (fiducial_ruwe, fiducial_distance)


def _get_bins(x, y, x_log, y_log, x_lim=None, y_lim=None, N_per_axis=None):
    
    N_per_axis = N_per_axis or np.min([np.unique(xy).size for xy in (x, y)])
    xr = (x.min(), x.max())
    yr = (y.min(), y.max())
    if x_lim is not None:
        xr = np.clip(xr, *np.sort(x_lim))
    if y_lim is not None:
        yr = np.clip(yr, *np.sort(y_lim))

    if x_log:
        x_space_args = np.log10(xr)
        x_space_func = np.logspace
    else:
        x_space_args = xr
        x_space_func = np.linspace

    if y_log:
        y_space_args = np.log10(yr)
        y_space_func = np.logspace
    else:
        y_space_args = yr
        y_space_func = np.linspace

    x_bins = x_space_func(*x_space_args, 1 + N_per_axis)
    y_bins = y_space_func(*y_space_args, 1 + N_per_axis)

    return (x_bins, y_bins)



def plot_mesh(x, y, z, statistic="mean", x_log=False, y_log=False, **kwargs):

    x_lim, y_lim = [kwargs.pop(f"{_}_lim", None) for _ in "xy"]
    x_label, y_label = [kwargs.pop(f"{_}_label", None) for _ in "xy"]
    
    N_per_axis = kwargs.pop("N_per_axis", np.min([np.unique(xy).size for xy in (x, y)]))

    bin_args = (x, y, x_log, y_log, x_lim, y_lim)
    bins = _get_bins(*bin_args, N_per_axis)

    counts, xe, ye, bin_number = binned_statistic_2d(x, y, z, statistic=statistic, bins=bins)

    kwds = dict()
    kwds.update(kwargs)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(xe, ye, counts.T, **kwds)
    cbar = plt.colorbar(pcm)
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    fig.tight_layout()
    return fig




def plot_detection_efficiency_contours(x, y, detection_efficiency, x_log=False, y_log=False, fill_value=None, 
                                       axes=None, **kwargs):

    # Pop out kwargs
    N_per_axis = kwargs.pop("N_per_axis", np.min([np.unique(xy).size for xy in (x, y)]))
    x_lim = kwargs.pop("x_lim", None)
    y_lim = kwargs.pop("y_lim", None)
    x_label = kwargs.pop("x_label", None)
    y_label = kwargs.pop("y_label", None)
    clabel_kwds = kwargs.pop("clabel_kwds", dict())
    full_output = kwargs.pop("full_output", False)

    bin_args = (x, y, x_log, y_log, x_lim, y_lim)
    x_bins, y_bins = _get_bins(*bin_args, N_per_axis)
    x_unique, y_unique = _get_bins(*bin_args, N_per_axis - 1)

    args = (x, y, detection_efficiency)
    kwds = dict(bins=(x_bins, y_bins))

    numer, xe, ye, bin_number = binned_statistic_2d(*args, statistic="sum", **kwds)
    denom, xe, ye, bin_number = binned_statistic_2d(*args, statistic="count", **kwds)

    Q = numer/denom
    if fill_value is not None:
        Q[~np.isfinite(Q)] = fill_value

    extent = (xe[0], xe[-1], ye[0], ye[-1])
    
    if axes is None:
        fig, ax = plt.subplots()
        twin_ax = ax.twinx()

    else:
        if isinstance(axes, tuple):
            ax, twin_ax = axes
        else:
            ax = axes
            twin_ax = ax.twinx() 
        fig = ax.figure

    set_kwds = dict()
    if x_log: set_kwds.update(xscale="log")
    if y_log: set_kwds.update(yscale="log")
    ax.set(**set_kwds)

    filled = kwargs.pop("filled", False)

    contour_kwds = dict(levels=[0, 0.25, 0.5, 0.75, 0.99, 1.0])
    contour_kwds.update(kwargs)

    if filled:
        twin_ax.contourf(x_unique, y_unique, Q.T, **contour_kwds)
    contour = twin_ax.contour(x_unique, y_unique, Q.T, **contour_kwds)

    if kwargs.get("contour_labels", False):
        clabel_kwds.setdefault("inline", True)
        clabel_kwds.setdefault("fontsize", 10)
        clabel_kwds.setdefault("fmt", "%1.1f")
        ax.clabel(contour, **clabel_kwds)
        
    ax.set_xlim(twin_ax.get_xlim())
    ax.set_ylim(twin_ax.get_ylim())

    for _ in (ax, twin_ax):
        _.set_xlim(x_lim)
        _.set_ylim(y_lim)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    twin_ax.set_yticklabels([])
    twin_ax.yaxis.set_tick_params(width=0)

    fig.tight_layout()

    return fig if not full_output else (fig, ax, twin_ax)




N_per_axis = 30
N_simulation = 100

print(f"Anticipated number of combinations: {(N_per_axis * N_simulation)**2:.1e}")

ms_path = f"grid_main_sequence_binary_population_{N_per_axis}_{N_simulation}.pkl"

if not os.path.exists(ms_path):
    print("Running MS simulation")
    args, _ = grid_main_sequence_binary_population(N_per_axis, N_simulation)

    P, M_1, M_2, f_1, f_2, i = args

    fiducial_ruwe, fiducial_distance = simulate_ruwe_at_fiducial_distance(*args, t)
    content = (args, fiducial_ruwe, fiducial_distance)

    with open(ms_path, "wb") as fp:
        pickle.dump(content, fp)


bh_path = f"grid_dark_passengers_binary_population_{N_per_axis}_{N_simulation}.pkl"
if not os.path.exists(bh_path):
    print("Running BH simulation")

    args, _ = grid_black_hole_companion_population(N_per_axis)
    P, M_1, M_2, f_1, f_2, i = args

    fiducial_ruwe, fiducial_distance = simulate_ruwe_at_fiducial_distance(*args, t)
    content = (args, fiducial_ruwe, fiducial_distance)

    with open(bh_path, "wb") as fp:
        pickle.dump(content, fp)


chosen_path = bh_path    


print(f"Loading from {chosen_path}")
with open(chosen_path, "rb") as fp:
    content = pickle.load(fp)

args, fiducial_ruwe, fiducial_distance = content
P, M_1, M_2, f_1, f_2, i = args



# Here's where we calculate things.
q = M_2/M_1

unique_P = np.unique(P.to(u.day).value)
unique_q = np.unique(q.value)

bins = (unique_P, unique_q)

# Set conditions.

at_distances = (10, 100, 1000)

master_fig, master_axes = plt.subplots(2, len(at_distances), 
                                       figsize=(10, 5))
master_axes = np.array(master_axes).flatten()

for d, at_distance in enumerate(at_distances):
    at_distance *= u.pc

    # Let us assume that anything with RUWE above this threshold is a binary
    ruwe_binarity_threshold = 1.5
    K_lower_threshold = 3 * u.km/u.s # approx
    K_upper_threshold = 20 * u.km/u.s # approx

    e = 0

    a = ((P/(2 * np.pi))**2 * constants.G * (M_1 + M_2))**(1/3)
    K = ((2 * np.pi * a * np.sin(i))/(P * np.sqrt(1 - e**2))).to(u.km/u.s)
    ruwe = (fiducial_ruwe * (fiducial_distance / at_distance)).value.flatten()

    de_ast = (ruwe >= ruwe_binarity_threshold).astype(float).flatten()

    # TODO: de_rv depends on the number of observed transits
    de_rv = ((K_upper_threshold >= K) * (K >= K_lower_threshold)).astype(float).flatten()
    de_rv_nocut = ((K >= K_lower_threshold)).astype(float).flatten()


    P_label = r"$P$ $/$ $\textrm{days}^{-1}$"
    K_label = r"$K$ $/$ $\textrm{km\,s}^{-1}$"
    q_label = r"$q$"

    levels = np.linspace(0, 1, 5)

    PK_plot_kwds = dict(x_log=True, y_log=False,
                        x_lim=None, y_lim=(0, 50),
                        fill_value=0,
                        levels=levels,
                        filled=False, contour_labels=True,
                        clabel_kwds=dict(fmt="%1.2f"),
                        x_label=None, y_label=K_label)

    Pq_plot_kwds = dict(x_log=True, y_log=False,
                        x_lim=None, y_lim=(0, 1),
                        fill_value=0,
                        levels=levels,
                        filled=False, contour_labels=True,
                        clabel_kwds=dict(fmt="%1.2f"),
                        x_label=P_label, y_label=q_label)

    ax_PK = master_axes[d]
    ax_Pq = master_axes[d + len(at_distances)]

    # Plot detection efficiency contours from astrometry.
    title = r"$\textrm{{at}}$ ${{{0:.0f}}}$ $\textrm{{pc}}$".format(at_distance.to(u.pc).value)
    *_, twin_ax = plot_detection_efficiency_contours(P.value, q.value, de_ast,
                                       cmap="Blues", axes=ax_Pq,
                                       full_output=True,
                                       **Pq_plot_kwds)
    ax_Pq.plot(ax_Pq.get_xlim(), [q.value[0], q.value[0]],
               "-", c="#666666", linestyle=":", lw=1, zorder=-1)



    # On the same figure, plot RV.
    plot_detection_efficiency_contours(P.value, q.value, de_rv,
                                    cmap="Purples", axes=ax_Pq, 
                                    zorder=-1,
                                    **Pq_plot_kwds)


    # Now plot P/K
    *_, twin_ax = plot_detection_efficiency_contours(P.value, K.value, de_ast,
                                       cmap="Blues", axes=ax_PK,
                                       full_output=True,
                                       **PK_plot_kwds)
    plot_detection_efficiency_contours(P.value, K.value, de_rv,
                                       cmap="Purples", axes=ax_PK, 
                                       zorder=-1,
                                       **PK_plot_kwds)
    twin_ax.set_xticklabels([])

    xi = np.logspace(*np.log10(P.value[[0, -1]]), 100)
    # Assume best conditions: low inclination and eccentricity ~ 0.8. Where is the limit of our simulations?
    yi = ((1/np.sqrt(1-0.8**2)) * ((4 * np.pi * constants.G * np.max(M_1))/(x * u.day))**(1/3)).to(u.km/u.s).value
    ax_PK.plot(xi, yi, "-", linestyle=":", c="#666666", lw=1)

    ax_PK.set_title(title)



for ax in master_axes:
    if ax.is_first_col():
        ax.yaxis.set_major_locator(MaxNLocator(5))
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])

master_fig.tight_layout()




# show me in P, K space where we have things
fig = plot_mesh(P.to(u.day).value, K.to(u.km/u.s).value, ruwe,
                statistic="min", x_log=True,
                x_label=P_label, y_label=K_label)



# Sanity check
fig = plot_mesh(P.to(u.day).value, q.value, ruwe,
                **Pq_plot_kwds)


# Plot detection efficiency contours in astrometry.


# Sanity plot
fig = plot_mesh(P.to(u.day).value, K.to(u.km/u.s).value, K,
                statistic="max",
                **PK_plot_kwds)

fig = plot_mesh(P.to(u.day).value, K.to(u.km/u.s).value, de_rv,
                statistic="mean",
                **PK_plot_kwds)

fig = plot_detection_efficiency_contours(P.value, K.value, de_ast,
                                         fill_value=0,
                                         **PK_plot_kwds)

fig = plot_mesh(P.value, K.value, ruwe, statistic="mean", **PK_plot_kwds)

x = np.logspace(*np.log10(P.to(u.day).value[[0, -1]]), 100)
# Assume best conditions: low inclination and zero eccentricity. Where is the limit of our simulations?
y = (((4 * np.pi * constants.G * np.max(M_1))/(x * u.day))**(1/3)).to(u.km/u.s).value
ax = fig.axes[0]
ax.plot(x, y, "-", linestyle=":", c="#666666", lw=1)



fig = plot_mesh(P.value, q.value, K,
                statistic="mean",
                **Pq_plot_kwds)



fig = plot_detection_efficiency_contours(P.to(u.day).value, q.value, de_rv,
                                         **Pq_plot_kwds)

raise a

fig = plot_mesh(P.to(u.day).value, K.to(u.km/u.s).value, ruwe,
                x_log=True, y_log=False,
                x_label=P_label, y_label=K_label,
                x_lim=None, y_lim=(0, 25), statistic="mean")#vmin=0, vmax=3)


fig = plot_detection_efficiency_contours(P.to(u.day).value, K.to(u.km/u.s).value, de_ast,
                                         x_log=True,
                                         x_label=P_label,
                                         y_label=K_label,
                                         x_lim=None, y_lim=(0, 25))#(0, 10))





raise a


population_functions = {
    "main_sequence": draw_main_sequence_binary_population,
    "black_hole": draw_black_hole_companion_population,
    "hot_jupiter": draw_hot_jupiter_host_population
}

for desc, population_function in population_functions.items():
    print(f"Running {desc} with {population_function}")

    args, ylabel = population_function(N)

    fiducial_ruwe, fiducial_distance = simulate_ruwe_at_fiducial_distance(*args, t)

    raise a
    # optimistic modelling and EDR3
    edr3_obs_start, edr3_obs_end, edr3_median_astrometric_n_good_obs_al = get_gaia_observing_span(3)
    edr3_t = edr3_obs_start + np.random.uniform(0, 1, edr3_median_astrometric_n_good_obs_al) * (edr3_obs_end - edr3_obs_start)
    edr3_detection_efficiency, *_ = simulate_ruwe_at_fiducial_distance(*args, edr3_t, distances,
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

    detection_efficiency, fiducial_ruwe, fiducial_distance = simulate_ruwe_at_fiducial_distance(*args, t, distances)

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


    