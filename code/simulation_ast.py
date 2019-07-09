

import itertools
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import (coordinates as coord, units as u)
from tqdm import tqdm
from scipy import (optimize as op)
from astropy import constants

from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_utils import mpl_style
from matplotlib.collections import LineCollection

import twobody

plt.style.use(mpl_style)


np.random.seed(0)

# For a range of:
# (1) periods.
# (2) mass ratios

# At fixed:
# (1) distance
# (2) omega
# (3) observed N random times
# (4) proper motions
# (5) vrad

# Need to visualise what fraction of sources (at given P/q)
# we would recover as binaries at:
# i = Y degrees
# distance = X

# Some important ntoes:
# [1] The magnitude of the AEN signal scales linearly with distance.
# [2] Inferences are independent of sky position, motion, or omega
# [3] Probably need a better sampling of m1/m2/q
# [4] Need better sampling of N_astrometric_obs: allow t to be matrix?
# [5] Do inference simulations
# [6] When do stars get removed entirely because astrometric solution is large?

# Do simulations at:
# i = 0
# i = 45
# i = 90


def approx_astrometric_excess_noise(t, P, m1, m2, f1=None, f2=None, **kwargs):

    if f1 is None:
        # Assume M-to-L index of 3.5 for main-sequence stars
        f1 = m1.to(u.solMass).value**3.5

    if f2 is None:
        # Assume M-to-L index of 3.5 for main-sequence stars
        f2 = m2.to(u.solMass).value**3.5


    m_total = m1 + m2
    w = np.array([f1, f2])/(f1 + f2)
    a = twobody.P_m_to_a(P, m_total).to(u.AU).value

    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    w1, w2 = (w[0], w[1])

    # TODO: replace this with integral!
    phi = 2 * np.pi * t / P
    N = phi.size

    dx = a1 * w1 * np.cos(phi) + a2 * w2 * np.cos(phi + np.pi)
    dy = a1 * w1 * np.sin(phi) + a2 * w2 * np.sin(phi + np.pi)

    approx_rms_in_au = np.sqrt(np.sum((dx - np.mean(dx))**2 + (dy - np.mean(dy))**2)/N).value
    approx_rms_in_mas = (approx_rms_in_au * u.au / (10 * u.pc)).to(u.mas, equivalencies=u.dimensionless_angles())

    return approx_rms_in_mas


def actual_astrometric_excess_noise(t, P, m1, m2, f1=None, f2=None, **kwargs):
    
    if f1 is None:
        # Assume M-to-L index of 3.5 for main-sequence stars
        f1 = m1.to(u.solMass).value**3.5

    if f2 is None:
        # Assume M-to-L index of 3.5 for main-sequence stars
        f2 = m2.to(u.solMass).value**3.5

    barycenter = twobody.Barycenter(origin=origin, t0=Time('J2015.5'))

    elements = twobody.TwoBodyKeplerElements(P=P, m1=m1, m2=m2, omega=omega, i=i)

    primary_orbit = twobody.KeplerOrbit(elements.primary, barycenter=barycenter)
    secondary_orbit = twobody.KeplerOrbit(elements.secondary, barycenter=barycenter)
    secondary_orbit.M0 = 180 * u.deg

    # Calculate the ICRS positions.
    primary_icrs = primary_orbit.icrs(t)
    secondary_icrs = secondary_orbit.icrs(t)

    # Position of the primary + a fraction towards the position of the secondary.
    w = np.atleast_2d([f1, f2])/(f1 + f2)
    photocenter_ra = w @ np.array([primary_icrs.ra, secondary_icrs.ra])
    photocenter_dec = w @ np.array([primary_icrs.dec, secondary_icrs.dec])

    # The AEN is ~ the rms distance on sky.
    photocenters = np.vstack([photocenter_ra, photocenter_dec])

    #rms = (np.std(photocenters.T - np.mean(photocenters, axis=1)) * u.deg).to(u.mas)
    rms = np.sqrt(np.sum((photocenters.T - np.mean(photocenters, axis=1))**2)/t.size)
    rms_in_mas = (rms * u.deg).to(u.mas)

    return rms_in_mas


# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start = Time('2014-07-25T10:30')
obs_end = Time('2016-05-23T11:35')

astrometric_n_good_obs_al = lambda **_: 250

# Put the sky position something where it will not wrap...
kwds = dict(i=0 * u.deg,
            omega=0 * u.deg,
            origin=coord.ICRS(ra=0.1 * u.deg,
                              dec=0 * u.deg,
                              distance=100 * u.pc,
                              pm_ra_cosdec=0 * u.mas/u.yr,
                              pm_dec=0 * u.mas/u.yr,
                              radial_velocity=0 * u.km/u.s))




def salpeter_imf(N, alpha, M_min, M_max):
    log_M_limits = np.log([M_min, M_max])

    max_ll = M_min**(1.0 - alpha)

    M = []
    while len(M) < N:
        Mi = np.exp(np.random.uniform(*log_M_limits))

        ln = Mi**(1 - alpha)
        if np.random.uniform(0, max_ll) < ln:
            M.append(Mi)

    return np.array(M)




N_repeats = 10
Ms = salpeter_imf(N_repeats, 2.35, 0.1, 100) * u.solMass


q_bins, P_bins = (20, 20)
Ps = np.logspace(-1.5, 4, P_bins)
qs = np.linspace(0.1, 1, q_bins)


qPs = np.array(list(itertools.product(qs, Ps)))
approx_aen = np.zeros((qPs.shape[0], N_repeats), dtype=float)
actual_aen = np.zeros((qPs.shape[0], N_repeats), dtype=float)


for i, (q, P) in enumerate(tqdm(qPs)):

    for j, m1 in enumerate(Ms):
        
        """
        N = astrometric_n_good_obs_al()
        
        t = obs_start + np.random.uniform(size=N) * (obs_end - obs_start)

        # Chose m1/m2 from q = m2/m1
        m2 = q * m1

        # Simulate.
        sim_kwds = kwds.copy()
        sim_kwds.update(t=t, P=P * u.day, m1=m1, m2=m2)

        predicted_aen[i, j] = astrometric_excess_noise(**sim_kwds).to(u.mas).value
        """
        N = astrometric_n_good_obs_al()
        t = obs_start + np.random.uniform(size=N) * (obs_end - obs_start)

        m2 = q * m1

        sim_kwds = kwds.copy()
        sim_kwds.update(t=t, P=P * u.day, m1=m1, m2=m2)

        approx_aen[i, j] = approx_astrometric_excess_noise(**sim_kwds).to(u.mas).value
        actual_aen[i, j] = actual_astrometric_excess_noise(**sim_kwds).to(u.mas).value




mean_aen = np.mean(predicted_aen, axis=1).reshape((q_bins, P_bins))

# Plot per Q first.
cmap = cm.viridis(qs)
fig, ax = plt.subplots()

lc = LineCollection([np.column_stack([Ps, ma]) for ma in mean_aen])

lc.set_array(np.asarray(qs))
ax.add_collection(lc)
ax.autoscale()


#    print(i, q, np.mean(mean_aen[i]))
cbar = plt.colorbar(lc)
cbar.set_label(r"$q$")



ax.set_xlabel(r"{period / days}$^{-1}$")
ax.set_ylabel(r"{AEN / mas}")
ax.semilogx()

v = (obs_end - obs_start).to(u.day).value
axvline_kwds = dict(c="#666666", zorder=-1, lw=1, ms=1)
ax.axvline(v, linestyle=":", **axvline_kwds)
ax.axvline(2 * v, linestyle="-", **axvline_kwds)


print(kwds)



qm, Pm = np.meshgrid(qs, Ps)

contourf_kwds = dict(cmap="magma", norm=LogNorm(), levels=None)

fig, ax = plt.subplots()
im = ax.contourf(Ps, qs, mean_aen, **contourf_kwds)
ax.semilogx()


ax.set_xlabel(r"{period / days}$^{-1}$")
ax.set_ylabel(r"$q$")

cbar = plt.colorbar(im)
cbar.set_label(r"{AEN / mas}")

fig.tight_layout()

axvline_kwds.update(zorder=10)

ax.axvline(v, linestyle=":", **axvline_kwds)
ax.axvline(2 * v, linestyle="-", **axvline_kwds)


print(kwds)
print(np.min(mean_aen), np.max(mean_aen))