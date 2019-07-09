

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

APPROXIMATIONS = []

def approx_astrometric_excess_noise(P, m1, m2, f1, f2, origin=None, **kwargs):
    print("ASSUMING FULLY SAMPLED")

    if f1 is None:
        # Assume M-to-L index of 3.5 for main-sequence stars
        f1 = m1.to(u.solMass).value**3.5

    if f2 is None:
        # Assume M-to-L index of 3.5 for main-sequence stars
        f2 = m2.to(u.solMass).value**3.5

    m_total = m1 + m2
    w = np.atleast_2d([f1, f2])/(f1 + f2)
    a = twobody.P_m_to_a(P, m_total).to(u.AU).value

    a12 = np.array([
        +(m2 * a)/m_total,
        -(m1 * a)/m_total
    ])

    N = 100
    theta = np.linspace(0, 2*np.pi, N)

    x_p = a12[0] * np.cos(theta)
    y_p = a12[0] * np.sin(theta)

    x_s = a12[1] * np.cos(theta)
    y_s = a12[1] * np.sin(theta)

    x = np.array([x_p, -x_s])
    y = np.array([y_p, -y_s])

    xpc = (w @ x)
    ypc = (w @ y)

    pos = np.vstack([xpc[0], ypc[0]])
    rms = np.sqrt(np.sum((pos - np.atleast_2d(np.mean(pos, axis=1)).T)**2)/N)

    rms2 = 10 * np.sqrt(np.sum(np.sum((pos - np.atleast_2d(np.mean(pos, axis=1)).T)**2, axis=0))/N)



    return rms2
    #rms is in AU
    #rms = (xpc - np.mean(x)


    photocenter = w * a12
    com = a12[0] * m1

    #aen = np.sqrt(np.sum((photocenter - a12)**2))
    aen = np.sqrt(np.sum(photocenter**2))

    aen = (aen * u.AU).to(u.mas, equivalencies=u.parallax()) * 1e-8
    rms = (rms * u.AU).to(u.mas, equivalencies=u.parallax()) * 1e-8


    raise a


def astrometric_excess_noise(t, P=1*u.yr, m1=1.0*u.solMass, m2=2.0*u.solMass, 
                             omega=0.0*u.deg, i=0*u.deg, f1=None, f2=None,
                             origin=None, full_output=False, **kwargs):

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
    rms = (rms * u.deg).to(u.mas)


    p_op = primary_orbit.orbital_plane(t)
    s_op = secondary_orbit.orbital_plane(t)

    ph_x = w @ np.array([p_op.x, s_op.x])
    ph_y = w @ np.array([p_op.y, s_op.y])

    ph = np.vstack([ph_x, ph_y])

    rms_au = np.sqrt(np.sum((ph.T - np.mean(ph, axis=1))**2)/t.size)



    if rms > (179 * u.deg):
        raise ValueError("select a sky position that won't wrap, or fix this")

    """
    a = twobody.P_m_to_a(P, m1 + m2)

    a1 = (m2 * a)/(m1 + m2)
    a2 = (m1 * a)/(m1 + m2)

    a12 = np.array([a1.to(u.AU).value, -a2.to(u.AU).value])

    v = (w * a12)**2

    foo = np.sqrt((primary_icrs.ra - 180 * u.deg)**2 + (primary_icrs.dec - 30 * u.deg)**2)
    """



    #approx_rms = approx_astrometric_excess_noise(P, m1, m2, f1=f1, f2=f2)
    #print(approx_rms, rms)

    # DO some approximations.

    m_total = m1 + m2
    w = np.atleast_2d([f1, f2])/(f1 + f2)
    a = twobody.P_m_to_a(P, m_total).to(u.AU).value

    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    N = 100
    theta = np.linspace(0, 2*np.pi, N)

    x_p = a1 * np.cos(theta)
    y_p = a1 * np.sin(theta)

    x_s = a2 * np.cos(theta + np.pi)
    y_s = a2 * np.sin(theta + np.pi)

    x = np.array([x_p, x_s])
    y = np.array([y_p, y_s])

    xpc = (w @ x)
    ypc = (w @ y)

    pos = np.vstack([xpc[0], ypc[0]])
    approx_rms_au = np.sqrt(np.sum((pos - np.atleast_2d(np.mean(pos, axis=1)).T)**2)/N)

    # rms in AU
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    ax = axes[0]
    ax.scatter(ph[0], ph[1], c="k", s=1)
    ax.scatter(p_op.x, p_op.y, c="tab:blue")
    ax.scatter(s_op.x, s_op.y, c="tab:red")

    ax.scatter(p_op.x[[0]], p_op.y[[0]], c="tab:blue", s=100)
    ax.scatter(s_op.x[[0]], s_op.y[[0]], c="tab:red", s=100)

    ax = axes[1]
    ax.scatter(x_p, y_p, c="tab:blue")
    ax.scatter(x_s, y_s, c="tab:red")

    ax.scatter(x_p[[0]], y_p[[0]], c="tab:blue", s=100)
    ax.scatter(x_s[[0]], y_s[[0]], c="tab:red", s=100)

    ax.scatter(xpc, ypc, c="k", s=1)


    print(rms_au, approx_rms_au)


    fig, ax = plt.subplots()
    xp, yp = (primary_icrs.ra, primary_icrs.dec)

    ax.scatter(xp, yp, c="tab:blue", s=1)

    xs, ys = (secondary_icrs.ra, secondary_icrs.dec)
    ax.scatter(xs, ys, c="tab:red", s=1)

    xpc, ypc = photocenters
    ax.scatter(xpc, ypc, c="#000000", s=1, alpha=0.5)

    def lims(vals, space=1e-5):
        v = np.hstack(vals).flatten()
        lims = (np.min(v), np.max(v))
        return (lims[0] - space, lims[1] + space)

    ax.set_xlim(lims([xs, xp]))
    ax.set_ylim(lims([ys, yp]))

    # Fit a line to the photocenter?
    line = lambda x, m, b: m*x + b
    p_opt, p_cov = op.curve_fit(line, xpc, ypc)

    xi = np.array(ax.get_xlim())
    ax.plot(xi, line(xi, *p_opt), c="g")

    ax.scatter(pos[0], pos[1], c="g", alpha=0.5)



    #raise a

    
    APPROXIMATIONS.append([
        rms_au,
        approx_rms_au, 
        P.to(u.day).value, 
        m1.to(u.solMass).value, 
        m2.to(u.solMass).value
    ])



    if not full_output:
        return rms
    
    return (rms, dict(barycenter=barycenter,
                      elements=elements,
                      primary_orbit=primary_orbit,
                      secondary_orbit=secondary_orbit,
                      primary_icrs=primary_icrs,
                      secondary_icrs=secondary_icrs,
                      photocenters=photocenters))

# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start = Time('2014-07-25T10:30')
obs_end = Time('2016-05-23T11:35')



astrometric_n_good_obs_al = lambda **_: 226



# Put the sky position something where it will not wrap...
kwds = dict(i=0 * u.deg,
            omega=0 * u.deg,
            origin=coord.ICRS(ra=1 * u.deg,
                              dec=0 * u.deg,
                              distance=1 * u.pc,
                              pm_ra_cosdec=0 * u.mas/u.yr,
                              pm_dec=0 * u.mas/u.yr,
                              radial_velocity=0 * u.km/u.s))


# Two extreme cases.

m1 = 1.0 * u.solMass
for q in (0.1, 0.9):
    m2 = q * m1

    N = 100
    t = obs_start + np.linspace(0, 1, N) * (obs_end - obs_start)

    kwds_ = kwds.copy()
    kwds_.update(P=668 * u.day, m1=m1, m2=m2, t=t, full_output=True)

    rms, meta = astrometric_excess_noise(**kwds_)

raise a



#N_repeats = 1
#Ms = salpeter_imf(N_repeats, 2.35, 0.1, 100) * u.solMass



# Sanity check.
sane = True
if not sane:

    np.random.seed(0)

    N = astrometric_n_good_obs_al()
    N = 100
    t = obs_start + np.random.uniform(size=N) * (obs_end - obs_start)
    t = obs_start + np.linspace(0, 1, N) * (obs_end - obs_start)

    kwds_ = kwds.copy()
    kwds_.update(P=668 * u.day, m1=1.5 * u.solMass, m2=0.5 * u.solMass, t=t,
                 full_output=True)

    rms, meta = astrometric_excess_noise(**kwds_)

    pc = meta["photocenters"]
    fig, ax = plt.subplots()
    x = (pc[0] - np.mean(pc[0])) * 10**7
    y = (pc[1] - np.mean(pc[1])) * 10**7
    scat = ax.scatter(x, y, c=t.mjd)

    cbar = plt.colorbar(scat)

    fig, ax = plt.subplots()
    scale = lambda _: (_ - np.mean(_)) * 10**7
    scale = lambda _: _
    xp, yp = (scale(meta["primary_icrs"].ra), scale(meta["primary_icrs"].dec))

    ax.scatter(xp, yp, c="tab:blue", s=1)

    xs, ys = scale(meta["secondary_icrs"].ra), scale(meta["secondary_icrs"].dec)
    ax.scatter(xs, ys, c="tab:red", s=1)

    xpc, ypc = meta["photocenters"]
    ax.scatter(xpc, ypc, c="#000000", s=1, alpha=0.5)

    def lims(vals, space=1e-7):
        v = np.hstack(vals).flatten()
        lims = (np.min(v), np.max(v))
        return (lims[0] - space, lims[1] + space)

    ax.set_xlim(lims([xs, xp]))
    ax.set_ylim(lims([ys, yp]))

    # Fit a line to the photocenter?
    line = lambda x, m, b: m*x + b
    p_opt, p_cov = op.curve_fit(line, xpc, ypc)

    xi = np.array(ax.get_xlim())
    ax.plot(xi, line(xi, *p_opt), c="g")

    R = rms.to(u.deg).value
    theta_i = np.linspace(0, 2*np.pi, 1000)
    xi = kwds["origin"].ra.deg + R * np.cos(theta_i)
    yi = kwds["origin"].dec.deg + R * np.sin(theta_i)

    ax.scatter(xi, yi, facecolor="g", s=1, alpha=0.5)

    raise a


"""
def Chabrier2003_imf(m):
    a = 0.0686
    b = 1.1024
    c = 0.9522
    d = 0.0192

    m = np.atleast_1d(m)
    _ = m < 1

    imf = np.zeros_like(m)
    imf[_] = (a/m[_]) * np.exp(-(np.log10(m[_]) + b)**2/c)
    imf[~_] = d * m[~_]**-2.3

    return imf
"""
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




q_bins, P_bins = (5, 5)
Ps = np.logspace(-1.5, 2.5, P_bins)
Ps = np.array([668])
qs = np.linspace(0.1, 1, q_bins)


qPs = np.array(list(itertools.product(qs, Ps)))
predicted_aen = np.zeros((qPs.shape[0], N_repeats), dtype=float)

for i, (q, P) in enumerate(tqdm(qPs)):

    for j, m1 in enumerate(Ms):
        N = astrometric_n_good_obs_al()
        
        t = obs_start + np.random.uniform(size=N) * (obs_end - obs_start)

        # Chose m1/m2 from q = m2/m1
        m2 = q * m1

        # Simulate.
        sim_kwds = kwds.copy()
        sim_kwds.update(t=t, P=P * u.day, m1=m1, m2=m2)

        predicted_aen[i, j] = astrometric_excess_noise(**sim_kwds).to(u.mas).value


APPROXIMATIONS = np.array(APPROXIMATIONS)
fig, ax = plt.subplots()
ax.scatter(APPROXIMATIONS.T[0], APPROXIMATIONS.T[1])
raise a

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