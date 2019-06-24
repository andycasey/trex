

# log P period distribution
# mass ratio
# distr of separations
# inclination distribution
# luminosity ratio
# draws at 10 pc, 1 kpc etc

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import (coordinates as coord, units as u)

import twobody
from mpl_utils import mpl_style

plt.style.use(mpl_style)


# P, m1, m2, inclination angles.

np.random.seed(123)

ra = np.random.uniform(0, 360) * u.deg
dec = np.random.uniform(-90, 90) * u.deg
vrad = np.random.normal(0, 100) * u.km/u.s


origin = coord.ICRS(
    ra=ra,
    dec=dec,
    distance=1 * u.pc,
    pm_ra_cosdec=0 * u.mas/u.yr,
    pm_dec=0 * u.mas/u.yr,
    radial_velocity=vrad)

t = Time('J2010') + np.linspace(0, 5, 50) * u.yr


def simulate(P=1*u.yr, m1=1.0*u.solMass, m2=2.0*u.solMass, omega=0.0*u.deg,
             i=0*u.deg, f1=10, f2=130, origin=None):


    barycenter = twobody.Barycenter(origin=origin, t0=Time('J2015.5'))

    elements = twobody.TwoBodyKeplerElements(P=P, m1=m1, m2=m2, omega=omega, i=i)

    primary_orbit = twobody.KeplerOrbit(elements.primary, barycenter=barycenter)
    secondary_orbit = twobody.KeplerOrbit(elements.secondary, barycenter=barycenter)

    # Calculate the ICRS positions.
    primary_icrs = primary_orbit.icrs(t)
    secondary_icrs = secondary_orbit.icrs(t)

    # Position of the primary + a fraction towards the position of the secondary.
    w = np.atleast_2d([f1, f2])/(f1 + f2)
    photocenter_ra = w @ np.array([primary_icrs.ra, secondary_icrs.ra])
    photocenter_dec = w @ np.array([primary_icrs.dec, secondary_icrs.dec])
    
    # The AEN is ~ the rms distance on sky.
    values = np.vstack([photocenter_ra, photocenter_dec])
    rms = (np.std(values.T - np.mean(values, axis=1)) * u.deg).to(u.mas)

    return rms


kwds = [
    dict(m1=1.5 * u.solMass, m2=1.3 * u.solMass, P=1.0 * u.yr, f1=1.1, f2=1),
    dict(m1=1.0 * u.solMass, m2=1.0 * u.solMass, P=1.0 * u.yr, f1=1, f2=1),
]

#f1/f2 = 10^((mag_1-mag_2)/2.5)
#mag_a - mag_b = -2.5log(fa/fb)

distance = np.logspace(1, 3, 100) * u.pc
rms = np.zeros((len(kwds), distance.size)) * u.mas

for i in range(distance.size):

    origin = coord.ICRS(
        ra=ra,
        dec=dec,
        distance=distance[i],
        pm_ra_cosdec=0 * u.mas/u.yr,
        pm_dec=0 * u.mas/u.yr,
        radial_velocity=vrad)

    for j, kwd in enumerate(kwds):
        rms[j, i] = simulate(origin=origin, **kwd)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.plot(distance, rms[0], "-", lw=2, label=r"$m_1/m_2 = 1.15$", c="tab:blue")
ax.plot(distance, rms[1], "-", lw=2, label=r"$m_1/m_2 = 1$", c="tab:red")

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(6))
ax.set_xlabel(r"$\textrm{distance}$ $/$ $\textrm{pc}$")
ax.set_ylabel(r"$\textrm{predicted astrometric rms}$ $/$ $\textrm{mas}$")

ax.legend(frameon=False)
ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))
fig.tight_layout()

fig.savefig("binary-represent.png", dpi=300)

raise a


# x axis is distance
# three curves: m1/m2 = 1, m1/m2 = 1.1, m1/m2 = 0.9


#F1 = 0
#f2 = 1
# m = 1/(f2 - f1)

# y = mx + c = 
# c = y/mx
# 
x = np.array([f1, f2])
yerr = np.ones_like(x)
y = np.array([1, 0])



foo, bar = simulate()

raise a


inclined = simulate(i=0.0 * u.deg)
planar = simulate(i=90*u.deg)

fig, ax = plt.subplots()
ax.scatter(inclined.ra - planar.ra, inclined.dec - planar.dec)


diff = np.array([inclined.ra - planar.ra, inclined.dec - planar.dec])

# What is the angular separation?
