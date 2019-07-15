

import itertools
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy import (coordinates as coord, units as u)
from astropy.coordinates.matrix_utilities import (matrix_product, rotation_matrix)
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



def approx_astrometric_excess_noise(t, P, m1, m2, f1, f2, distance, **kwargs):
    """
    A crude approximation for astrometric excess noise that does not account
    for orbital inclination, projection to the reference plane, and other stuff.
    """

    m_total = m1 + m2
    w = np.array([f1, f2])/(f1 + f2)
    a = twobody.P_m_to_a(P, m_total).to(u.AU).value

    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    w1, w2 = (w[0], w[1])

    # TODO: replace this with integral!
    phi = (2 * np.pi * t / P).value
    N = phi.size

    dx = a1 * w1 * np.cos(phi) + a2 * w2 * np.cos(phi + np.pi)
    dy = a1 * w1 * np.sin(phi) + a2 * w2 * np.sin(phi + np.pi)

    rms_in_au = np.sqrt(np.sum((dx - np.mean(dx))**2 + (dy - np.mean(dy))**2)/N).value
    rms_in_mas = (rms_in_au * u.au / distance).to(u.mas, equivalencies=u.dimensionless_angles())

    meta = dict(weights=w,
                a=a,
                a1=a1,
                a2=a2,
                w1=w1,
                w2=w2,
                phi=phi,
                dx=dx,
                dy=dy,
                rms_in_au=rms_in_au)

    return (rms_in_mas, meta)




def astrometric_excess_noise(t, P, m1, m2, f1=None, f2=None, e=0, t0=None,
                             omega=0*u.deg, i=0*u.deg, Omega=0*u.deg, 
                             origin=None, **kwargs):

    ## TODO: Refactor this behemoth!!

    if f1 is None:
        f1 = m1.to(u.solMass).value**3.5
    if f2 is None:
        f2 = m2.to(u.solMass).value**3.5

    if t0 is None:
        t0 = Time('J2015.5')

    N = t.size
    
    # Compute orbital positions.
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        M1 = (2*np.pi * (t.tcb - t0.tcb) / P).to(u.radian)
        # Set secondary to have Omega = 180 deg.
        M2 = (2*np.pi * (t.tcb - t0.tcb) / P - np.pi).to(u.radian)
        
    # eccentric anomaly
    E1 = twobody.eccentric_anomaly_from_mean_anomaly(M1, e)
    E2 = twobody.eccentric_anomaly_from_mean_anomaly(M2, e)

    # mean anomaly
    F1 = twobody.true_anomaly_from_eccentric_anomaly(E1, e)
    F2 = twobody.true_anomaly_from_eccentric_anomaly(E2, e)

    # Calc a1/a2.
    m_total = m1 + m2
    a = twobody.P_m_to_a(P, m_total)
    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    r1 = (a1 * (1. - e * np.cos(E1))).to(u.au).value
    r2 = (a2 * (1. - e * np.cos(E2))).to(u.au).value

    # Calculate xy positions in orbital plane.
    x = np.vstack([
        r1 * np.cos(F1),
        r2 * np.cos(F2),
    ]).value
    y = np.vstack([
        r1 * np.sin(F1),
        r2 * np.sin(F2)
    ]).value

    # Calculate photocenter in orbital plane.
    w = np.atleast_2d([f1, f2])/(f1 + f2)
    x, y = np.vstack([w @ x, w @ y])
    z = np.zeros_like(x)

    # Calculate photocenter velocities in orbital plane (necessary to take barycentric motion into
    # account).
    fac = (2*np.pi * a / P / np.sqrt(1 - e**2)).to(u.au/u.s).value
    vx = np.vstack([
        -fac * np.sin(F1),
        -fac * np.sin(F2)
    ]).value
    vy = np.vstack([
        fac * (np.cos(F1) + e),
        fac * (np.cos(F2) + e)
    ]).value
    vx, vy = np.vstack([w @ vx, w @ vy])
    vz = np.zeros_like(vx)

    # TODO: handle units better w/ dot product
    x, y, z = (x * u.au, y * u.au, z * u.au)
    vx, vy, vz = (vx * u.au/u.s, vy * u.au/u.s, vz * u.au/u.s)
    
    xyz = coord.CartesianRepresentation(x=x, y=y, z=z)
    vxyz = coord.CartesianDifferential(d_x=vx, d_y=vy, d_z=vz)
    xyz = xyz.with_differentials(vxyz)

    vxyz = xyz.differentials["s"]
    xyz = xyz.without_differentials()

    
    # Construct rotation matrix from orbital plane system to reference plane system.
    R1 = rotation_matrix(-omega, axis='z')
    R2 = rotation_matrix(i, axis='x')
    R3 = rotation_matrix(Omega, axis='z')
    Rot = matrix_product(R3, R2, R1)

    # Rotate photocenters to the reference plane system.
    XYZ = coord.CartesianRepresentation(matrix_product(Rot, xyz.xyz))
    VXYZ = coord.CartesianDifferential(matrix_product(Rot, vxyz.d_xyz))
    XYZ = XYZ.with_differentials(VXYZ)

    barycenter = twobody.Barycenter(origin=origin, t0=t0)
    kw = dict(origin=barycenter.origin)
    rp = twobody.ReferencePlaneFrame(XYZ, **kw)

    # Calculate the ICRS positions.
    icrs_cart = rp.transform_to(coord.ICRS).cartesian
    icrs_pos = icrs_cart.without_differentials()
    icrs_vel = icrs_cart.differentials["s"]

    bary_cart = barycenter.origin.cartesian
    bary_vel = bary_cart.differentials["s"]

    dt = t - barycenter.t0
    dx = (bary_vel * dt).to_cartesian()

    pos = icrs_pos + dx
    vel = icrs_vel + bary_vel

    icrs = coord.ICRS(pos.with_differentials(vel))

    # Add error to each one.
    intrinsic_error = 5.7 * 10**-6 * u.mas # arcseconds # muas * 10**-6 * u.mas
    
    N = t.size
    ra = icrs.ra.to(u.deg).value + np.random.normal(0, intrinsic_error.to(u.deg).value, size=N)
    dec = icrs.dec.to(u.deg).value + np.random.normal(0, intrinsic_error.to(u.deg).value, size=N)

    # Calc a chi^2 assuming a single position.
    v_ = np.array([ra, dec]).T

    # NOTE THIS IS NOT STRICTLY THE ASTROMETRIC CHI2 BECAUSE WE ARE ASSUMING THE
    # EFFECTS OF PROPER MOTION AND PARALLAX HAVE BEEN REMOVED!!
    # TODO HACK FIX THIS

    #astrometric_chi2_al = np.sum(((v_ - np.mean(v_, axis=0))/intrinsic_error.to(u.deg).value)**2)
    #mv = np.mean(v_, axis=0)
    #astrometric_chi2_al = np.sum((v_ - mv)**2 / mv)
    mv = np.mean(v_, axis=0)
    rms = np.sum((v_ - mv)**2 / intrinsic_error) / N

    ruwe = np.sqrt(rms/(N - 2))

    #     data=np.sqrt(group["astrometric_chi2_al"][()]/(group["astrometric_n_good_obs_al"][()] - 5)))

    # Calc a RUWE

    ra, dec = v_.T

    fig, ax = plt.subplots()
    ax.scatter(ra, dec)

    ra_mean = np.mean(ra)
    dec_mean = np.mean(dec)

    # common xlim/ylims
    limits = 1.1 * max(np.ptp(ra), np.ptp(dec))

    ax.set_xlim(ra_mean - 0.5 * limits,
                ra_mean + 0.5 * limits)
    ax.set_ylim(dec_mean - 0.5 * limits,
                dec_mean + 0.5 * limits)



    raise a


    raise a

    # Use expected Gaia errors (given G mag and VmI colour) for a single transit.
    # https://arxiv.org/pdf/1404.5861.pdf says 5.7 \mu as and some unknown CCD
    # centroid positioning error.

    # Calculate a \chi^2 
    # Calculate a RUWE


    rms_in_au = np.sqrt(np.sum((pc - np.mean(pc, axis=0))**2)/N)

    # Convert to reference plane.
    rms_in_mas = (rms_in_au * u.au / DISTANCE).to(u.mas, equivalencies=u.dimensionless_angles())

    raise a

    return (rms_in_mas, dict(rms_in_au=rms_in_au))




# From https://www.cosmos.esa.int/web/gaia/dr2
obs_start = Time('2014-07-25T10:30')
obs_end = Time('2016-05-23T11:35')

astrometric_n_good_obs_al = lambda **_: 256

# Put the sky position something where it will not wrap...
kwds = dict(i=45 * u.deg,
            omega=0 * u.deg,
            origin=coord.ICRS(ra=0.1 * u.deg,
                              dec=0 * u.deg,
                              distance=1000 * u.pc,
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


qPs = np.array([
    [0.5, 668]
])
approx_aen = np.zeros((qPs.shape[0], N_repeats), dtype=float)
actual_aen = np.zeros((qPs.shape[0], N_repeats), dtype=float)

extras = np.zeros((qPs.shape[0], N_repeats, 6), dtype=float)

for i, (q, P) in enumerate(tqdm(qPs)):

    P = P * u.day

    for j, m1 in enumerate(Ms):
        
        N = astrometric_n_good_obs_al()
        #t_actual = obs_start + np.random.uniform(size=N) * (obs_end - obs_start)
        t_actual = obs_start + np.linspace(0, 1, N) * (obs_end - obs_start)
        t_approx = (t_actual - obs_start).to(u.day)

        #t_actual = obs_start + np.linspace(0, P, N)
        #t_approx = (t_actual - obs_start).to(u.day)

        m2 = q * m1

        sim_kwds = kwds.copy()
        sim_kwds.update(P=P,
                        m1=m1, m2=m2,
                        f1=m1.to(u.solMass).value**3.5,
                        f2=m2.to(u.solMass).value**3.5)

        approx, approx_meta = approx_astrometric_excess_noise(t=t_approx, distance=10 * u.pc, **sim_kwds)
        
        approx_aen[i, j] = approx.to(u.mas).value

        actual, actual_meta = astrometric_excess_noise(t=t_actual, **sim_kwds)
        actual_aen[i, j] = actual.to(u.mas).value

        raise a

        extras[i, j, :] = [P.value, q, m1.value, m2.value, approx_meta["rms_in_au"], actual_meta["rms_in_au"]]

        print(actual, approx)

        if False and q < 0.2:


            pc_kwds = dict(s=1, c="k")

            K = 3
            fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), sharex=True, sharey=True)

            
            # Plot positions.
            s = np.linspace(1, 50, N)
            axes[1].scatter(actual_meta["x"].T[0], actual_meta["y"].T[0], s=s, c="tab:blue")
            axes[1].scatter(actual_meta["x"].T[1], actual_meta["y"].T[1], s=s, c="tab:red")

            # Plot photocenter.
            axes[0].scatter(approx_meta["dx"], approx_meta["dy"], **pc_kwds)
            axes[1].scatter(actual_meta["pc_xy"].T[0], actual_meta["pc_xy"].T[1], **pc_kwds)

            t0 = Time('J2015.5')
            # Compute orbital positions.
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                M1 = (2*np.pi * (t_actual.tcb - t0.tcb) / P).to(u.radian)
                M2 = (2*np.pi * (t_actual.tcb - t0.tcb) / P - np.pi).to(u.radian)
                
            e = 0

            # eccentric anomaly
            E1 = twobody.eccentric_anomaly_from_mean_anomaly(M1, e)
            E2 = twobody.eccentric_anomaly_from_mean_anomaly(M2, e)

            # mean anomaly
            F1 = twobody.true_anomaly_from_eccentric_anomaly(E1, e)
            F2 = twobody.true_anomaly_from_eccentric_anomaly(E2, e)

            # Calc a1/a2.
            m_total = m1 + m2
            a = twobody.P_m_to_a(P, m_total)
            a1 = m2 * a / m_total
            a2 = m1 * a / m_total

            r1 = a1 * (1. - e * np.cos(E1))
            r2 = a2 * (1. - e * np.cos(E2))

            x1 = (r1 * np.cos(F1)).to(u.au).value
            y1 = (r1 * np.sin(F1)).to(u.au).value

            x2 = (r2 * np.cos(F2)).to(u.au).value
            y2 = (r2 * np.sin(F2)).to(u.au).value

            # Calc photocenter.
            x = np.vstack([x1, x2]).T
            y = np.vstack([y1, y2]).T

            w = actual_meta["weights"]
            _pc_xy = np.vstack([w @ x.T, w @ y.T]).T
            _rms_au = np.sqrt(np.sum((_pc_xy - np.mean(_pc_xy, axis=0))**2)/N)

            foo = astrometric_excess_noise(t_actual, P, m1, m2)


            axes[2].scatter(x1, y1, s=s, c="tab:blue")
            axes[2].scatter(x2, y2, s=s, c="tab:red")

            axes[2].scatter(_pc_xy.T[0], _pc_xy.T[1], c="k")

            axvline_kwds = dict(c="#666666", linestyle=":", linewidth=0.5, zorder=-1, ms=1)
            for ax in axes:
                ax.axvline(0, **axvline_kwds)
                ax.axhline(0, **axvline_kwds)
            raise a



mean_aen = np.mean(approx_aen, axis=1).reshape((q_bins, P_bins))

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