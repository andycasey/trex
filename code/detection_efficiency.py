
""" Simulate the efficiency in detecting binaries through radial velocity variations. """

import os
import itertools # dat feel
import operator
import functools
import numpy as np
import h5py as h5
import pickle
from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from scipy import special, integrate
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import twobody

def simulate_rv_jitter(Ks, Ps, N, observing_span, rv_nb_transits_callable=None, varphi_callable=None, 
                       intrinsic_rv_error_callable=None, time_observed_callable=None):

    default_rv_nb_transits_callable = lambda *_: np.random.randint(3, 30)
    default_varphi_callable = lambda *_: np.random.uniform(0, 2 * np.pi)
    default_intrinsic_rv_error_callable = lambda T, *_: 0.1 * np.ones(T)
    default_time_observed_callable = lambda T, *_: np.random.uniform(0, observing_span.to(u.day).value, T)

    # Set some default callables.
    rv_nb_transits_callable = rv_nb_transits_callable or default_rv_nb_transits_callable
    varphi_callable = varphi_callable or default_varphi_callable
    intrinsic_rv_error_callable = intrinsic_rv_error_callable or default_intrinsic_rv_error_callable
    time_observed_callable = time_observed_callable or default_time_observed_callable

    # todo: consider non-ciruclar orbits
    rv_callable = lambda t, K, P, varphi: K * np.sin(2 * np.pi * t / P + varphi)

    # Set up arrays for the pain.    
    KPs = np.array(list(itertools.product(Ks, Ps)))
    v_stds = np.empty((KPs.shape[0], N))
    Ts = np.empty_like(v_stds)

    for i, (K, P) in enumerate(tqdm(KPs)):
        for j in range(N):

            args = (K, P, j)
            # number of rv nb transits
            T = rv_nb_transits_callable(*args)
            t = time_observed_callable(T, *args)

            varphi = varphi_callable(*args)

            rv = rv_callable(t, K, P, varphi)
            noise = np.random.normal(0, 1, T) * intrinsic_rv_error_callable(T, *args)

            Ts[i, j] = T
            v_stds[i, j] = np.std(rv + noise)
    
    return (v_stds, Ts)


def plot_simulated_rv_jitter(Ks, Ps, v_stds, data=None, observing_span=None, 
                             lower_detection_bound=None, upper_detection_bound=None,
                             vmin=1e-2, vmax=1e3, average_func=np.median):

    Z = average_func(v_stds, axis=1).reshape((Ks.size, Ps.size))

    steps = np.log10(vmax) - np.log10(vmin)
    levels = np.logspace(np.log10(vmin), np.log10(vmax), 1 + 1 * steps)

    contourf_kwds = dict(norm=LogNorm(vmin=vmin, vmax=vmax), cmap="Blues", levels=levels)
    #contourf_kwds = dict(cmap="Blues", vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots()
    im = ax.contourf(Ps, Ks, Z, **contourf_kwds)
    ax.loglog()

    cbar = plt.colorbar(im)
    cbar.set_label(r"$\sigma_\mathrm{rv}$ / km\,s$^{-1}$")

    ax.set_xlabel(r"{period / days}")
    ax.set_ylabel(r"$K$ / km\,s$^{-1}$")

    fig.tight_layout()

    # Show observing span.
    limit_color = "#000000"
    line_kwds = dict(c=limit_color, zorder=10, lw=1)
    if observing_span is not None:
        ax.axvline(observing_span.to(u.day).value, linestyle="-", **line_kwds)
        ax.axvline(2 * observing_span.to(u.day).value, linestyle=":", **line_kwds)

    if upper_detection_bound is not None:

        contour_args = (Ps, Ks, (Z >= upper_detection_bound.to(u.km/u.s).value).astype(int))
        #ax.contourf(*contour_args, 3, hatches=['', 'xx'], lw=0.5, colors="white", alpha=0)
        ax.contour(*contour_args, 1, lw=1.5, colors=limit_color)

    if lower_detection_bound is not None:
        ax.axhline(lower_detection_bound.to(u.km/u.s).value, **line_kwds)

    xlim, ylim = (ax.get_xlim(), ax.get_ylim())
    if data is not None:
        data_kwds = dict(zorder=10, s=1, c="#000000")
        ax.scatter(data["Per"], data["K1"], **data_kwds)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return fig




def p_detectable(v_std, Ts, a=15, b=1.5, c=0.5):
    p = np.random.beta(a, b, size=v_std.shape)
    p[v_std <= c] = 0
    p[v_std > upper_detection_bound.to(u.km/u.s).value] = 0
    return p


def plot_crude_rv_detection_efficiency(Ks, Ps, v_stds, Ts, p_detectable_callable=None, 
                                       data=None, observing_span=None, 
                                       lower_detection_bound=None, upper_detection_bound=None):

    p_detectable_callable = p_detectable_callable or p_detectable
    Z = np.sum(p_detectable_callable(v_stds, Ts), axis=1)
    Z = Z.reshape((Ks.size, Ps.size)) / v_stds.shape[1]

    fig, ax = plt.subplots()
    #im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=1,
    #               extent=(Ps[0], Ps[-1], Ks[0], Ks[-1]))
    K, P = np.meshgrid(Ks, Ps)
    im = ax.pcolor(P, K, Z.T, cmap="Blues", vmin=0, vmax=1)
    ax.loglog()

    cbar = plt.colorbar(im)
    cbar.set_label(r"{detection efficiency}")

    ax.set_xlabel(r"{period / days}")
    ax.set_ylabel(r"$K$ / km\,s$^{-1}$")

    limit_color = "#000000"
    line_kwds = dict(c=limit_color, zorder=10, lw=1)
    if observing_span is not None:
        ax.axvline(observing_span.to(u.day).value, linestyle="-", **line_kwds)
        ax.axvline(2 * observing_span.to(u.day).value, linestyle=":", **line_kwds)

    if upper_detection_bound is not None:
        ZZ = np.mean(v_stds, axis=1).reshape((Ks.size, Ps.size))
        contour_args = (Ps, Ks, (ZZ >= upper_detection_bound.to(u.km/u.s).value).astype(int))
        #ax.contourf(*contour_args, 3, hatches=['', 'xx'], lw=0.5, colors="white", alpha=0)
        ax.contour(*contour_args, 1, lw=1.5, colors=limit_color)

    if lower_detection_bound is not None:
        ax.axhline(lower_detection_bound.to(u.km/u.s).value, **line_kwds)

    xlim, ylim = (ax.get_xlim(), ax.get_ylim())
    if data is not None:
        data_kwds = dict(zorder=10, s=1, c="#000000")
        ax.scatter(data["Per"], data["K1"], **data_kwds)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()

    return fig


def approximate_astrometric_ruwe(t, P, m1, m2, distance, f1=None, f2=None, t0=None, 
                                 i=0*u.deg, psi=0, **kwargs):
    """
    Approximate the on-sky astrometric excess noise for a binary system with the
    given system parameters at a certain distance.

    This approximating function ignores the following effects:

    (1) The distortions that arise due to sky projections.
    (2) Inclination effects.
    (3) Omega effects.

    In part it also assumes:

    (1) The times were observed pseudo-randomly.
    (2) The orbit is fully sampled.

    :param t:
        The times that the system was observed.

    :param P:
        The period of the binary system.

    :param m1:
        The mass of the primary star.

    :param m2:
        The mass of the secondary system.

    :param distance:
        The distance from the observer to the center of mass of the binary
        system.

    :param f1: [optional]
        The flux of the primary star. If `None` is given then this is assumed to
        be $m_1^{3.5}$.

    :param f2: [optional]
        The flux of the secondary. If `None` is given then this is assumed to be
        $m_2^{3.5}$.

    :returns:
        A two-part tuple containing the root-mean-squared deviations in on-sky
        position (in units of milliarcseconds), and a dictionary containing meta
        information about the binary system.
    """

    if f1 is None:
        f1 = m1.to(u.solMass).value**3.5
    if f2 is None:
        f2 = m2.to(u.solMass).value**3.5

    if t0 is None:
        t0 = Time('J2015.5')

    m_total = m1 + m2
    w = np.array([f1, f2])/(f1 + f2)
    a = twobody.P_m_to_a(P, m_total).to(u.AU).value

    a1 = m2 * a / m_total
    a2 = m1 * a / m_total

    w1, w2 = (w[0], w[1])

    # Put this randomly out of phase
    #psi = np.random.uniform(0, 2 * np.pi)
    psi = 0

    # TODO: replace this with integral!
    dt = (t - t0).to(u.day)
    phi = (2 * np.pi * dt / P).value + psi
    N = phi.size

    dx = a1 * w1 * np.cos(phi) + a2 * w2 * np.cos(phi + np.pi)
    dy = a1 * w1 * np.sin(phi) + a2 * w2 * np.sin(phi + np.pi)

    planar_rms_in_au = np.sqrt(np.sum((dx - np.mean(dx))**2 + (dy - np.mean(dy))**2)/N).value

    # Need some corrections for when the period is longer than the observing timespan, and the
    # inclination angle is non-zero.

    # For this it really depends on what t0/Omega is: if you see half the orbit in one phase or
    # another...
    # TODO: this requires a thinko.
    
    """
    Approximate given some inclination angle.
    At zero inclination, assume circle on sky such that:
    
        rms = sqrt(ds^2 + ds^2) = sqrt(2ds^2)

    and 
        
        ds = np.sqrt(0.5 * rms^2)

    Now when inclined (even at 90) we still get ds + contribution:

        rms_new = sqrt(ds^2 + (cos(i) * ds)^2)
    """

    ds = np.sqrt(0.5 * planar_rms_in_au**2)
    rms_in_au = np.sqrt(ds**2 + (np.cos(i) * ds)**2)
    rms_in_mas = (rms_in_au * u.au / distance).to(u.mas, equivalencies=u.dimensionless_angles())

    # Intrinsic error on position in one direction is.
    # These are the final values. The individual epochs are probably about a 10th of this.
    intrinsic_ra_error = 0.029 # mas
    intrinsic_dec_error = 0.026 # mas

    intrinsic_ra_error /= 10
    intrinsic_dec_error /= 10

    chi2 = N * rms_in_mas.to(u.mas).value**2 / np.sqrt(intrinsic_ra_error**2 + intrinsic_dec_error**2)

    approx_ruwe = np.sqrt(chi2/(N - 2))

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

    return (approx_ruwe, meta)




def simulate_astrometric_ruwe(qs, Ps, Ds, M_1, T, f1_callable=None, f2_callable=None,
                              meta_callable=None):

    N = len(M_1)
    iters = (qs, Ps, Ds)
    shape = list(map(len, iters))
    C = functools.reduce(operator.mul, shape, 1)

    ruwe = np.empty((*shape, N))
    meta = []

    for _, (q, P, D) in enumerate(tqdm(itertools.product(*iters), total=C)):
        for l in range(N):

            args = (q, P, D, l)

            M_2_ = q * M_1[l]
            f1 = f1_callable(M_1[l], q, P, D)
            f2 = f1_callable(M_2_, q, P, D)

            ruwe_, meta_, = approximate_astrometric_ruwe(T, P, M_1[l], M_2_, D, f1=f1, f2=f2)
            if meta_callable is not None:
                meta.append(meta_callable(meta_))

            i, j, k = np.unravel_index(_, shape)
            ruwe[i, j, k, l] = ruwe_

    return (ruwe, meta)




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



def _plot_ast_quantity(qs, Ps, Ds, ruwe, callable, observing_span=None, vmin=None, vmax=None):

    N_D = len(Ds)
    Q = np.zeros((qs.size, Ps.size, Ds.size), dtype=float)
    figs = []

    for i, D in enumerate(Ds):

        Q[:, :, i] = callable(ruwe[:, :, i]) / ruwe.shape[1]

        fig, ax = plt.subplots()
        q, P = np.meshgrid(qs, Ps)
        im = ax.pcolor(P.value, q, Q[:, :, i].T, vmin=vmin, vmax=vmax)
        ax.set_xscale("log")

        cbar = plt.colorbar(im)
        ax.set_xlabel(r"{period / days}")
        ax.set_ylabel(r"$q$")

        limit_color = "#000000"
        line_kwds = dict(c=limit_color, zorder=10, lw=1)
        if observing_span is not None:
            ax.axvline(observing_span.to(u.day).value, linestyle="-", **line_kwds)
            ax.axvline(2 * observing_span.to(u.day).value, linestyle=":", **line_kwds)
            ax.axvline(4 * observing_span.to(u.day).value, linestyle="-.", **line_kwds)

        fig.tight_layout()
        figs.append((fig, cbar))

    return (figs, Q)


def plot_mean_ruwe(qs, Ps, Ds, ruwe, observing_span=None, **kwargs):

    figs_, Q = _plot_ast_quantity(qs, Ps, Ds, ruwe, lambda R: np.mean(R, axis=-1),
                                  observing_span=observing_span, **kwargs)
    figs = []
    for fig, cbar in figs_:
        cbar.set_label(r"{mean RUWE}")
        figs.append(fig)

    return figs


def plot_crude_ast_detection_efficiency(qs, Ps, Ds, ruwe, p_detectable_callable, observing_span=None):

    f = lambda R: np.sum(p_detectable_callable(R), axis=-1)
    figs_, DE = _plot_ast_quantity(qs, Ps, Ds, ruwe, f,
                                   vmin=0, vmax=1, observing_span=observing_span)
    figs = []
    for fig, cbar in figs_:
        cbar.set_label(r"{detection efficiency}")
        figs.append(fig)

    return (figs, DE)


if __name__ == "__main__":

    here = os.path.dirname(__file__)

    do_rv_detection_efficiency = False
    do_ast_detection_efficiency = False

    np.random.seed(42)

    log_P_min, log_P_max = (-1.5, 6) # log_10(P / day)
    log_K_min, log_K_max = (-0.5, 3) # log_10(K / km/s)

    N_rv_bins = 25 # number of bins to do in P and K
    N_rv_sims = 100 # number of simulations to do per P, K bin
    Ks = np.logspace(log_K_min, log_K_max, N_rv_bins)
    Ps = np.logspace(log_P_min, log_P_max, N_rv_bins)

    # Assume that we observe each system at a uniformly random time.
    # From https://www.cosmos.esa.int/web/gaia/dr2
    observing_start, observing_end = (Time('2014-07-25T10:30'), Time('2016-05-23T11:35')) 
    observing_span = (observing_end - observing_start).to(u.day) # ~ 668 days

    # Prepare to simulate astrometric detection efficency given P and q.
    N_ast_bins = 20 # number of bins per P, q, and D
    N_ast_sims = 100 # number of mass simulations per P, q, D bin
    N_ast_obs = 200 # number of astrometric observations per simulation

    q_min, q_max = (0.1, 1)
    M_min, M_max = (0.1, 100) # sol masses
    log_D_min, log_D_max = (1, 3) # log_10(D / pc)
    # log_P ranges as per before

    qs = np.linspace(q_min, q_max, N_ast_bins)
    qs = np.linspace(q_min, q_max, 20)
    print("Doing non-standard qs")
    Ps = np.logspace(log_P_min, log_P_max, 30) * u.day
    Ds = np.logspace(log_D_min, log_D_max, N_ast_bins) * u.pc
    Ds = np.array([10]) * u.pc
    print(f"ONLY DOING SOME DISTANCES {Ds} AND WE WILL SCALE LINEARLY WITH DISTANCE")
    M_1 = salpeter_imf(N_ast_sims, 2.35, M_min, M_max) * u.solMass
    #M_1 = np.array([1.]) * u.solMass
    #print(f"ONLY DOING SOME MASSES {M_1}")


    # Assume main-sequence systems such that f \propto M^3.5
    f1_callable = lambda M, *_: M.value**3.5
    f2_callable = lambda M, *_: M.value**3.5

    T = observing_start + np.random.uniform(0, 1, N_ast_obs) * (observing_end - observing_start)

    if do_rv_detection_efficiency:

        print("Calculating RV detection efficiency!")

        v_stds, Ts = simulate_rv_jitter(Ks, Ps, N_rv_sims, observing_span)

        scalar = np.sqrt(16.5 * np.pi * 2)
        scalar = 1.5 * np.sqrt(2 * np.pi) 
        lower_detection_bound = 0.5 * u.km/u.s
        upper_detection_bound = 20 * scalar * u.km / u.s 

        def p_detectable(v_std, Ts, a=15, b=1.5, c=0.5):
            p = np.random.beta(a, b, size=v_std.shape)
            p[v_std <= c] = 0
            p[v_std > upper_detection_bound.to(u.km/u.s).value] = 0
            return p

        data = Table.read("../data/Pourbaix-et-al-sb9-subset.csv")

        fig = plot_simulated_rv_jitter(Ks, Ps, v_stds,
                                       data=data, observing_span=observing_span,
                                       lower_detection_bound=lower_detection_bound,
                                       upper_detection_bound=upper_detection_bound)

        fig = plot_crude_rv_detection_efficiency(Ks, Ps, v_stds, Ts,
                                                 p_detectable_callable=p_detectable,
                                                 data=data, observing_span=observing_span,
                                                 lower_detection_bound=lower_detection_bound,
                                                 upper_detection_bound=upper_detection_bound)

    else:
        print("Not doing RV detection efficiency!")

    # Will need some representative sample of sources (e.g. apparent magnitude and distance, etc)

    if do_ast_detection_efficiency:
        print("Calculating astrometric detection efficiency!")

        ruwe, meta = simulate_astrometric_ruwe(qs, Ps, Ds, M_1, T, 
                                               f1_callable=f1_callable, f2_callable=f2_callable,
                                               meta_callable=None)

        ruwe_detectable = lambda ruwe, *_: ruwe >= 5

        args = (qs, Ps, Ds, ruwe)
        figs_mean = plot_mean_ruwe(qs, Ps, Ds, np.log10(ruwe), observing_span=observing_span)

        figs, DE = plot_crude_ast_detection_efficiency(*args, observing_span=observing_span,
                                                       p_detectable_callable=ruwe_detectable)


    else:
        print("Not doing astrometric detection efficiency!")


    """
    Simulate detection efficiency properly.

    1. Given distributions of:

        a. P, q, e ~ 0, M_1

        b. Create M_2

    2. Create histogram of: bp_rp, absolute_g_mag, apparent_g_mag

    3. At each grid point in (bp_rp, absolute_g_mag, apparent_g_mag):

        4. Estimate RUWE at some fiducial distance, assuming same intrinsic errors in measuring positions

        5. Scale to the distances in that bin.

        6. Simulate detection efficiency in astrometry given some {w, mu_single, sigma_single, sigma_multiple} and the variances in those properties.

    6. For every bin over multiple simulations:
        
        a. Sum up the number of stars, S.

        c. Sum up the number of binaries, B.

        d. Calculate the true binary fraction, T_B = S / DE, using detection efficiency DE.

        e. Add to the total number of binaries, T_B.

        f. Calculate \sum_{TB} / N sources.
    """    

    def periods(N, log_P_mu=4.8, log_P_sigma=2.3, log_P_min=-2, log_P_max=12):

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

    def mass_ratios(M_1):
        N = len(M_1)
        q = np.empty(N)
        for i, M in enumerate(M_1):
            q[i] = np.random.uniform(0.1 / M.value, 1)
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


    np.random.seed(0)

    detection_efficiency_ms_ms_path = os.path.join(here, "detection-efficiency-ms-ms.pkl")
    if os.path.exists(detection_efficiency_ms_ms_path):
        with open(detection_efficiency_ms_ms_path, "rb") as fp:
            meta = pickle.load(fp)
            q, P, D, M_1, T, ruwe = (meta["q"], meta["P"], meta["D"], meta["M_1"], meta["T"], meta["ruwe"])
            N = q.size


    else:
        N = 100
        D = 10 * u.pc
        P = periods(N) * u.day
        M_1 = salpeter_imf(N) * u.solMass
        q = mass_ratios(M_1)
        M_2 = q * M_1

        # Let us assume that the f1/f2_callables do not vary.
        # Simulate RUWE values for some fiducial distance.
        # Assume main-sequence systems such that f \propto M^3.5
        f1_callable = lambda M, *_: M.value**3.5
        f2_callable = lambda M, *_: M.value**3.5

        T = observing_start + np.random.uniform(0, 1, N_ast_obs) * (observing_end - observing_start)

        # Now for each bin point, calculate the GP predictions for w, mu_s, sigma_s, sigma_m.
        ruwe, meta = simulate_astrometric_ruwe(q, P, np.atleast_1d(D), M_1, T, 
                                               f1_callable=f1_callable, f2_callable=f2_callable,
                                               meta_callable=None)

        # Save the results of this as part of f1_callable, f2_callable.
        with open(detection_efficiency_ms_ms_path, "wb") as fp:
            pickle.dump(dict(q=q, P=P, D=D, M_1=M_1, T=T, ruwe=ruwe), fp)


    # Create a histogram of (bp_rp, absolute_g_mag, apparent_g_mag)
    sources = h5.File("../data/sources.hdf5", "r")

    N_bins = 20


    # At each bin point, calculate the GP parameters for w, mu_s, sigma_s, etc..
    # At each bin point, take some RUWE values and calculate p_single draws


