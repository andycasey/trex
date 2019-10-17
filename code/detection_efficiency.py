
""" Simulate the efficiency in detecting binaries through radial velocity variations. """

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
from astropy.table import Table
from astropy import units as u
from astropy.time import Time
from scipy import special, integrate
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import twobody
import utils

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
        ax.contour(*contour_args, 0, lw=1.5, colors=limit_color)

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
        ax.contour(*contour_args, 0, lw=1.5, colors=limit_color)

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

    results = h5.File("../results/rc.7/results-5482.h5", "r")

    config = yaml.load(results.attrs["config"], Loader=yaml.Loader)

    pwd = os.path.dirname(results.attrs["config_path"]).decode("utf-8")
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")
    sources = data["sources"]



    do_rv_detection_efficiency = True
    do_ast_detection_efficiency = False

    np.random.seed(42)

    log_P_min, log_P_max = (-1.5, 6) # log_10(P / day)
    log_K_min, log_K_max = (-0.5, 3) # log_10(K / km/s)

    N_rv_bins = 50 # number of bins to do in P and K
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

        print("Calculating RV detection efficiency on a grid!")

        v_stds, Ts = simulate_rv_jitter(Ks, Ps.value, N_rv_sims, observing_span)

        scalar = np.sqrt(16.5 * np.pi * 2)
        scalar = 1.5 * np.sqrt(2 * np.pi) 
        lower_detection_bound = 1 * u.km/u.s
        upper_detection_bound = 20 * scalar * u.km / u.s 

        def p_detectable(v_std, Ts, a=15, b=1.5, c=0.5):
            p = np.random.beta(a, b, size=v_std.shape)
            p[v_std <= c] = 0
            p[v_std > upper_detection_bound.to(u.km/u.s).value] = 0
            return p

        data = Table.read("../data/Pourbaix-et-al-sb9-subset.csv")

        fig = plot_simulated_rv_jitter(Ks, Ps.value, v_stds,
                                       data=data, observing_span=observing_span,
                                       lower_detection_bound=lower_detection_bound,
                                       upper_detection_bound=upper_detection_bound)

        fig = plot_crude_rv_detection_efficiency(Ks, Ps.value, v_stds, Ts,
                                                 p_detectable_callable=p_detectable,
                                                 data=data, observing_span=observing_span,
                                                 lower_detection_bound=lower_detection_bound,
                                                 upper_detection_bound=upper_detection_bound)


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


        

        foo = np.log10(np.array([data[ln] for ln in ("K1", "Per")]))
        foo = foo[:, foo[0] > -0.25]
        mu = np.mean(foo, axis=1)
        rho = np.corrcoef(foo)
        var = np.atleast_2d(np.var(foo, axis=1))

        # inflate variance
        var *= 5
        cov = np.sqrt(var) * rho * np.sqrt(var).T
        #cov = np.cov(foo)

        draws = np.random.multivariate_normal(mu, cov, 1000)

        fig, ax = plt.subplots()
        ax.scatter(foo[0], foo[1], c='k')
        ax.scatter(draws.T[0], draws.T[1], c='#666666', alpha=0.5)



        # inflate variance
        N = 1000
        periods = draw_periods(1000)

        for p in periods:
            ax.axvline(np.log10(p), c="b", alpha=0.1, lw=0.5)

        # Assign K values to periods.
        Ks = np.zeros_like(periods)
        for i, period in enumerate(periods):
            idx = np.argmin(np.abs(np.log10(period) - draws.T[0]))
            Ks[i] = 10**draws[idx, 1]
            draws[idx, 0] = -100

        fig, ax = plt.subplots()
        ax.scatter(periods, Ks)
        ax.loglog()






        raise a


        # Use *mean* RV detection efficiency.
        theta, _ = np.mean(results["models/rv/gp_predictions/theta"][()], axis=0)
        mu_single, _ = np.mean(results["models/rv/gp_predictions/mu_single"][()], axis=0)
        sigma_single, _ = np.mean(results["models/rv/gp_predictions/sigma_single"][()], axis=0)
        sigma_multiple, _ = np.mean(results["models/rv/gp_predictions/sigma_multiple"][()], axis=0)


        def calc_p_single(y, theta, mu_single, sigma_single, sigma_multiple, mu_multiple_scalar):

            with warnings.catch_warnings(): 
                # I'll log whatever number I want python you can't tell me what to do
                warnings.simplefilter("ignore") 

                mu_multiple = np.log(mu_single + mu_multiple_scalar * sigma_single) + sigma_multiple**2
                
                ln_s = np.log(theta) + utils.normal_lpdf(y, mu_single, sigma_single)
                ln_m = np.log(1-theta) + utils.lognormal_lpdf(y, mu_multiple, sigma_multiple)

                # FIX BAD SUPPORT.

                # This is a BAD MAGIC HACK where we are just going to flip things.
                """
                limit = mu_single - 2 * sigma_single
                bad_support = (y <= limit) * (ln_m > ln_s)
                ln_s_bs = np.copy(ln_s[bad_support])
                ln_m_bs = np.copy(ln_m[bad_support])
                ln_s[bad_support] = ln_m_bs
                ln_m[bad_support] = ln_s_bs
                """
                ln_s = np.atleast_1d(ln_s)
                ln_m = np.atleast_1d(ln_m)

                lp = np.array([ln_s, ln_m]).T

                #assert np.all(np.isfinite(lp))

                p_single = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

            return (p_single, ln_s, ln_m)

        x = np.sort(v_stds.flatten())
        p_single, ln_s, ln_m = calc_p_single(x, theta, mu_single, sigma_single, sigma_multiple,
                                             mu_multiple_scalar=3)

        # Do many draws.
        K = 100
        p_single_draws = np.empty((K, p_single.size))
        Q = len(results["models/rv/gp_predictions/theta"])

        for k in tqdm(range(K), desc="Drawing RVs"):
            index = np.random.choice(Q)
            theta = results["models/rv/gp_predictions/theta"][()][index, 0]
            mu_single = results["models/rv/gp_predictions/mu_single"][()][index, 0]
            sigma_single = results["models/rv/gp_predictions/sigma_single"][()][index, 0]
            sigma_multiple = results["models/rv/gp_predictions/sigma_multiple"][()][index, 0]

            p_single_draws[k], *_ = calc_p_single(x, theta, mu_single, sigma_single, sigma_multiple,
                                                  mu_multiple_scalar=3)



        fig, ax = plt.subplots()
        #ax.plot(x, p_single, lw=2, c="#000000")
        #for k in range(K):
        #    ax.plot(x, p_single_draws[k], alpha=0.1, c="#666666", lw=1, zorder=-1)

        ax.set_xscale("log")
        ax.set_ylabel(r"$p(\textrm{single}|\sigma_{rv})$")
        ax.set_xlabel(r"$\sigma_{rv}$ / km\,s$^{-1}$")


        raise a




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


    raise a

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




    lns = config["models"][model_name]["kdtree_label_names"]
    X = np.array([sources[ln][()] for ln in lns]).T
    finite = np.all(np.isfinite(X), axis=1)

    N_data_bins = 20
    H, edges = np.histogramdd(X[finite], bins=N_data_bins)
    centroids = [e[:-1] + 0.5 * np.diff(e) for e in edges]


    # For each bin, make GP predictions of the bin centroid.
    parameter_names = list(results[f"models/{model_name}/gp_model"].keys())[2:]
    gps = {}
    for parameter_name in parameter_names:

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
        gps[parameter_name] = (gp, Y)

    # Now make predictions at each centroid.
    mg = np.array(np.meshgrid(*centroids)).reshape((len(lns), -1)).T

    gp_predictions = np.empty((mg.shape[0], 4, 2))
    gp_predictions[:] = np.nan

    for i, parameter_name in enumerate(parameter_names):

        gp, Y = gps[parameter_name]
        p, p_var = gp.predict(Y, mg, return_cov=False, return_var=True)
        gp_predictions[:, i, 0] = p
        gp_predictions[:, i, 1] = p_var

        print(i, parameter_name)

    # Calculate the p_single values we would get for all RUWE values, given the
    # GP predictions in each bin.
    percentiles = [5, 16, 50, 84, 95]



    def clipped_predictions(theta, mu_single, sigma_single, sigma_multiple, bounds):
        theta = np.clip(theta, *bounds["theta"])
        mu_single = np.clip(mu_single, *bounds["mu_single"])
        sigma_single = np.clip(sigma_single, *bounds["sigma_single"])
        sigma_multiple = np.clip(sigma_single, *bounds["sigma_multiple"])

        return np.array([theta, mu_single, sigma_single, sigma_multiple])


    def calc_p_single(y, theta, mu_single, sigma_single, sigma_multiple, mu_multiple_scalar):

        with warnings.catch_warnings(): 
            # I'll log whatever number I want python you can't tell me what to do
            warnings.simplefilter("ignore") 

            mu_multiple = np.log(mu_single + mu_multiple_scalar * sigma_single) + sigma_multiple**2
            
            ln_s = np.log(theta) + utils.normal_lpdf(y, mu_single, sigma_single)
            ln_m = np.log(1-theta) + utils.lognormal_lpdf(y, mu_multiple, sigma_multiple)

            # FIX BAD SUPPORT.

            # This is a BAD MAGIC HACK where we are just going to flip things.
            """
            limit = mu_single - 2 * sigma_single
            bad_support = (y <= limit) * (ln_m > ln_s)
            ln_s_bs = np.copy(ln_s[bad_support])
            ln_m_bs = np.copy(ln_m[bad_support])
            ln_s[bad_support] = ln_m_bs
            ln_m[bad_support] = ln_s_bs
            """
            ln_s = np.atleast_1d(ln_s)
            ln_m = np.atleast_1d(ln_m)

            lp = np.array([ln_s, ln_m]).T

            #assert np.all(np.isfinite(lp))

            p_single = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

        return (p_single, ln_s, ln_m)


    def calc_probs(ys, gp_predictions, bounds, N_draws, percentiles):

        ys = np.atleast_2d(ys)
        Q, M = ys.shape

        shape = (Q, M + 1, N_draws)
        obj_p_singles = np.empty(shape)
        obj_ln_s = np.empty(shape)
        obj_ln_m = np.empty(shape)

        # gp_predictions has shape M, 4, 2
        mu = gp_predictions[:, :, 0]
        var = gp_predictions[:, :, 1]

        for q, ym in enumerate(tqdm(ys)):

            for m, y in enumerate(ym):

                # Do draws for all models
                _slice = (q, m, slice(0, None))
                if not np.all(np.isfinite(np.hstack([y, mu[m], var[m]]))):
                    obj_p_singles[_slice] = np.nan
                    obj_ln_s[_slice] = np.nan
                    obj_ln_s[_slice] = np.nan
                    continue

                draws = clipped_predictions(
                    *np.random.multivariate_normal(mu[m], np.diag(var[m]), size=N_draws).T, 
                    bounds[m])

                obj_p_singles[_slice], obj_ln_s[_slice], obj_ln_m[_slice] \
                    = calc_p_single(y, *draws, mu_multiple_scalar=3)

        # Now calculate the joint probabilities
        obj_ln_s[:, -1, :] = np.nansum(obj_ln_s[:, :-1, :], axis=1)
        obj_ln_m[:, -1, :] = np.nansum(obj_ln_m[:, :-1, :], axis=1)

        lp = np.array([obj_ln_s[:, -1, :], obj_ln_m[:, -1, :]])

        obj_p_singles[:, -1, :] = np.exp(lp[0] - special.logsumexp(lp, axis=0))

        # Then calculate percentiles ofthose probabilities.
        obj_p_singles[~np.isfinite(obj_p_singles)] = -1
        p = np.percentile(obj_p_singles, percentiles, axis=-1).transpose((1, 0, 2))
        p[p == -1] = np.nan
        return p



    SMALL = 1e-5
    b = config["models"]["ast"]["bounds"]
    b.update(theta=[0, 1])
    b["theta"] = np.array(b["theta"]) + [+SMALL, -SMALL]
    bounds = [b]


    #p_50 = np.memmap("temp.np", dtype=float, mode="w+", shape=(mg.shape[0], len(percentiles)))
    p_single_50 = np.empty((mg.shape[0], len(percentiles)))

    v = np.atleast_2d(np.random.choice(ruwe.flatten(), 1000)).T

    for i in tqdm(range(mg.shape[0])):
        foo = calc_probs(v, gp_predictions[[i]], bounds, 128, percentiles)

        raise a

    raise a


    # Calculate the p_single values we would get for all RUWE values, given the
    # GP predictions at that bin.

    # Store either p_50, or something else?





    '''
    # Ignore what follows: I was testing if I could calculate completeness for every source!


    def _cross_match(A_source_ids, B_source_ids):

        A = np.array(A_source_ids, dtype=np.long)
        B = np.array(B_source_ids, dtype=np.long)

        ai = np.where(np.in1d(A, B))[0]
        bi = np.where(np.in1d(B, A))[0]
        
        a_idx, b_idx = (ai[np.argsort(A[ai])], bi[np.argsort(B[bi])])

        # Sanity checks
        assert a_idx.size == b_idx.size
        assert np.all(A[a_idx] == B[b_idx])
        return (a_idx, b_idx)




    print("Preparing for completeness calculations")
    
    SMALL = 1e-5

    results = h5.File("../results/rc.7/results-5482.h5", "r")

    config = yaml.load(results.attrs["config"], Loader=yaml.Loader)

    pwd = os.path.dirname(results.attrs["config_path"]).decode("utf-8")
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")
    sources = data["sources"]


    N = len(results["results/source_id"])
    model_names = results["models"].keys()

    # Get all bounds.
    M = len(model_names)

    # Build some big ass arrays
    ys = np.empty((N, M))
    ys[:] = np.nan

    bounds = []

    gp_predictions = np.empty((N, M, 4, 2))
    gp_predictions[:] = np.nan

    source_indices = results["results"]["source_indices"][()]

    for m, (model_name, model_config) in enumerate(config["models"].items()):

        predictor_label_name = model_config["predictor_label_name"]

        model_source_indices = results[f"models/{model_name}/gp_predictions/source_indices"][()]

        a_idx, b_idx = _cross_match(source_indices, model_source_indices)
        ys[a_idx, m] = sources[predictor_label_name][()][model_source_indices]

        gp_predictions[a_idx, m, 0] = results[f"models/{model_name}/gp_predictions/theta"][()]
        gp_predictions[a_idx, m, 1] = results[f"models/{model_name}/gp_predictions/mu_single"][()]
        gp_predictions[a_idx, m, 2] = results[f"models/{model_name}/gp_predictions/sigma_single"][()]
        gp_predictions[a_idx, m, 3] = results[f"models/{model_name}/gp_predictions/sigma_multiple"][()]

        b = model_config["bounds"]
        b.update(theta=[0, 1])
        b["theta"] = np.array(b["theta"]) + [+SMALL, -SMALL]
        bounds.append(b)


    def clipped_predictions(theta, mu_single, sigma_single, sigma_multiple, bounds):
        theta = np.clip(theta, *bounds["theta"])
        mu_single = np.clip(mu_single, *bounds["mu_single"])
        sigma_single = np.clip(sigma_single, *bounds["sigma_single"])
        sigma_multiple = np.clip(sigma_single, *bounds["sigma_multiple"])

        return np.array([theta, mu_single, sigma_single, sigma_multiple])


    def calc_p_single(y, theta, mu_single, sigma_single, sigma_multiple, mu_multiple_scalar):

        with warnings.catch_warnings(): 
            # I'll log whatever number I want python you can't tell me what to do
            warnings.simplefilter("ignore") 

            mu_multiple = np.log(mu_single + mu_multiple_scalar * sigma_single) + sigma_multiple**2
            
            ln_s = np.log(theta) + utils.normal_lpdf(y, mu_single, sigma_single)
            ln_m = np.log(1-theta) + utils.lognormal_lpdf(y, mu_multiple, sigma_multiple)

            # FIX BAD SUPPORT.

            # This is a BAD MAGIC HACK where we are just going to flip things.
            """
            limit = mu_single - 2 * sigma_single
            bad_support = (y <= limit) * (ln_m > ln_s)
            ln_s_bs = np.copy(ln_s[bad_support])
            ln_m_bs = np.copy(ln_m[bad_support])
            ln_s[bad_support] = ln_m_bs
            ln_m[bad_support] = ln_s_bs
            """
            ln_s = np.atleast_1d(ln_s)
            ln_m = np.atleast_1d(ln_m)

            lp = np.array([ln_s, ln_m]).T

            #assert np.all(np.isfinite(lp))

            p_single = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

        return (p_single, ln_s, ln_m)


    def calc_probs(ys, gp_predictions, bounds, N_draws, percentiles):

        ys = np.atleast_2d(ys)
        Q, M = ys.shape

        shape = (Q, M + 1, N_draws)
        obj_p_singles = np.empty(shape)
        obj_ln_s = np.empty(shape)
        obj_ln_m = np.empty(shape)

        # gp_predictions has shape M, 4, 2
        mu = gp_predictions[:, :, 0]
        var = gp_predictions[:, :, 1]

        for q, ym in enumerate(tqdm(ys)):

            for m, y in enumerate(ym):

                # Do draws for all models
                _slice = (q, m, slice(0, None))
                if not np.all(np.isfinite(np.hstack([y, mu[m], var[m]]))):
                    obj_p_singles[_slice] = np.nan
                    obj_ln_s[_slice] = np.nan
                    obj_ln_s[_slice] = np.nan
                    continue

                draws = clipped_predictions(
                    *np.random.multivariate_normal(mu[m], np.diag(var[m]), size=N_draws).T, 
                    bounds[m])

                obj_p_singles[_slice], obj_ln_s[_slice], obj_ln_m[_slice] \
                    = calc_p_single(y, *draws, mu_multiple_scalar=3)

        # Now calculate the joint probabilities
        obj_ln_s[:, -1, :] = np.nansum(obj_ln_s[:, :-1, :], axis=1)
        obj_ln_m[:, -1, :] = np.nansum(obj_ln_m[:, :-1, :], axis=1)

        lp = np.array([obj_ln_s[:, -1, :], obj_ln_m[:, -1, :]])

        obj_p_singles[:, -1, :] = np.exp(lp[0] - special.logsumexp(lp, axis=0))

        # Then calculate percentiles ofthose probabilities.
        obj_p_singles[~np.isfinite(obj_p_singles)] = -1
        p = np.percentile(obj_p_singles, percentiles, axis=-1).transpose((1, 0, 2))
        p[p == -1] = np.nan
        return p


    # Given RUWE, calculate for all.
    ast_gp_predictions = gp_predictions[:, [0], :, :]

    p_50 = np.empty((ruwe.size, 1))

    for i, ruwe_ in enumerate(tqdm(ruwe.flatten())):

        foo = calc_probs(ruwe_, ast_gp_predictions[i], bounds, 128, [50])
        p_50[i] = foo[0, 0, 0]

        
    raise a
    '''

    N_bins = 20


    # At each bin point, calculate the GP parameters for w, mu_s, sigma_s, etc..
    # At each bin point, take some RUWE values and calculate p_single draws


