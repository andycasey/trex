
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial, special

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from matplotlib.colors import LogNorm 
# First: Plot GP predictions across HRD.
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import (constants, units as u)


import mpl_utils

plt.style.use(mpl_utils.mpl_style)


_common_continuous_cmap = "coolwarm"

_common_latex_labels = {
    "bp-rp": r"$G_\textrm{BP} - G_\textrm{RP}$",
    "absolute_g_mag": r"$M_\textrm{G}$",
}


def density_rv_excess_vs_absolute_magnitude(K_est, absolute_mag, **kwargs):

    fig, ax_background = plt.subplots()

    x, y = (K_est, absolute_mag)


    xp_min, xp_max = (0, np.log10(2000))
    MAGIC = 0.97

    num_bins = 100

    bins = (np.logspace(xp_min, xp_max, num_bins + 1),
            np.linspace(-8, 8, num_bins + 1))

    ax_background.set_xlabel(r"$K$ $/$ $\textrm{km\,s}^{-1}$")
    ax_background.set_ylabel(_common_latex_labels["absolute_g_mag"])

    ax_background.set_ylim([-8, 8])
    ax_background.set_xlim(MAGIC, 10**xp_max - 1) # you wouldn't believe why I have to do this even if I told you.

    fig.tight_layout()

    ax = fig.add_axes(ax_background.get_position(), frameon=False)


    _, im = mpl_utils.plot_binned_statistic(x, y, y,
                                            function="count", ax=ax_background, 
                                            full_output=True, interpolation="none",
                                            bins=bins,
                                            norm=LogNorm(),
                                            cmap="Greys",
                                            )
    #ax_background.set_ylim(ax_background.get_ylim()[::-1])

    # Here comes the smoke and mirrors.
    ax.set_xlim(bins[0][0], bins[0][-1])
    ax.set_ylim(bins[1][0], bins[1][-1])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xscale("log")

    ax_background.set_xlabel(None)
    ax_background.set_ylabel(None)
    ax_background.set_xticklabels([])
    ax_background.set_yticklabels([])
    ax_background.xaxis.set_tick_params(width=0)
    ax_background.yaxis.set_tick_params(width=0)


    # I have my reservations about this approximation.
    # I think that Badenes took \log{g} to be exponential-based instead of base-10.
    base = 10

    a, b = (5.5/16, 2.25)
    approx_logg = lambda G: a * np.array(G) + b
    approx_MG = lambda logg: (1/a) * (logg - b)

    print(approx_logg(ax_background.get_ylim()))

    lines = [
        (0.5, [-10, 10], dict(c="tab:green"), None, r"$0.5$ $M_\odot$ $\textrm{RLOF}$"),
        (1.0, [approx_MG(0), 10], dict(c="tab:red"), r"$\,\,\,\textrm{TRGB}$", r"$1.0$ $M_\odot$ $\textrm{RLOF}$"),
        (2.0, [approx_MG(1.25), 10], dict(c="tab:blue"), r"$\,\,\,\textrm{TRGB}$", r"$2.0$ $M_\odot$ $\textrm{RLOF}$"),
    ]

    ylim = ax.get_ylim()


    for mass, v, plot_kwds, text_label, label in lines:    
        RV_max = 0.87 * (constants.G * mass * u.solMass * (base**(approx_logg(v)) * u.m/u.s**2))**0.25
        K_max = 0.5 * RV_max.to(u.km/u.s)

        ax.plot(K_max, v, "-", label=label, **plot_kwds)

        if v[0] > ylim[0]:
            ax.scatter([K_max[0].value], [v[0]], s=50, edgecolor="k", lw=1, zorder=100, **plot_kwds)
            ax.text(K_max[0].value, v[0], text_label, 
                    verticalalignment="center")

    ax.set_xlabel(r"$K$ $/$ $\textrm{km\,s}^{-1}$")
    ax.set_ylabel(_common_latex_labels["absolute_g_mag"])

    ax.set_ylim(ylim)
    ax.set_xlim(MAGIC, 10**xp_max - 1) # you wouldn't believe why I have to do this even if I told you.
    plt.show()

    ax.legend(loc="lower right", frameon=False)

    return fig


def scatter_period_and_rv_semiamplitude_for_known_binaries(P, K, ratio, P_err=None, K_err=None,
                                                           xlim=None, ylim=None,
                                                           **kwargs):

    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6.15, 5.2)))

    
    if not isinstance(P, dict):
        P = dict(default=P)
        K = dict(default=K)
        ratio = dict(default=ratio)
        P_err = dict(default=P_err)
        K_err = dict(default=K_err)

    collection_kwds = kwargs.pop("collection_kwds", dict())
    common_scatter_kwds = kwargs.pop("scatter_kwds", dict())

    for k in P.keys():


        mask = np.isfinite(ratio[k])

        
        scatter_kwds = dict(s=15, c=ratio[k][mask], vmin=0, vmax=1)
        scatter_kwds.update(common_scatter_kwds)
        scatter_kwds.update(collection_kwds.get(k, dict()))

        x, y = (P[k][mask], K[k][mask])

        scat = ax.scatter(x, y, **scatter_kwds)

        if P_err[k] is not None or K_err[k] is not None:
            errorbar_kwds = dict(fmt="none", lw=1, c="#CCCCCC", zorder=-1)
            errorbar_kwds.update(kwargs.pop("errorbar_kwds", dict()))
            x_err, y_err = (P_err[k][mask], K_err[k][mask])

            ax.errorbar(x, y, xerr=x_err, yerr=y_err, **errorbar_kwds)



    logs = (True, True)#, True)
    xa = np.ptp(ax.get_xlim())
    ya = np.ptp(ax.get_ylim())
    if all(logs):
        ax.loglog()
        xa, ya = (np.log10(xa), np.log10(ya))
    elif logs[0] and not logs[1]:
        ax.semilogx()
        xa = np.log10(xa)
    elif logs[1] and not logs[0]:
        ax.semilogy()
        ya = np.log10(ya)
    else:
        None

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(r"$P\,/\,\textrm{days}^{-1}$")
    ax.set_ylabel(r"$K\,/\,\mathrm{km\,s}^{-1}$")

    #ax.set_aspect(xa/ya)

    #fig.tight_layout()
    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scat, cax=cax)
    cbar.set_label(r"$p(\mathrm{single}|j_\textrm{rv})$")

    # Show observing period of Gaia Dr2.
    observing_span = 668 # days : https://www.cosmos.esa.int/web/gaia/dr2
    ylim = ax.get_ylim()
    ax.plot([observing_span, observing_span],
            ylim, "-",
            c="#666666", zorder=-1, linestyle=":", linewidth=1)
    ax.plot([2 * observing_span, 2*observing_span],
            ylim, "-",
            c="#666666", zorder=-1, linestyle="-.", linewidth=1)
    
    ax.set_ylim(ylim)

    fig.tight_layout()

    return fig


def scatter_excess_rv_jitter_for_known_binaries(K_catalog, K_est, K_catalog_err=None, K_est_err=None,
                                                log=False, xlabel=None, ylabel=None, **kwargs):

    height_ratio = 5

    fig = kwargs.pop("figure", None) 
    if fig is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5, 5)))
        divider = make_axes_locatable(ax)
        ax_diff = divider.append_axes("top", size=1, pad=0.0)
        #fig = plt.figure(figsize=kwargs.pop("figsize", (5, 5.5)))
        #gs = gridspec.GridSpec(2, 1, height_ratios=[1, height_ratio])
        #ax, ax_diff = (plt.subplot(gs[1]), plt.subplot(gs[0]))

    else:
        ax, ax_diff = fig.axes[:2]

    x, y = (K_catalog, K_est)
    x_err, y_err = (K_catalog_err, K_est_err)

    scatter_kwds = dict(s=10)
    scatter_kwds.update(kwargs.pop("scatter_kwds", dict()))

    if "c" in scatter_kwds:
        order = np.argsort(scatter_kwds["c"])[::-1]
        scatter_kwds["c"] = scatter_kwds["c"][order]
        x, y = (x[order], y[order])
        x_err, y_err = (x_err[order], y_err[order])

    scat = ax.scatter(x, y, **scatter_kwds)

    if x_err is not None or y_err is not None:
        errorbar_kwds = dict(fmt="none", lw=1, c="#CCCCCC", zorder=-1)
        errorbar_kwds.update(kwargs.pop("errorbar_kwds", dict()))
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, **errorbar_kwds)


    ax_diff.scatter(x, y-x, **scatter_kwds)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    lims = np.array([ax.get_xlim(), ax.get_ylim()])
    if log:
        lims = (np.max([0.5, np.min(lims)]), np.max(lims))
    else:
        lims = (-3, np.max(lims))

    ax.plot(lims, lims, "-", c="#666666", linestyle=":", zorder=-1, linewidth=0.5)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax_diff.set_xlim(lims)

    lim = np.max(np.abs(ax_diff.get_ylim()))
    ax_diff.set_ylim(-lim, +lim)

    ax_diff.plot(lims, [0, 0], "-", c="#666666", linestyle=":", zorder=-1, linewidth=0.5)

    ax_diff.yaxis.set_major_locator(MaxNLocator(3))
    ax_diff.set_xticks([])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    if log:
        ax.loglog()

    
    aspect = lambda ax: np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim())
    ax.set_aspect(aspect(ax))
    ax_diff.set_aspect(aspect(ax_diff) / height_ratio)


    if kwargs.get("colorbar", False):
        
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(scat, cax=cax)
        cbar.set_label(kwargs.get("colorbar_label", None))

    for i in range(5): # crazy, i know, yet here we are
        fig.tight_layout()
    
    return fig






def binned_gp_expectation_values(bp_rp, apparent_mag, absolute_mag, 
                                 expectation_values, num_bins=100,
                                 function="mean", band="g", colorbar_labels=None, **kwargs):

    if not isinstance(expectation_values, tuple):
        raise TypeError("expectation_values must be a tuple")

    K = len(expectation_values) # the number of GP expectation values to show
    L = 2 # (apparent mag, absolute mag)

    latex_labels = kwargs.pop("latex_labels", dict())
    if latex_labels is None:
        latex_labels = dict()
    get_label = lambda _, __=None: latex_labels.get(_, __ or _)

    figsize = kwargs.pop("figsize", (7.4, 7.6))
    if figsize is None:
        figsize = (L * 3, K * 3)

    fig = plt.figure(figsize=figsize)

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(2, 2),
                    axes_pad=0.75,
                    label_mode="all",
                    share_all=False,
                    cbar_location="top",
                    cbar_mode="edge",
                    cbar_size="5%",
                    cbar_pad=0.05)


    for i in range(2):

        ax_abs, ax_app = axes_row = (grid[i], grid[i + 2])

        for ax in axes_row:
            ax.set_facecolor("#eeeeee")

        _, im = mpl_utils.plot_binned_statistic(bp_rp, 
                                                apparent_mag, 
                                                expectation_values[i],
                                                function=function, ax=ax_app, 
                                                full_output=True, **kwargs)

        _, im = mpl_utils.plot_binned_statistic(bp_rp, 
                                                absolute_mag, 
                                                expectation_values[i],
                                                function=function, ax=ax_abs, 
                                                full_output=True, **kwargs)

        for ax in axes_row:

            ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(6))

            ax.set_xlabel(get_label("bp_rp", r"$G_\textrm{BP} - G_\textrm{RP}$"))

        ax_abs.set_ylabel(get_label(f"absolute_{band}_mag", f"$M_{0}$".format(band)))
        ax_app.set_ylabel(get_label(f"apparent_{band}_mag", f"${0}$".format(band)))

        caxes = (grid.cbar_axes[0], grid.cbar_axes[1])
        cax = caxes[i]
        cax.colorbar(im)
        cax.xaxis.set_major_locator(MaxNLocator(5))

        cax.set_xlabel(get_label(colorbar_labels[i], colorbar_labels[i]))

        # For some fucking reason I can't get ticks to appear, and the padding between the colorbar
        # and the ticks and the label is all fucked up.


        for cax in grid.cbar_axes:
            if cax not in caxes:
                cax.set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.925)

    return fig


def log_density_gp_expectation_value(x, y, **kwargs):

    fig, ax = plt.subplots()

    _, im = mpl_utils.plot_binned_statistic(x, y, y,
                                            function="count", ax=ax, 
                                            full_output=True, **kwargs)

    return fig



def binned_posterior_probability(bp_rp, apparent_mag, absolute_mag, ratios, **kwargs):

    fig = kwargs.pop("fig", None)
    if fig is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 8)))
    else:
        ax = kwargs.pop("ax", fig.axes[0])
        #ax = fig.axes[0]

    latex_labels = kwargs.pop("latex_labels", dict())
    band = kwargs.pop("band", None)

    kwds = dict(function="mean", ax=ax, full_output=True)
    kwds.update(kwargs)
    _, im = mpl_utils.plot_binned_statistic(bp_rp, absolute_mag, ratios, **kwds)

    ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))

    ax.set_xlabel(_common_latex_labels["bp-rp"])
    # TODO: actually use the kwd["band"] entry.
    ax.set_ylabel(_common_latex_labels["absolute_g_mag"])
    ax.set_facecolor("#eeeeee")   
    fig.tight_layout()

    return (fig, im)





def hist_literature_single_stars_and_binaries(sb9, soubiran, **kwargs):

    N = 3
    fig, axes = plt.subplots(1, 3, figsize=kwargs.pop("figsize", (11, 3.5)))

    latex_labels = (r"$\textrm{radial velocity jitter}$ $j_\mathrm{rv}$ $/$ \textrm{km\,s}$^{-1}$", 
                    r"$\textrm{astrometric jitter}$ $j_\mathrm{ast}$", 
                    r"$\textrm{photometric jitter}$ $j_\mathrm{phot}$")

    kwds = dict(histtype="barstacked")
    labels = (r"$\textrm{literature binaries}$", #$\textrm{(Pourbaix et al. 2004)}$", 
             r"$\textrm{literature single stars}$",)#" $\textrm{(Soubiran et al. 2013)}$"))

    for i, ax in enumerate(axes):
        ok = np.isfinite(sb9.T[i] * soubiran.T[i])
        #ax.semilogx
        x1, x2 = x = [soubiran.T[i][ok], sb9.T[i][ok]][::-1]
        x = np.hstack(x)

        """
        bins = np.logspace(int(np.log10(x.min())) - 0.5, 0.5 + int(np.log10(x.max())), 20)
        print(bins)

        ax.hist([x1, x2], bins=bins, histtype="barstacked")
        """

        if ax.is_last_col():
            kwds["label"] = labels

        bins = np.logspace(int(np.log10(x.min())) - 0.5, 0.5 + int(np.log10(x.max())), 20)
        kwds["bins"] = np.log10(bins)
        kwds.update(kwargs)
        hist = ax.hist([np.log10(x1), np.log10(x2)], **kwds)

        ax.yaxis.set_major_locator(MaxNLocator(4))

        ticks = np.unique(np.log10(bins).astype(int))
        print(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([r"$10^{{{0}}}$".format(t) for t in ticks])
        
        minor_ticks = []
        for k in range(ticks[0] - 1, ticks[-1] + 1):
            for j in range(2, 10):
                minor_ticks.append(k + np.log10(j))

        xlim = ax.get_xlim()
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xlim(xlim)

        ax.set_xlabel(latex_labels[i])

        ax.set_aspect((np.ptp(ax.get_xlim()))/np.ptp(ax.get_ylim()))

        try:
            kwds.pop("label")
        except KeyError:
            None
    
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return fig


if __name__ == "__main__":




    def cross_match(A_source_ids, B_source_ids):

        A = np.array(A_source_ids, dtype=np.long)
        B = np.array(B_source_ids, dtype=np.long)

        ai = np.where(np.in1d(A, B))[0]
        bi = np.where(np.in1d(B, A))[0]
        
        a_idx, b_idx = (ai[np.argsort(A[ai])], bi[np.argsort(B[bi])])

        # Sanity checks
        assert a_idx.size == b_idx.size
        assert np.all(A[a_idx] == B[b_idx])
        return (a_idx, b_idx)




    def _get_binned_gp_expectation_values(sources, results, model_name, band="g", parameter_names=None, **kwargs):

        if parameter_names is None:
            parameter_names = ("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple")

        band = f"{band.lower()}"
        bands = ("bp", "rp", "g")
        if band not in bands:
            raise ValueError(f"band must be one of {bands}")


        # Get expectation values.
        expectation_values = []
        for p in parameter_names:
            d = results[f"models/{model_name}/gp_predictions/{p}"][()][:, 0]
            expectation_values.append(np.abs(d) if p.startswith("sigma_") else d)

        expectation_values = tuple(expectation_values)

        # Now get the data.
        source_indices = results[f"models/{model_name}/gp_predictions/source_indices"][()]

        bp_rp = sources["bp_rp"][()][source_indices]

        # TODO: need the right absolute/apparent mags to use! FUCK!
        apparent_mag = sources["phot_g_mean_mag"][()][source_indices]
        absolute_mag = sources["absolute_g_mag"][()][source_indices]

        mask = None

        kwds = dict(bp_rp=bp_rp,
                    band=band,
                    apparent_mag=apparent_mag,
                    absolute_mag=absolute_mag,
                    expectation_values=expectation_values,
                    colorbar_labels=parameter_names,
                    min_entries_per_bin=5,
                    mask=mask)
        kwds.update(kwargs)

        return kwds



    def _get_binned_posterior_probability_data(sources, results, model_name=None, band="g", **kwargs):


        if model_name is None:
            model_name = "joint"

        #ratios = results[f"results/p_{model_name}_single"][()]
        group = results["results"]
        model_index = list(group.attrs["model_names"]).index(model_name)
        percentile_index = group.attrs["percentiles"].searchsorted(50)
        v = group.attrs["percentiles"][percentile_index]
        if v != 50:
            print(f"Warning: percentile taken for p_rv_single expectation value is not 50 (50 != {v})")

        ratios = group["p_single_percentiles"][()][:, model_index, percentile_index]

        source_indices = results["results/source_indices"]

        bp_rp = sources["bp_rp"][()][source_indices]

        # TODO: need the right absolute/apparent mags to use! FUCK!
        apparent_mag = sources[f"phot_{band}_mean_mag"][()][source_indices]
        absolute_mag = sources[f"absolute_{band}_mag"][()][source_indices]

        mask = None
        latex_labels = dict()

        kwds = dict(bp_rp=bp_rp,
                    band=band,
                    apparent_mag=apparent_mag,
                    absolute_mag=absolute_mag,
                    ratios=ratios,
                    min_entries_per_bin=1,
                    mask=mask,
                    latex_labels=latex_labels)
        kwds.update(kwargs)

        return kwds


    def _get_rv_excess(sources, results, model_name="rv", **kwargs):


        group = results["results"]
        source_indices = group["source_indices"][()]

        #source_ids = results[f"{group_name}/source_id"][()]
        source_id = sources["source_id"][()][source_indices]
        absolute_g_mag = sources["absolute_g_mag"][()][source_indices]

        rv_mu_single = group[f"{model_name}_mu_single"][()]
        rv_sigma_single = group[f"{model_name}_sigma_single"][()]
        
        K_est = group["K"][()]
        K_est_err = group["K_err"][()]

        #p_rv_single = group["p_rv_single"][()]
        model_index = list(group.attrs["model_names"]).index(model_name)
        percentile_index = group.attrs["percentiles"].searchsorted(50)
        v = group.attrs["percentiles"][percentile_index]
        if v != 50:
            print(f"Warning: percentile taken for p_rv_single expectation value is not 50 (50 != {v})")

        p_rv_single = group["p_single_percentiles"][()][:, model_index, percentile_index]

        return dict(K_est=K_est,
                    K_est_err=K_est_err,
                    source_id=source_id,
                    absolute_g_mag=absolute_g_mag,
                    rv_mu_single=rv_mu_single,
                    rv_sigma_single=rv_sigma_single,
                    p_rv_single=p_rv_single)



        
    def _get_rv_excess_for_apw(sources, results, apw_catalog, use_apw_mask=True,**kwargs):


        # Do the cross-match.
        rv_data = _get_rv_excess(sources, results)

        if use_apw_mask:
            mask = apw_catalog["converged"]
            apw_catalog = Table(apw_catalog[mask])


        catalog_source_ids = apw_catalog["source_id"]

        idx, catalog_idx = cross_match(rv_data["source_id"], catalog_source_ids)

        apw_K = apw_catalog["K"][catalog_idx]
        e_apw_K = apw_catalog["K_err"][catalog_idx]

        # Get our subset.
        source_id = rv_data["source_id"][idx]
        rv_mu_single = rv_data["rv_mu_single"][idx]
        rv_sigma_single = rv_data["rv_sigma_single"][idx]

        K_est = rv_data["K_est"][idx]
        K_est_err = rv_data["K_est_err"][idx]
        #p_rv_single = rv_data["p_rv_single"][idx]

        group = results["results"]
        model_index = list(group.attrs["model_names"]).index("rv")
        percentile_index = group.attrs["percentiles"].searchsorted(50)
        v = group.attrs["percentiles"][percentile_index]
        if v != 50:
            print(f"Warning: percentile taken for p_rv_single expectation value is not 50 (50 != {v})")

        p_rv_single = group["p_single_percentiles"][()][idx, model_index, percentile_index]





        # Since all of these are binaries, let's just show the excess as is.
        """
        K_est = rv_jitter - model_mu[:, 0]
        e_K_est = np.sqrt(model_sigma[:, 0]**2 + model_mu[:, 1] + model_sigma[:, 1])
        """

        """
        K_est = np.sqrt(2) * rv_jitter - model_mu[:, 0]
        
        # Add formal errors.
        rv_nb_transits = sources["rv_nb_transits"][()][indices][idx]
        N = rv_nb_transits
        e_K_est = rv_jitter / np.sqrt(2) * np.sqrt(1 - (2/(N-1)) * (special.gamma(N/2)/special.gamma((N-1)/2))**2)
        """


        kwds = dict(apw_K=apw_K,
                    e_apw_K=e_apw_K,
                    K_est=K_est,
                    e_K_est=K_est_err,
                    apw_P=apw_catalog["P"][catalog_idx],
                    e_apw_P=apw_catalog["P_err"][catalog_idx],
                    apw_e=apw_catalog["e"][catalog_idx],
                    p_rv_single=p_rv_single)


        kwds.update(kwargs)

        return kwds



    def _get_rv_excess_for_sb9(sources, results, sb9_catalog, use_sb9_mask=True, **kwargs):


        # Do the cross-match.

        rv_data = _get_rv_excess(sources, results)


        # Only include good results from SB9
        if use_sb9_mask:
            sb9_mask = (sb9_catalog["f_K1"] != ">") \
                     * (sb9_catalog["f_K1"] != "a") \
                     * (sb9_catalog["f_K1"] != ">a") \
                     * (sb9_catalog["Grade"] > 0) \
                     * (sb9_catalog["e_Per"] > 0)
            sb9_catalog = Table(sb9_catalog[sb9_mask])

        sb9_source_ids = sb9_catalog["source_id"]       

        idx, sb9_idx = cross_match(rv_data["source_id"], sb9_source_ids)

        sb9_K1 = sb9_catalog["K1"][sb9_idx]
        e_sb9_K1 = sb9_catalog["e_K1"][sb9_idx]
        rv_nb_transits = sb9_catalog["rv_nb_transits"][sb9_idx]

        #ok = (rv_nb_transits > 10) * (sb9_catalog["e"][sb9_idx] < 0.10) * (sb9_catalog["Per"][sb9_idx] < 668)
        #idx, sb9_idx = (idx[ok], sb9_idx[ok])
        #sb9_K1 = sb9_catalog["K1"][sb9_idx]
        #e_sb9_K1 = sb9_catalog["e_K1"][sb9_idx]
        #rv_nb_transits = sb9_catalog["rv_nb_transits"][sb9_idx]


        # Get our subset.
        source_id = rv_data["source_id"][idx]
        rv_mu_single = rv_data["rv_mu_single"][idx]
        rv_sigma_single = rv_data["rv_sigma_single"][idx]

        K_est = rv_data["K_est"][idx]
        K_est_err = rv_data["K_est_err"][idx]
        #p_rv_single = rv_data["p_rv_single"][idx]

        group = results["results"]
        model_index = list(group.attrs["model_names"]).index("rv")
        percentile_index = group.attrs["percentiles"].searchsorted(50)
        v = group.attrs["percentiles"][percentile_index]
        if v != 50:
            print(f"Warning: percentile taken for p_rv_single expectation value is not 50 (50 != {v})")

        p_rv_single = group["p_single_percentiles"][()][idx, model_index, percentile_index]


        """
        # rv_jitter is the standard deviation among measurements.
        # (Unbiased estimator (see p6 of Katz et al)

        # K ~= sqrt(2) * std_dev(v)

        #K_est = (rv_jitter - model_mu[:, 0]) #/ np.sqrt(0.5 * np.pi * rv_nb_transits)
        
        K_est = np.sqrt(2) * rv_jitter - model_mu[:, 0] # 1.97 / 1.50
        
        # Add formal errors.
        N = rv_nb_transits
        e_K_est = rv_jitter * np.sqrt(1 - (2/(N-1)) * (special.gamma(N/2)/special.gamma((N-1)/2))**2)

        #K_est = np.sqrt(np.pi/2) * rv_jitter#/np.sqrt(rv_nb_transits)
        #K_est = rv_jitter - model_mu[:, 0]
        #e_K_est = np.sqrt(model_sigma[:, 0]**2 + model_mu[:, 1] + model_sigma[:, 1])

        #diff = diff[ok]
        """
        diff = sb9_K1 - K_est
        
        print(f"Diff mean: {np.nanmean(diff):.2f} median: {np.nanmedian(diff):.2f}")


        kwds = dict(sb9_K1=sb9_K1,
                    e_sb9_K1=e_sb9_K1,
                    K_est=K_est,
                    e_K_est=K_est_err,
                    sb9_P=sb9_catalog["Per"][sb9_idx],
                    e_sb9_P=sb9_catalog["e_Per"][sb9_idx],
                    rv_nb_transits=rv_nb_transits,
                    sb9_catalog=sb9_catalog[sb9_idx],
                    sb9_e=sb9_catalog["e"][sb9_idx],
                    p_rv_single=p_rv_single)

        kwds.update(kwargs)

        return kwds




    import os
    import sys
    import h5py as h5
    import yaml
    from hashlib import md5
    from tqdm import tqdm

    from astropy.table import Table


    results_dir = sys.argv[1]
    results_path = os.path.join(results_dir, "results.h5")

    # TODO: Store data in the results dir so we don't have to do this all the time.
    # TODO: And somehow store the PWD?
    pwd = "../../"
    data_path = os.path.join(pwd, "data/5482.hdf5")

    # LOad up.
    data = h5.File(data_path, "r")
    results = h5.File(results_path, "r")
    
    sources = data["sources"]

    def savefig(fig, basename):
        path = os.path.join(results_dir, f"figures/{basename}")
        fig.savefig(f"{path}.pdf", dpi=300)
        fig.savefig(f"{path}.png", dpi=150)
        print(f"Saved figure to {path}")


    # Before we even show any results:
    # Make comparison of Soubiran and SB9
    catalog_path = os.path.join(pwd, "data/catalogs")
    soubiran_catalog = Table.read(os.path.join(catalog_path, "soubiran-2013-xm-gaia.fits"))
    sb9_catalog = Table.read(os.path.join(catalog_path, "sb9-xm-gaia.fits"))

    # Soubiran has fewer cros-smatches than SB9.
    # Calculate distance metrics.
    def _xm_literature_single_stars_and_binaries(sb9_catalog, soubiran_catalog, parameter_names=None):
        if parameter_names is None:
            parameter_names = ("bp_rp", "phot_g_mean_mag", "absolute_g_mag")

        for catalog in (sb9_catalog, soubiran_catalog):
            if "absolute_g_mag" not in catalog.dtype.names:
                catalog["absolute_g_mag"] = catalog["phot_g_mean_mag"] + 5 * np.log10(catalog["parallax"]/100.0)

        A = np.atleast_2d([sb9_catalog[pn] for pn in parameter_names]).T
        B = np.atleast_2d([soubiran_catalog[pn] for pn in parameter_names]).T

        # Only finites.
        A = A[np.all(np.isfinite(A), axis=1)]
        B = B[np.all(np.isfinite(B), axis=1)]

        D = spatial.distance_matrix(A, B)

        A_idx = np.zeros(B.shape[0], dtype=int) - 1

        D_flat = D.flatten()
        for k, index in enumerate(tqdm(np.argsort(D_flat))):

            i, j = np.unravel_index(index, D.shape)

            if A_idx[j] > 0 or i in A_idx:
                # That source from A is already assigned.
                continue

            # Assign pairs.
            A_idx[j] = i
        
        dist = np.sum((A[A_idx] - B)**2, axis=1)
        keep = dist < 0.1

        distinguishing_parameters = ("rv_jitter", "ast_jitter", "phot_g_variability")

        for catalog in (sb9_catalog, soubiran_catalog):
            for parameter in distinguishing_parameters:
                if parameter in catalog.dtype.names: continue

                if parameter == "rv_jitter":
                    catalog[parameter] = catalog["radial_velocity_error"] * np.sqrt(catalog["rv_nb_transits"] * np.pi / 2)

                elif parameter == "ast_jitter":
                    catalog[parameter] = np.sqrt(catalog["astrometric_chi2_al"]/(catalog["astrometric_n_good_obs_al"] - 5))

                elif parameter == "phot_g_variability":
                    catalog[parameter] = np.sqrt(catalog["astrometric_n_good_obs_al"]) * (catalog["phot_g_mean_flux_error"]/catalog["phot_g_mean_flux"])



        X1 = sb9_catalog["rv_jitter","ast_jitter","phot_g_variability"][A_idx]
        X1 = X1.as_array().view(np.float).data.reshape((-1, 3))[keep]

        X2 = soubiran_catalog["rv_jitter","ast_jitter","phot_g_variability"]
        X2 = X2.as_array().view(np.float).data.reshape((-1, 3))[keep]

        return dict(sb9=X1, soubiran=X2)


    # Plot the bayes factor density as a function of parallax or distance.
    def _get_bayes_factors(sources, results, xlabel, model_name=None, mask_function=None, *kwargs):

        dataset_name = "bf_multiple" if model_name is None else f"bf_{model_name}_multiple"
        log_bf = np.log10(results[f"results/{dataset_name}"][()])


        source_indices = results["results/source_indices"][()]
        x = sources[xlabel][()][source_indices]
        
        if mask_function is not None:
            mask = mask_function(sources, source_indices, log_bf)
        else:
            mask = None

        kwds = dict(log_bf=log_bf, x=x, mask=mask, source_indices=source_indices)
        return kwds







    def binned_bayes_factor(x, log_bf, **kwargs):

        fig = kwargs.pop("fig", None)
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (8, 8)))
        else:
            ax = fig.axes[0]

        latex_labels = kwargs.pop("latex_labels", dict())
        

        kwds = dict(function="mean", ax=ax, full_output=True)
        kwds.update(kwargs)
        _, im = mpl_utils.plot_binned_statistic(x, log_bf, log_bf, **kwds)

        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(r"{x}")
        
        ax.set_ylabel(r"{BF}")
        ax.set_facecolor("#eeeeee")   
        fig.tight_layout()

        return fig

    

    # Plot log density of sources and their excess RV jitter.
    kwds = _get_rv_excess(sources, results)

    fig = density_rv_excess_vs_absolute_magnitude(K_est=kwds["K_est"],
                                                  absolute_mag=kwds["absolute_g_mag"])
    savefig(fig, "K-vs-MG-with-estimated-Kmax-limits")


    '''
    def mask_function(s, i, log_bf):
        return ((s["parallax"][()][i]/s["parallax_error"][()][i]) >= 5) \
             * (np.isfinite(s["parallax"][()][i])) 


    max_log_bf = 10
    kwds = _get_bayes_factors(sources, results, xlabel="parallax", model_name="ast",
                              mask_function=mask_function)

    kwds.update(function="count", bins=200, interpolation=None, norm=LogNorm(),
                x=1.0/kwds["x"])
    source_indices = kwds.pop("source_indices", None)
    kwds.update(bins=(np.linspace(0, 3, 200), np.linspace(0, max_log_bf, 200)))


    fig = binned_bayes_factor(**kwds)
    ax = fig.axes[0]
    ax.set_ylim(ax.get_ylim()[::-1])
    #fig.axes[0].set_ylim(0, 310)

    ax.axhline(8, c="#000000", zorder=10, linestyle=":", linewidth=2)



    def rv_mask_function(s, i, log_bf):
        return (s["rv_nb_transits"][()][i] > 5)
    kwds = _get_bayes_factors(sources, results, xlabel="parallax", model_name="rv",
                              mask_function=rv_mask_function)

    kwds.update(function="count", bins=200, interpolation=None, norm=LogNorm(),
                x=1.0/kwds["x"])
    kwds.update(bins=(np.linspace(0, 3, 200), np.linspace(0, max_log_bf, 200)))
    source_indices = kwds.pop("source_indices", None)

    fig = binned_bayes_factor(**kwds)
    ax = fig.axes[0]
    ax.set_ylim(ax.get_ylim()[::-1])
    #fig.axes[0].set_ylim(0, 310)

    ax.axhline(8, c="#000000", zorder=10, linestyle=":", linewidth=2)




    kwds = _get_bayes_factors(sources, results, xlabel="parallax", model_name=None,
                              mask_function=lambda *args: rv_mask_function(*args) * mask_function(*args))

    kwds.update(function="count", bins=200, interpolation=None, norm=LogNorm(),
                x=1.0/kwds["x"])
    kwds.update(bins=(np.linspace(0, 3, 200), np.linspace(0, max_log_bf, 200)))
    source_indices = kwds.pop("source_indices", None)

    fig = binned_bayes_factor(**kwds)
    ax = fig.axes[0]
    ax.set_ylim(ax.get_ylim()[::-1])
    #fig.axes[0].set_ylim(0, 310)

    ax.axhline(8, c="#000000", zorder=10, linestyle=":", linewidth=2)

    '''



    # Plot radial velocity semi-amplitude against our estimate for binary systems in the SB9 catalog.    
    sb9_path = os.path.join(pwd, "data/catalogs/sb9-xm-gaia.fits")
    sb9_catalog = Table.read(sb9_path)

    sb9_kwds = _get_rv_excess_for_sb9(sources, results, sb9_catalog, use_sb9_mask=True)

    apw_path = os.path.join(pwd, "data/catalogs/apw-highK-unimodal-xm-gaia.fits")
    apw_catalog = Table.read(apw_path)

    apw_kwds = _get_rv_excess_for_apw(sources, results, apw_catalog, use_apw_mask=True)






    scatter_kwds = dict(cmap=_common_continuous_cmap)


    # Now joint (SB9 + APW)
    xlim = (0.1, 216798)
    ylim = (0.48, 371)
    sb9_kwds = _get_rv_excess_for_sb9(sources, results, sb9_catalog, use_sb9_mask=False)
    apw_kwds = _get_rv_excess_for_apw(sources, results, apw_catalog, use_apw_mask=False)
    kwds = dict(P=dict(sb9=sb9_kwds["sb9_P"],
                       apw=apw_kwds["apw_P"]),
                K=dict(sb9=sb9_kwds["sb9_K1"],
                       apw=apw_kwds["apw_K"]),
                ratio=dict(sb9=sb9_kwds["p_rv_single"],
                           apw=apw_kwds["p_rv_single"]),
                P_err=dict(sb9=sb9_kwds["e_sb9_P"],
                           apw=apw_kwds["e_apw_P"]),
                K_err=dict(sb9=sb9_kwds["e_sb9_K1"],
                           apw=apw_kwds["e_apw_K"]),
                collection_kwds=dict(sb9=dict(marker="o",
                                              label=r"$\textrm{Pourbaix \emph{et al.} (2004)}$"), 
                                    apw=dict(marker="^",
                                             label=r"$\textrm{Price-Whelan \emph{et al.} (2018)}$")),
                scatter_kwds=scatter_kwds,
                xlim=xlim,
                ylim=ylim)
    fig = scatter_period_and_rv_semiamplitude_for_known_binaries(**kwds)
    savefig(fig, "scatter-period-and-rv-semiamplitude-for-known-binaries-all")

    
    # Plot radial velocity semi-amplitude against our estimate for binary systems from APW
    scatter_kwds.update(marker="o")
    fig = scatter_period_and_rv_semiamplitude_for_known_binaries(P=sb9_kwds["sb9_P"],
                                                                 K=sb9_kwds["sb9_K1"],
                                                                 ratio=sb9_kwds["p_rv_single"],
                                                                 P_err=sb9_kwds["e_sb9_P"],
                                                                 K_err=sb9_kwds["e_sb9_K1"],
                                                                 scatter_kwds=scatter_kwds,
                                                                 xlim=xlim, ylim=ylim)

    savefig(fig, "scatter-period-and-rv-semiamplitude-for-known-binaries-sb9")


    scatter_kwds.update(marker="^")
    fig = scatter_period_and_rv_semiamplitude_for_known_binaries(P=apw_kwds["apw_P"],
                                                                 K=apw_kwds["apw_K"],
                                                                 ratio=apw_kwds["p_rv_single"],
                                                                 P_err=apw_kwds["e_apw_P"],
                                                                 K_err=apw_kwds["e_apw_K"],
                                                                 scatter_kwds=scatter_kwds,
                                                                 xlim=xlim, ylim=ylim)
    savefig(fig, "scatter-period-and-rv-semiamplitude-for-known-binaries-apw")



    fig = scatter_excess_rv_jitter_for_known_binaries(K_catalog=sb9_kwds["sb9_K1"],
                                                      K_catalog_err=sb9_kwds["e_sb9_K1"],
                                                      K_est=sb9_kwds["K_est"],
                                                      K_est_err=sb9_kwds["e_K_est"],
                                                      scatter_kwds=dict(c=sb9_kwds["sb9_e"], 
                                                                        vmin=0, vmax=1, 
                                                                        cmap=_common_continuous_cmap),
                                                      colorbar=True, 
                                                      colorbar_label=r"$\textrm{eccentricity}$ $\textrm{(Pourbaix \emph{et al.} 2004)}$",
                                                      log=True,
                                                      xlabel=r"$K\,/\,\mathrm{km\,s}^{-1}$ $\textrm{(Pourbaix \emph{et al.} 2004)}$",
                                                      ylabel=r"$K\,/\,\mathrm{km\,s}^{-1}$ $\textrm{(this work)}$")
    savefig(fig, "scatter-excess-rv-jitter-for-known-binaries-sb9")



    fig = scatter_excess_rv_jitter_for_known_binaries(K_catalog=apw_kwds["apw_K"],
                                                      K_catalog_err=apw_kwds["e_apw_K"],
                                                      K_est=apw_kwds["K_est"],
                                                      K_est_err=apw_kwds["e_K_est"],
                                                      scatter_kwds=dict(c=apw_kwds["apw_e"], 
                                                                        vmin=0, vmax=1,
                                                                        cmap=_common_continuous_cmap),
                                                      colorbar=True,
                                                      colorbar_label=r"$\textrm{eccentricity}$ $\textrm{(Price-Whelan \emph{et al.} 2018)}$",
                                                      log=True,
                                                      xlabel=r"$K\,/\,\mathrm{km\,s}^{-1}$ $\textrm{(Price-Whelan \emph{et al.} 2018)}$",
                                                      ylabel=r"$K\,/\,\mathrm{km\,s}^{-1}$ $\textrm{(this work)}$")
    savefig(fig, "scatter-excess-rv-jitter-for-known-binaries-apw")











    # Plot the distributions of jitter for comparable catalogs of single stars and binaries.
    kwds = _xm_literature_single_stars_and_binaries(sb9_catalog, soubiran_catalog)
    kwds.update(color=["#000000", "#BBBBBB"])



    fig = hist_literature_single_stars_and_binaries(**kwds)
    savefig(fig, "hist-literature-single-stars-and-binaries")


    sensible_mask = lambda k: (k["absolute_mag"] < 10) \
                            * (k["absolute_mag"] > -6) \
                            * (k["bp_rp"] > 0.25) \
                            * (k["bp_rp"] < 4)

    mainsequence_mask = lambda k: sensible_mask(k) \
                                * (k["absolute_mag"] > 2) \
                                * (k["bp_rp"] < 2.4)

    common_kwds = dict(min_entries_per_bin=10,
                       bins=200,
                       interpolation="none",
                       subsample=None,
                       vmin=0, vmax=1,
                       cmap=_common_continuous_cmap,
                       )

    for function in ("mean", ):#"median"):

        # Plot RV binned
        kwds = _get_binned_posterior_probability_data(sources, results, "rv", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=sensible_mask(kwds), function=function)
        
        # TODO: put colorbar on
        fig, _ = binned_posterior_probability(**kwds)
        savefig(fig, f"binned-posterior-probability-rv-{function}")
        

        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, "ast", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=sensible_mask(kwds), function=function)

        # TODO: put colorbar on
        fig, _ = binned_posterior_probability(**kwds)
        savefig(fig, f"binned-posterior-probability-ast-{function}")


        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, model_name=None, band="g")
        kwds.update(common_kwds)
        kwds.update(mask=sensible_mask(kwds), function=function)

        # TODO: put colorbar on
        fig, _ = binned_posterior_probability(**kwds)
        savefig(fig, f"binned-posterior-probability-joint-{function}")



        def plot_binned_posterior_probability_all_models(function, sensible_mask, common_kwds):

            fig = plt.figure(figsize=(12, 4))
            grid = AxesGrid(fig, 111,
                            nrows_ncols=(1, 3),
                            axes_pad=0.75,
                            label_mode="all",
                            share_all=False, # of course, AxesGrid does not respect share_all like the documentation suggests...
                            cbar_location="right",
                            cbar_mode="edge",
                            cbar_size="5%",
                            cbar_pad=0.05)


            model_names = ("ast", "rv", None)
            titles = (r"$\textrm{astrometry}$", 
                      r"$\textrm{radial velocity}$", 
                      r"$\textrm{astrometry and radial velocity}$")
            for k, (model_name, title) in enumerate(zip(model_names, titles)):
                ax = grid[k]

                kwds = _get_binned_posterior_probability_data(sources, results, model_name=model_name, band="g")
                kwds.update(common_kwds)
                kwds.update(mask=sensible_mask(kwds), function=function)
                kwds.update(fig=fig, ax=ax)

                _, im = binned_posterior_probability(**kwds)

                #for _ in grid[:3]:
                #    ax.get_shared_y_axes().remove(_)

                #if k > 0:
                #    ax.set_ylabel("")
                #    ax.set_yticklabels([])

                ax.set_title(title)

            cax = grid.cbar_axes[0]
            cax.colorbar(im)
            cax.yaxis.set_major_locator(MaxNLocator(5))
            cax.set_ylabel(r"$\langle{}p_\textrm{single}\rangle$")

            return fig


        fig = plot_binned_posterior_probability_all_models(function, sensible_mask, common_kwds)
        savefig(fig, f"binned-posterior-probability-{function}")



        # TODO: make same but put all three on one figure.

        # Do it again for just the main-sequence.
        ms_common_kwds = common_kwds.copy()
        ms_common_kwds.update(bins=150)


        # Plot RV binned
        kwds = _get_binned_posterior_probability_data(sources, results, "rv", band="g")
        kwds.update(ms_common_kwds)
        kwds.update(mask=mainsequence_mask(kwds), function=function)
        
        # TODO: put colorbar on
        fig, _ = binned_posterior_probability(**kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-rv-{function}")
        

        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, "ast", band="g")
        kwds.update(ms_common_kwds)
        kwds.update(mask=mainsequence_mask(kwds), function=function)

        # TODO: put colorbar on
        fig, _ = binned_posterior_probability(**kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-ast-{function}")


        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, model_name=None, band="g")
        kwds.update(ms_common_kwds)
        kwds.update(mask=mainsequence_mask(kwds), function=function)

        # TODO: put colorbar on
        fig, _ = binned_posterior_probability(**kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-joint-{function}")

        # TODO: make same but put all three on one figure.

        fig = plot_binned_posterior_probability_all_models(function, mainsequence_mask, common_kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-{function}")


        raise a


        # Plot the typical prediction from the GP across the parameters of interest.
        common_gp_expectation_kwds = dict(function=function,
                                          subsample=None,
                                          bins=150,
                                          interpolation="none",
                                          min_entries_per_bin=5,
                                          cmap=_common_continuous_cmap,
                                          norm_percentiles=(5, 50, 95),
                                          latex_labels=dict(bp_rp=r"$\textrm{G}_\textrm{BP} - \textrm{G}_\textrm{RP}$",
                                                            absolute_g_mag=r"$M_\textrm{G}$",
                                                            apparent_g_mag=r"$\textrm{G}$",
                                                            absolute_rp_mag=r"$M_{G_\textrm{RP}}$",
                                                            apparent_rp_mag=r"$\textrm{G}_\textrm{RP}$"))

        # Do astrometry.
        kwds = _get_binned_gp_expectation_values(sources, results, "ast", band="g",
                                                 parameter_names=("mu_single", "sigma_single"))

        kwds.update(common_gp_expectation_kwds)
        kwds["latex_labels"].update(mu_single=r"$\langle\mu_\mathrm{ast,single}\rangle$",
                                    sigma_single=r"$\langle\sigma_\mathrm{ast,single}\rangle$")
        
        fig = binned_gp_expectation_values(**kwds)
        savefig(fig, f"binned-gp-expectation-values-ast-{kwds['function']}")


        # Do radial velocity.
        kwds = _get_binned_gp_expectation_values(sources, results, "rv", band="rp",
                                                 parameter_names=("mu_single", "sigma_single"))
        kwds.update(common_gp_expectation_kwds)
        kwds["latex_labels"].update(mu_single=r"$\langle\mu_\mathrm{rv,single}\rangle$",
                                    sigma_single=r"$\langle\sigma_\mathrm{rv,single}\rangle$")

        fig = binned_gp_expectation_values(**kwds)
        savefig(fig, f"binned-gp-expectation-values-rv-{kwds['function']}")
