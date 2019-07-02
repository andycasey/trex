
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from matplotlib.colors import LogNorm 
# First: Plot GP predictions across HRD.
from mpl_toolkits.axes_grid1 import ImageGrid, AxesGrid

import mpl_utils


def density_rv_excess_vs_absolute_magnitude(K_est, absolute_mag, **kwargs):

    fig, ax = plt.subplots()

    x, y = (K_est, absolute_mag)


    bins = (np.logspace(0, 3.1, 201),
            np.linspace(-10, 10, 201))


    _, im = mpl_utils.plot_binned_statistic(x, y, y,
                                            function="count", ax=ax, 
                                            full_output=True, interpolation="none",
                                            bins=bins,
                                            norm=LogNorm(),
                                            cmap="Greys",
                                            )
    ax.set_ylim(ax.get_ylim()[::-1])


    # TODO: create a mirrored axis or something?
    # TODO: this is all fucked up.
    # TODO: basically we want to have a log-x scale for the bins (which works)
    #       but then we can't tell the axis that the scaling is log otherwise it blows up

    # TODO: alter Equation 3 from arxiv:1711.00660 and show in figure.

    ticks = np.unique(np.log10(bins[0]).astype(int))
    ax.set_xticks(10**ticks)
    ax.set_xticklabels([r"$10^{{{0}}}$".format(_) for _ in ticks])

    # TODO, this is all fucked up
    print("this is not for publication")

    return fig







def scatter_period_and_rv_semiamplitude_for_known_binaries(P, K, ratio, P_err=None, K_err=None,
                                                           **kwargs):

    fig, ax = plt.subplots()

    
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

        
        scatter_kwds = dict(s=15, c=ratio[k][mask], cmap="copper", vmin=0, vmax=1)
        scatter_kwds.update(common_scatter_kwds)
        scatter_kwds.update(collection_kwds.get(k, dict()))

        x, y = (P[k][mask], K[k][mask])

        scat = ax.scatter(x, y, **scatter_kwds)

        if P_err[k] is not None or K_err[k] is not None:
            errorbar_kwds = dict(fmt="none", c="#000000", zorder=-1)
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

    ax.set_xlabel(r"$\mathrm{period}$ $/$ $\textrm{days}^{-1}$")
    ax.set_ylabel(r"$K$ $/$ $\mathrm{km\,s}^{-1}$")

    ax.set_aspect(xa/ya)

    fig.tight_layout()

    cbar = plt.colorbar(scat)
    cbar.set_label(r"$p(\mathrm{single}|j_{rv})$")

    # Show observing period of Gaia Dr2.
    observing_span = 668 # days : https://www.cosmos.esa.int/web/gaia/dr2
    ax.axvline(observing_span, c="#666666", zorder=-1, linestyle=":", linewidth=1)
    ax.axvline(2 * observing_span, c="#666666", zorder=-1, linestyle="-", linewidth=1)

    return fig



def scatter_excess_rv_jitter_for_known_binaries(K_catalog, K_est, K_catalog_err=None, K_est_err=None,
                                                **kwargs):

    height_ratio = 5

    fig = kwargs.pop("figure", None) 
    if fig is None:
        fig = plt.figure(figsize=kwargs.pop("figsize", (5, 5.5)))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, height_ratio])
        ax, ax_diff = (plt.subplot(gs[1]), plt.subplot(gs[0]))

    else:
        ax, ax_diff = fig.axes[:2]

    x, y = (K_catalog, K_est)
    x_err, y_err = (K_catalog_err, K_est_err)

    scatter_kwds = dict(s=10, facecolor="tab:blue")
    scatter_kwds.update(kwargs.pop("scatter_kwds", dict()))

    ax.scatter(x, y, **scatter_kwds)
    if x_err is not None or y_err is not None:
        errorbar_kwds = dict(fmt="none", edgecolor="#000000", zorder=-1)
        errorbar_kwds.update(kwargs.pop("errorbar_kwds", dict()))
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, **errorbar_kwds)


    ax_diff.scatter(x, y-x, **scatter_kwds)

    ax.set_xlabel(r"$K\,\,/\,\,\mathrm{km\,s}^{-1}$")
    ax.set_ylabel(r"$K_\mathrm{est}\,\,/\,\,\mathrm{km\,s}^{-1}$")

    lims = np.array([ax.get_xlim(), ax.get_ylim()])
    lims = (0, np.max(lims))

    ax.plot(lims, lims, c="#666666", linestyle=":", zorder=-1, linewidth=0.5)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax_diff.set_xlim(lims)

    lim = np.max(np.abs(ax_diff.get_ylim()))
    ax_diff.set_ylim(-lim, +lim)

    ax_diff.axhline(0, c="#666666", linestyle=":", zorder=-1, linewidth=0.5)

    ax_diff.yaxis.set_major_locator(MaxNLocator(3))
    ax_diff.set_xticks([])
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    
    aspect = lambda ax: np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim())
    ax.set_aspect(aspect(ax))
    ax_diff.set_aspect(aspect(ax_diff) / height_ratio)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)
    
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

            ax.set_xlabel(get_label("bp_rp", "bp - rp"))

        ax_abs.set_ylabel(get_label(f"absolute_{band}_mag", f"absolute {band} mag"))
        ax_app.set_ylabel(get_label(f"apparent_{band}_mag", f"apparent {band} mag"))

        caxes = (grid.cbar_axes[0], grid.cbar_axes[1])
        cax = caxes[i]
        cax.colorbar(im)

        cax.toggle_label(True)
        cax.axis[cax.orientation].set_label(get_label(colorbar_labels[i], colorbar_labels[i]))

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
        ax = fig.axes[0]

    latex_labels = kwargs.pop("latex_labels", dict())
    band = kwargs.pop("band", None)

    kwds = dict(function="mean", ax=ax, full_output=True)
    kwds.update(kwargs)
    _, im = mpl_utils.plot_binned_statistic(bp_rp, absolute_mag, ratios, **kwds)

    ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(r"{bp - rp}")
    ax.set_ylabel(r"{absolute G magnitude}")
    ax.set_facecolor("#eeeeee")   
    fig.tight_layout()

    return fig


def hist_literature_single_stars_and_binaries(sb9, soubiran, **kwargs):

    N = 3
    fig, axes = plt.subplots(1, 3, figsize=kwargs.pop("figsize", (9, 3.5)))

    latex_labels = (r"{radial velocity jitter} $j_\mathrm{rv}$ $/$ {km\,s}$^{-1}$", r"{astrometric jitter} $j_\mathrm{ast}$", r"{photometric jitter} $j_\mathrm{phot}$")

    kwds = dict(histtype="barstacked")

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

        bins = np.logspace(int(np.log10(x.min())) - 0.5, 0.5 + int(np.log10(x.max())), 20)
        kwds["bins"] = np.log10(bins)
        kwds.update(kwargs)
        ax.hist([np.log10(x1), np.log10(x2)], **kwds)

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
            d = results[f"{model_name}/gp_predictions/{p}"][()][:, 0]
            expectation_values.append(np.abs(d) if p.startswith("sigma_") else d)

        expectation_values = tuple(expectation_values)

        # Now get the data.
        indices = results["indices/data_indices"][()]

        bp_rp = sources["bp_rp"][()][indices]

        # TODO: need the right absolute/apparent mags to use! FUCK!
        apparent_mag = sources["phot_g_mean_mag"][()][indices]
        absolute_mag = sources["absolute_g_mag"][()][indices]

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



    def _get_binned_posterior_probability_data(sources, results, model_name, band="g", **kwargs):


        if model_name == "joint":
            ratios = results[f"model_selection/likelihood/joint_ratio_single"][()]
        else:
            ratios = results[f"model_selection/likelihood/{model_name}/ratio_single"][()]

        indices = results["indices/data_indices"][()]

        bp_rp = sources["bp_rp"][()][indices]

        # TODO: need the right absolute/apparent mags to use! FUCK!
        apparent_mag = sources[f"phot_{band}_mean_mag"][()][indices]
        absolute_mag = sources[f"absolute_{band}_mag"][()][indices]

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


    def _get_rv_excess(sources, results, **kwargs):

        group_name = "rv/gp_predictions"
        source_ids = results[f"{group_name}/source_id"][()]

        indices = results["indices/data_indices"][()]
        rv_jitter = sources["rv_jitter"][()][indices]
        absolute_g_mag = sources["absolute_g_mag"][()][indices]

        model_mu = results[f"{group_name}/mu_single"][()]
        model_sigma = results[f"{group_name}/sigma_single"][()]

        
        K_est = rv_jitter - model_mu[:, 0]
        K_est_err = np.sqrt(model_sigma[:, 0]**2 + model_mu[:, 1] + model_sigma[:, 1])


        return dict(K_est=K_est,
                    K_est_err=K_est_err,
                    source_ids=source_ids,
                    absolute_g_mag=absolute_g_mag)



        
    def _get_rv_excess_for_apw(sources, results, apw_catalog, **kwargs):


        # Do the cross-match.
        group_name = "rv/gp_predictions"
        source_ids = results[f"{group_name}/source_id"][()]
        catalog_source_ids = apw_catalog["source_id"]


        idx, catalog_idx = cross_match(source_ids, catalog_source_ids)

        apw_K = apw_catalog["K"][catalog_idx]
        e_apw_K = apw_catalog["K_err"][catalog_idx]

        # Calculate the RV excess from our model.
        # TODO: Put this into a single place!

        source_ids = source_ids[idx]
        model_mu = results[f"{group_name}/mu_single"][()][idx]
        model_sigma = results[f"{group_name}/sigma_single"][()][idx]

        # TODO: Here we are assuming what the RV jitter term is instead of reading
        #       it from the config.
        indices = results["indices/data_indices"][()]
        rv_jitter = sources["rv_jitter"][()][indices][idx]

        ratio = results["model_selection/likelihood/rv/ratio_single"][()][idx]

        # Since all of these are binaries, let's just show the excess as is.

        K_est = rv_jitter - model_mu[:, 0]
        e_K_est = np.sqrt(model_sigma[:, 0]**2 + model_mu[:, 1] + model_sigma[:, 1])

        kwds = dict(apw_K=apw_K,
                    e_apw_K=e_apw_K,
                    K_est=K_est,
                    e_K_est=e_K_est,
                    apw_P=apw_catalog["P"][catalog_idx],
                    e_apw_P=apw_catalog["P_err"][catalog_idx],
                    ratio=ratio)

        kwds.update(kwargs)

        return kwds



    def _get_rv_excess_for_sb9(sources, results, sb9_catalog, use_sb9_mask=True, **kwargs):


        # Do the cross-match.
        group_name = "rv/gp_predictions"
        source_ids = results[f"{group_name}/source_id"][()]

        # Only include good results from SB9
        if use_sb9_mask:
            sb9_mask = (sb9_catalog["f_K1"] != ">") \
                     * (sb9_catalog["f_T0"] == 0) \
                     * (sb9_catalog["Grade"] > 0) \
                     * (sb9_catalog["f_omega"] != "a") \
                     * (sb9_catalog["o_K1"] > 0)
            sb9_catalog = Table(sb9_catalog[sb9_mask])

        sb9_source_ids = sb9_catalog["source_id"]       

        idx, sb9_idx = cross_match(source_ids, sb9_source_ids)

        sb9_K1 = sb9_catalog["K1"][sb9_idx]
        e_sb9_K1 = sb9_catalog["e_K1"][sb9_idx]

        # Calculate the RV excess from our model.

        source_ids = source_ids[idx]
        model_mu = results[f"{group_name}/mu_single"][()][idx]
        model_sigma = results[f"{group_name}/sigma_single"][()][idx]

        # TODO: Here we are assuming what the RV jitter term is instead of reading
        #       it from the config.
        indices = results["indices/data_indices"][()]
        rv_jitter = sources["rv_jitter"][()][indices][idx]
        rv_nb_transits = sources["rv_nb_transits"][()][indices][idx]

        ratio = results["model_selection/likelihood/rv/ratio_single"][()][idx]

        # Since all of these are binaries, let's just show the excess as is.

        K_est = (rv_jitter - model_mu[:, 0]) #/ np.sqrt(0.5 * np.pi * rv_nb_transits)
        e_K_est = np.sqrt(model_sigma[:, 0]**2 + model_mu[:, 1] + model_sigma[:, 1])

        kwds = dict(sb9_K1=sb9_K1,
                    e_sb9_K1=e_sb9_K1,
                    K_est=K_est,
                    e_K_est=e_K_est,
                    sb9_P=sb9_catalog["Per"][sb9_idx],
                    e_sb9_P=sb9_catalog["e_Per"][sb9_idx],
                    ratio=ratio)

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
    results_path = os.path.join(results_dir, "results.hdf5")

    # TODO: Store data in the results dir so we don't have to do this all the time.
    # TODO: And somehow store the PWD?
    pwd = "../"
    data_path = "../data/sources.hdf5"

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

        

    # Plot the distributions of jitter for comparable catalogs of single stars and binaries.
    kwds = _xm_literature_single_stars_and_binaries(sb9_catalog, soubiran_catalog)
    kwds.update(color=["#000000", "#BBBBBB"])
    fig = hist_literature_single_stars_and_binaries(**kwds)
    savefig(fig, "hist-literature-single-stars-and-binaries")



    # Plot radial velocity semi-amplitude against our estimate for binary systems from APW
    apw_path = os.path.join(pwd, "data/catalogs/apw-highK-unimodal-xm-gaia.fits")
    apw_catalog = Table.read(apw_path)

    apw_kwds = _get_rv_excess_for_apw(sources, results, apw_catalog)
    fig = scatter_period_and_rv_semiamplitude_for_known_binaries(P=apw_kwds["apw_P"],
                                                                 K=apw_kwds["apw_K"],
                                                                 ratio=apw_kwds["ratio"],
                                                                 P_err=apw_kwds["e_apw_P"],
                                                                 K_err=apw_kwds["e_apw_K"])
    savefig(fig, "scatter-period-and-rv-semiamplitude-for-known-binaries-apw")


    fig = scatter_excess_rv_jitter_for_known_binaries(K_catalog=apw_kwds["apw_K"],
                                                      K_catalog_err=apw_kwds["e_apw_K"],
                                                      K_est=apw_kwds["K_est"],
                                                      K_est_err=apw_kwds["e_K_est"])
    savefig(fig, "scatter-excess-rv-jitter-for-known-binaries-apw")



    # Plot radial velocity semi-amplitude against our estimate for binary systems in the SB9 catalog.    
    sb9_path = os.path.join(pwd, "data/catalogs/sb9-xm-gaia.fits")
    sb9_catalog = Table.read(sb9_path)

    sb9_kwds = _get_rv_excess_for_sb9(sources, results, sb9_catalog)


    fig = scatter_period_and_rv_semiamplitude_for_known_binaries(P=sb9_kwds["sb9_P"],
                                                                 K=sb9_kwds["sb9_K1"],
                                                                 ratio=sb9_kwds["ratio"],
                                                                 P_err=sb9_kwds["e_sb9_P"],
                                                                 K_err=sb9_kwds["e_sb9_K1"])
    savefig(fig, "scatter-period-and-rv-semiamplitude-for-known-binaries-sb9")

    fig = scatter_excess_rv_jitter_for_known_binaries(K_catalog=sb9_kwds["sb9_K1"],
                                                      K_catalog_err=sb9_kwds["e_sb9_K1"],
                                                      K_est=sb9_kwds["K_est"],
                                                      K_est_err=sb9_kwds["e_K_est"])
    savefig(fig, "scatter-excess-rv-jitter-for-known-binaries-sb9")



    # Now joint (SB9 + APW)
    sb9_kwds = _get_rv_excess_for_sb9(sources, results, sb9_catalog, use_sb9_mask=False)
    kwds = dict(P=dict(sb9=sb9_kwds["sb9_P"],
                       apw=apw_kwds["apw_P"]),
                K=dict(sb9=sb9_kwds["sb9_K1"],
                       apw=apw_kwds["apw_K"]),
                ratio=dict(sb9=sb9_kwds["ratio"],
                           apw=apw_kwds["ratio"]),
                P_err=dict(sb9=sb9_kwds["e_sb9_P"],
                           apw=apw_kwds["e_apw_P"]),
                K_err=dict(sb9=sb9_kwds["e_sb9_K1"],
                           apw=apw_kwds["e_apw_K"]),
                collection_kwds=dict(sb9=dict(marker="s"), apw=dict(marker="^")))
    fig = scatter_period_and_rv_semiamplitude_for_known_binaries(**kwds)
    savefig(fig, "scatter-period-and-rv-semiamplitude-for-known-binaries-all")



    # Plot log density of sources and their excess RV jitter.
    kwds = _get_rv_excess(sources, results)

    fig = density_rv_excess_vs_absolute_magnitude(K_est=kwds["K_est"],
                                                  absolute_mag=kwds["absolute_g_mag"])

    # TODO: save this one


    sensible_mask = lambda k: (k["absolute_mag"] < 10) \
                            * (k["absolute_mag"] > -6) \
                            * (k["bp_rp"] > 0.25) \
                            * (k["bp_rp"] < 4)

    mainsequence_mask = lambda k: sensible_mask(k) \
                                * (k["absolute_mag"] > 2)

    common_kwds = dict(min_entries_per_bin=10,
                       bins=200,
                       interpolation="none",
                       subsample=None,
                       cmap="magma")

    for function in ("mean", "median"):

        # Plot RV binned
        kwds = _get_binned_posterior_probability_data(sources, results, "rv", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=sensible_mask(kwds), function=function)
        
        # TODO: put colorbar on
        fig = binned_posterior_probability(**kwds)
        savefig(fig, f"binned-posterior-probability-rv-{function}")
        

        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, "ast", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=sensible_mask(kwds), function=function)

        # TODO: put colorbar on
        fig = binned_posterior_probability(**kwds)
        savefig(fig, f"binned-posterior-probability-ast-{function}")


        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, "joint", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=sensible_mask(kwds), function=function)

        # TODO: put colorbar on
        fig = binned_posterior_probability(**kwds)
        savefig(fig, f"binned-posterior-probability-joint-{function}")

        # TODO: make same but put all three on one figure.

    # Do it again for just the main-sequence.
    for function in ("mean", "median"):

        # Plot RV binned
        kwds = _get_binned_posterior_probability_data(sources, results, "rv", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=mainsequence_mask(kwds), function=function)
        
        # TODO: put colorbar on
        fig = binned_posterior_probability(**kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-rv-{function}")
        

        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, "ast", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=mainsequence_mask(kwds), function=function)

        # TODO: put colorbar on
        fig = binned_posterior_probability(**kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-ast-{function}")


        # Plot ast
        kwds = _get_binned_posterior_probability_data(sources, results, "joint", band="g")
        kwds.update(common_kwds)
        kwds.update(mask=mainsequence_mask(kwds), function=function)

        # TODO: put colorbar on
        fig = binned_posterior_probability(**kwds)
        savefig(fig, f"main-sequence-binned-posterior-probability-joint-{function}")

        # TODO: make same but put all three on one figure.



    for function in ("mean", "median"):

        # Plot the typical prediction from the GP across the parameters of interest.
        common_gp_expectation_kwds = dict(function=function,
                                          subsample=None,
                                          bins=150,
                                          interpolation="none",
                                          min_entries_per_bin=5,
                                          cmap="magma",
                                          norm_percentiles=(16, 50, 84),
                                          latex_labels=dict(bp_rp=r"{bp - rp}",
                                                            absolute_g_mag=r"{absolute G magnitude}",
                                                            apparent_g_mag=r"{apparent G magnitude}"))

        # Do astrometry.
        kwds = _get_binned_gp_expectation_values(sources, results, "ast", band="g",
                                                 parameter_names=("mu_single", "sigma_single"))

        kwds.update(common_gp_expectation_kwds)
        kwds["latex_labels"].update(mu_single=r"$\mu_\mathrm{ast,single}$",
                                    sigma_single=r"$\sigma_\mathrm{ast,single}$")
        
        fig = binned_gp_expectation_values(**kwds)
        savefig(fig, f"binned-gp-expectation-values-ast-{kwds['function']}")


        # Do radial velocity.
        kwds = _get_binned_gp_expectation_values(sources, results, "rv", band="rp",
                                                 parameter_names=("mu_single", "sigma_single"))
        kwds.update(common_gp_expectation_kwds)
        kwds["latex_labels"].update(mu_single=r"$\mu_\mathrm{rv,single}$",
                                    sigma_single=r"$\sigma_\mathrm{rv,single}$")

        fig = binned_gp_expectation_values(**kwds)
        savefig(fig, f"binned-gp-expectation-values-rv-{kwds['function']}")
