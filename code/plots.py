
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
    def _get_closest_sb9_subset(sb9_catalog, soubiran_catalog, parameter_names=None):
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

        #A = A[A_idx][keep]
        #B = B[keep]

        #return dict(sb9=A, soubiran==B)


        fig, axes = plt.subplots(1, A.shape[1] + 1)
        for i, ax in enumerate(axes[:-1]):
            ax.hist([A[:, i], B[:, i]], bins=25)

        axes[-1].hist(dist[keep])

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

                else:
                    assert False


        N_bins = 20
        bins = dict(rv_jitter=np.linspace(0, 20, N_bins),
                    ast_jitter=np.linspace(0, 25, N_bins),
                    phot_g_variability=np.linspace(0, 0.5, N_bins))
        bins = dict()

        fig, axes = plt.subplots(2, 3)

        def _get_data(pn):
            x1 = sb9_catalog[pn][A_idx][keep]
            x2 = soubiran_catalog[pn][keep]

            finite = np.isfinite(x1 * x2)

            V = np.hstack([np.array(x1[finite]), np.array(x2[finite])])

            return (x1[finite], x2[finite], V.min(), V.max())

        for i, (ax, pn) in enumerate(zip(axes[0], distinguishing_parameters)):

            x1, x2, Vmin, Vmax = _get_data(pn)


            kwds = dict(bins=np.linspace(Vmin, 1.1 * Vmax, N_bins), alpha=0.5)
            ax.hist(x1, label="multiple", facecolor="tab:blue", **kwds)
            ax.hist(x2, label="single", facecolor="tab:red", **kwds)


        x1 = sb9_catalog["rv_jitter"][A_idx][keep]
        x2 = soubiran_catalog["rv_jitter"][keep]

        y1 = sb9_catalog["ast_jitter"][A_idx][keep]
        y2 = soubiran_catalog["ast_jitter"][keep]

        axes[1, 0].scatter(x1, y1, facecolor="tab:blue", alpha=0.5)
        axes[1, 0].scatter(x2, y2, facecolor="tab:red", alpha=0.5)
        axes[1, 0].set_xlabel("rv")
        axes[1, 0].set_ylabel("ast")


        z1 = sb9_catalog["phot_g_variability"][A_idx][keep]
        z2 = soubiran_catalog["phot_g_variability"][keep]

        axes[1, 1].scatter(x1, z1, facecolor="tab:blue", alpha=0.5)
        axes[1, 1].scatter(x2, z2, facecolor="tab:red", alpha=0.5)
        axes[1, 1].set_xlabel("rv")
        axes[1, 1].set_ylabel("phot")



        axes[1, 2].scatter(y1, z1, facecolor="tab:blue", alpha=0.5)
        axes[1, 2].scatter(y2, z2, facecolor="tab:red", alpha=0.5)
        axes[1, 2].set_xlabel("ast")
        axes[1, 2].set_ylabel("phot")

        X1 = sb9_catalog["rv_jitter","ast_jitter","phot_g_variability"][A_idx]
        X1 = X1.as_array().view(np.float).data.reshape((-1, 3))

        X2 = soubiran_catalog["rv_jitter","ast_jitter","phot_g_variability"]
        X2 = X2.as_array().view(np.float).data.reshape((-1, 3))

        return (X1, X2)

        


    X1, X2 = _get_closest_sb9_subset(sb9_catalog, soubiran_catalog)

    def corner_scatter(X, label_names=None, show_ticks=False, fig=None, figsize=None,
                       **kwargs):
        """
        Make a corner plot where the data are shown as scatter points in each axes.

        :param X:
            The data, :math:`X`, which is expected to be an array of shape
            [n_samples, n_features].

        :param label_names: [optional]
            The label names to use for each feature.

        :param show_ticks: [optional]
            Show ticks on the axes.

        :param fig: [optional]
            Supply a figure (with [n_features, n_features] axes) to plot the data.

        :param figsize: [optional]
            Specify a size for the figure. This parameter is ignored if a `fig` is
            supplied.

        :returns:
            A figure with a corner plot showing the data.
        """

        N, D = X.shape
        assert N > D, "Stahp doing it wrong"
        K = D

        if fig is None:
            if figsize is None:
                figsize = (2 * K, 2 * K)
            fig, axes = plt.subplots(K, K, figsize=figsize)
        
        axes = np.array(fig.axes).reshape((K, K)).T

        bins = kwargs.pop("bins", [25] * D)

        kwds = dict(s=5, c="tab:blue", alpha=1, rasterized=True)
        kwds.update(kwargs)
        
        for i, x in enumerate(X.T):
            for j, y in enumerate(X.T):
                #if j == 0: continue

                try:
                    ax = axes[i, j]

                except:
                    continue


                if i > j:
                    ax.set_visible(False)
                    continue

                elif i == j:
                    ax.set_facecolor("#eeeeee")
                    ax.hist(x[np.isfinite(x)], bins=bins[i], facecolor=kwds["c"], alpha=kwds["alpha"], log=False)
                    ax.set_yticks([])
                    
                    if not isinstance(bins[i], int):
                        ax.set_xlim(bins[i][0], bins[i][-1])

                else:
                    ax.scatter(x, y, **kwds)

                    if not isinstance(bins[i], int):
                        ax.set_xlim(bins[i][0], bins[i][-1])

                    if not isinstance(bins[j], int):
                        ax.set_ylim(bins[j][0], bins[j][-1])


                if not show_ticks:
                    ax.set_xticks([])
                    ax.set_yticks([])

                else:
                    if not ax.is_last_row():
                        ax.set_xticks([])
                        ax.set_xticklabels([])
                    else:
                        ax.xaxis.set_major_locator(MaxNLocator(3))

                    if not ax.is_first_col():
                        ax.set_yticklabels([])
                        ax.set_yticks([])
                    else:
                        ax.yaxis.set_major_locator(MaxNLocator(3))


                if ax.is_last_row() and label_names is not None:
                    ax.set_xlabel(label_names[i])
                    
                if ax.is_first_col() and label_names is not None:
                    ax.set_ylabel(label_names[j])

                if ax.is_first_col() and ax.is_first_row():
                    ax.set_ylabel("")

                if i == j:
                    ax.set_yticks([])

        fig.tight_layout()
        
        return fig


    ok = np.all(np.isfinite(X1), axis=1) * np.all(np.isfinite(X2), axis=1) \
       * (X1[:, 2] < 1)

    X1 = X1[ok]
    X2 = X2[ok]

    X_all = np.vstack([X1, X2])
    for i in range(X_all.shape[1]):
        X_all[~np.isfinite(X_all[:, i]), i] = np.nanmean(X_all[:, i])

    bins = []
    for i in range(X_all.shape[1]):
        min_, max_ = (np.min(X_all[:, i]), np.max(X_all[:, i]))
        ptp = np.ptp(X_all[:, i])

        bins.append(np.linspace(min_ - 0.05 * ptp, max_ + 0.05 * ptp, 25))

    #bins = [np.linspace(np.min(X_all[:, i]), np.max(X_all[:, i]), 25) for i in range(X_all.shape[1])]


    fig = corner_scatter(X1, show_ticks=True, label_names=("rv","ast", "phot"), bins=bins, alpha=0.75)
    fig = corner_scatter(X2, show_ticks=True, label_names=("rv", "ast", "phot"), fig=fig,
                         c="tab:red", bins=bins, zorder=10, alpha=0.75)


    raise a
    



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



    raise a


    # Plot log density of sources and their excess RV jitter.

    kwds = _get_rv_excess(sources, results)

    fig = density_rv_excess_vs_absolute_magnitude(K_est=kwds["K_est"],
                                                  absolute_mag=kwds["absolute_g_mag"])


    raise a # because these get expensive.


    for function in ("mean", "median"):

        # Plot the typical prediction from the GP across the parameters of interest.
        common_gp_expectation_kwds = dict(function=function,
                                          subsample=None,
                                          bins=150,
                                          interpolation="none",
                                          min_entries_per_bin=5,
                                          cmap="magma",
                                          norm_percentiles=(5, 50, 95),
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




    raise a



    kwds = _get_binned_posterior_probability_data(sources, results, "ast", band="g")

    mask = None
    bin_number = 200

    kwds.update(min_entries_per_bin=5,
                bins=(np.linspace(0, 4, bin_number + 1),
                      np.linspace(-7.5, 7.5, bin_number + 1)),
                interpolation="none",
                subsample=1_000_000,
                function="median",
                mask=mask, 
                cmap="magma")


    fig = binned_posterior_probability(**kwds)
   

    # Everything until the next raise a is good
    raise a




    # Plot binned posterior probability.









    raise a



    kwds = _get_binned_posterior_probability_data(sources, results, "ast")
    mask = (kwds["absolute_mag"] > -3) \
         * (kwds["bp_rp"] < 4) \
         * (kwds["absolute_mag"] < 10)

    kwds.update(min_entries_per_bin=5,
                bins=200, 
                interpolation="none",
                function="median",
                mask=mask, cmap="magma")


    fig = binned_posterior_probability(**kwds)
   

    ax = fig.axes[0]
    ax.set_facecolor("#eeeeee")



    raise a



    kwds = _get_binned_gp_expectation_values(
                                       sources, results, "ast", 
                                       parameter_names=("mu_single", "sigma_single", "theta"),
                                       function="median",
                                       subsample=100000,
                                       bins=100, interpolation="none",
                                       min_entries_per_bin=10,
                                       cmap="magma")

    fig = binned_gp_expectation_values(**kwds)





    mask = (kwds["expectation_values"][1] < 0.5) \
         * (kwds["apparent_mag"] > 5)

    fig = log_density_gp_expectation_value(kwds["apparent_mag"], kwds["expectation_values"][1], 
                                            norm=LogNorm(), bins=200,
                                            interpolation="none",
                                            mask=mask, cmap="Greys",
                                            min_entries_per_bin=5)
    ax = fig.axes[0]
    ax.set_facecolor("#FFFFFF")
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))








    fig = log_density_gp_expectation_value(kwds["absolute_mag"], np.clip(kwds["expectation_values"][2], 0, 1),
                                            norm=LogNorm(), bins=250,
                                            interpolation="none",
                                            mask=None, cmap="magma",
                                            min_entries_per_bin=1)
    fig.axes[0].set_facecolor("#eeeeee")

    fig.axes[0].set_ylim(fig.axes[0].get_ylim()[::-1])



    # Plot the mean likelihood ratio.

