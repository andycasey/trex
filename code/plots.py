
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

# First: Plot GP predictions across HRD.

import mpl_utils



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




def _get_rv_excess_for_sb9(sources, results, sb9_catalog, **kwargs):


    # Do the cross-match.
    group_name = "rv/gp_predictions"
    source_ids = results[f"{group_name}/source_id"][()]
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

    # Since all of these are binaries, let's just show the excess as is.

    K_est = rv_jitter - model_mu[:, 0]
    e_K_est = np.sqrt(model_sigma[:, 0]**2 + model_mu[:, 1] + model_sigma[:, 1])



    kwds = dict(sb9_K1=sb9_K1,
                e_sb9_K1=e_sb9_K1,
                K_est=K_est,
                e_K_est=e_K_est)
    kwds.update(kwargs)

    return kwds









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

    figsize = kwargs.pop("figsize", (6, 4.90))
    if figsize is None:
        figsize = (L * 3, K * 3)

    fig, axes = plt.subplots(K, L, figsize=figsize, constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, axes_row in enumerate(axes):

        ax_abs, ax_app = axes_row
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
            ax.set_aspect("auto")

            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(6))

            if ax.is_last_row():

                # labels
                None
                ax.set_xlabel(get_label("bp_rp", "bp - rp"))

            else:
                ax.set_xticks([])

            if ax.is_first_col():
                # labels
                ax.set_ylabel(get_label(f"absolute_{band}_mag", f"absolute {band} mag"))

            else:
                ax.set_ylabel(get_label(f"apparent_{band}_mag", f"apparent {band} mag"))

                
        # Do axes, etc.
        """
        cbar = fig.colorbar(mappable=im, ax=tuple(axes_row))
        """
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax_app)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(5))

        if colorbar_labels is not None:
            cbar.set_label(colorbar_labels[i])
        

    #for ax in axes.flatten():
    #    ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

    fig.tight_layout()

    return fig


def log_density_gp_expectation_value(x, y, **kwargs):

    fig, ax = plt.subplots()

    _, im = mpl_utils.plot_binned_statistic(x, y, y,
                                            function="count", ax=ax, 
                                            full_output=True, **kwargs)

    return fig



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

    latex_labels = dict(
        bp_rp=r"$\textrm{bp - rp}$",
        absolute_g_mag=r"$\textrm{absolute G magnitude}$",
        apparent_g_mag=r"$\textrm{apparent G magnitude}$",
        mu_single=r"$\mu_\textrm{single}$",
        sigma_single=r"$\sigma_\textrm{single}$",
        mu_multiple=r"$\mu_\textrm{multiple}$",
        sigma_multiple=r"$\sigma_\textrm{multiple}$"
    )
    latex_labels = None


    kwds = dict(bp_rp=bp_rp,
                band=band,
                apparent_mag=apparent_mag,
                absolute_mag=absolute_mag,
                expectation_values=expectation_values,
                colorbar_labels=parameter_names,
                min_entries_per_bin=5,
                mask=mask,
                latex_labels=latex_labels)
    kwds.update(kwargs)

    return kwds


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





if __name__ == "__main__":


    import os
    import sys
    import h5py as h5
    import yaml
    from hashlib import md5

    from astropy.table import Table


    config_path = sys.argv[1]

    with open(config_path, "r") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    pwd = os.path.dirname(config_path)

    # Generate a unique hash.
    config_copy = config.copy()
    for k in config_copy.pop("ignore_keywords_when_creating_hash", []):
        if k in config_copy: 
            del config_copy[k]

    unique_hash = md5((f"{config_copy}").encode("utf-8")).hexdigest()[:5]

    pwd = os.path.dirname(config_path)
    results_path = os.path.join(pwd, config["results_path"].format(unique_hash=unique_hash))

    # TODO: REMOVE THIS
    results = h5.File(results_path, "r")

    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")

    sources = data["sources"]

    # Load SB9 catalog.
    """
    Plot radial velocity semi-amplitude against our estimate for binary systems
    in the SB9 catalog.
    """


    sb9_path = os.path.join(pwd, "data/catalogs/sb9_xm_gaia.fits")
    sb9_catalog = Table.read(sb9_path)

    kwds = _get_rv_excess_for_sb9(sources, results, sb9_catalog)

    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    fig = plt.figure()
    ax, ax_diff = (plt.subplot(gs[1]), plt.subplot(gs[0]))

    x, y = (kwds["sb9_K1"], kwds["K_est"])
    x_err, y_err = (kwds["e_sb9_K1"], kwds["e_K_est"])

    scatter_kwds = dict(s=10, facecolor="tab:blue")

    ax.scatter(x, y, **scatter_kwds)

    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=None, edgecolor="#000000",
                zorder=-1)

    ax_diff.scatter(x, y-x, **scatter_kwds)

    # Set common x limts.

    # Set y lims 

    ax.set_xlabel(r"$K_1\,\,/\,\,\mathrm{km\,s}^{-1}$")
    ax.set_ylabel(r"$K_\mathrm{est}\,\,/\,\,\mathrm{km\,s}^{-1}$")

    raise a
    lims = np.array([ax.get_xlim(), ax.get_ylim()])
    lims = (0, np.max(lims))
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    lim = np.max(np.abs(ax_diff.get_ylim()))
    ax_diff.set_ylim(-lim, +lim)

    ax_diff.axhline(0, c="#666666", linestyle=":", zorder=-1, linewidth=0.5)

    ax_diff.yaxis.set_major_locator(MaxNLocator(3))
    ax_diff.xaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    fig.tight_layout()




    raise a


    kwds = _get_binned_posterior_probability_data(sources, results, "rv", band="rp")
    mask = (kwds["absolute_mag"] > -10) \
         * (kwds["bp_rp"] < 4) \
         * (kwds["absolute_mag"] < 10)

    kwds.update(min_entries_per_bin=2,
                bins=200, 
                interpolation="none",
                function="median",
                mask=mask, cmap="magma")


    fig = binned_posterior_probability(**kwds)
   



    raise a



    kwds = _get_binned_gp_expectation_values(
                                       sources, results, "rv",
                                       band="rp", 
                                       parameter_names=("mu_single", "sigma_single"),
                                       function="median",
                                       subsample=None,
                                       bins=150, interpolation="none",
                                       min_entries_per_bin=2,
                                       cmap="magma")

    mask = None
    mask = (kwds["bp_rp"] < 3) \
         * (kwds["bp_rp"] > 0.2) \
         * (kwds["absolute_mag"] > -4) \
         * (kwds["apparent_mag"] > 6)

    kwds.update(figsize=(9.5, 7.5),
                mask=mask,
                norm_percentiles=(16, 50, 84))

    fig = binned_gp_expectation_values(**kwds)





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



    from matplotlib.colors import LogNorm

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

