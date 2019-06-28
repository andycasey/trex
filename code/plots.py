
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MaxNLocator

# First: Plot GP predictions across HRD.

import mpl_utils







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

