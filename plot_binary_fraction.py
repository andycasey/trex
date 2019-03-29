

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from scipy.stats import binned_statistic, binned_statistic_2d
from collections import OrderedDict
from mpl_utils import (mpl_style, plot_binned_statistic, plot_histogram_steps)

plt.style.use(mpl_style)

galah = fits.open("/Users/arc/Downloads/velociraptor.v11-xm-galah.fits")[1].data
raveon = fits.open("/Users/arc/Downloads/velociraptor.v11-xm-raveon.fits")[1].data
apogee = fits.open("/Users/arc/Downloads/velociraptor.v11-xm-apogee.fits")[1].data
#velociraptor = fits.open("data/velociraptor-catalog.rc.11.fits")[1].data


datasets = OrderedDict([
    [r"$\textsl{GALAH}$", galah],
    [r"$\textsl{APOGEE}$ $(\textrm{DR14})$", apogee],
    [r"$\textsl{RAVE}$-$\textrm{on}$", raveon],
#    [r"$\textsl{{Gaia}}$ $(\textrm{{{0}}})$".format(len(velociraptor)), velociraptor]
])

label_names = (
    "bp_rp",
    "fe_h",
    "rv_single_epoch_scatter", 
    "astrometric_unit_weight_error")

latex_labels = dict(
    bp_rp=r"$\textrm{bp} - \textrm{rp}$",
    fe_h=r"$[\textrm{Fe}/\textrm{H}]$",
    rv_single_epoch_scatter=r"$\sigma_{v_r} / \textrm{km\,s}^{-1}$",
    astrometric_unit_weight_error=r"$\textrm{RUWE} / \textrm{mas}$")

bin_number = 25
specific_bins = dict(
  fe_h=np.linspace(-2.5, 0.5, bin_number),
  astrometric_unit_weight_error=np.linspace(0, 5, bin_number),
  rv_single_epoch_scatter=np.linspace(0, 25, bin_number))

for label_name in label_names:
    if label_name not in specific_bins:
        ranges = np.array([
          [np.min(dataset[label_name]), np.max(dataset[label_name])] \
          for _, dataset in datasets.items()])

        min_range, max_range = np.nanmin(ranges), np.nanmax(ranges)
        specific_bins[label_name] = np.linspace(min_range, max_range, bin_number)


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


from matplotlib.cm import Spectral_r
p = np.linspace(0, 255, 5)

colors = discrete_cmap(7, "Spectral_r")

fig, axes = plt.subplots(1, 2, figsize=(8, 6))
axes = np.atleast_1d(axes).flatten()

for j, (dataset_name, dataset) in enumerate(datasets.items()):

    tau_mask = np.isfinite(dataset["tau_single"])

    p_binary = 1 - np.round(dataset["tau_single"]).astype(int)


    for i, (ax, label_name) in enumerate(zip(axes, label_names)):
        #print(j, label_name, dataset_name, label_name in dataset.dtype.names)
        #if label_name not in dataset.dtype.names: 
        #    raise a
        #    continue

        mask = tau_mask \
             * np.isfinite(dataset[label_name])

        bins = specific_bins.get(label_name, bin_number)

        mean, edge, bin_index = binned_statistic(dataset[label_name][mask],
                                                 p_binary[mask],
                                                 statistic="mean", bins=bins)
        var, _, __ = binned_statistic(dataset[label_name][mask],
                                      p_binary[mask],
                                      statistic=np.var, bins=bins)

        count, _, __ = binned_statistic(dataset[label_name][mask],
                                      p_binary[mask],
                                      statistic="count", bins=bins)


        yerr = 2 * np.sqrt(var/(count - 1))
        #yerr = np.sqrt(var)
        center = edge[:-1] + 0.5 * np.diff(edge)

        label = dataset_name if i == 0 else None
        plot_histogram_steps(ax, center, mean, yerr, lw=2, label=label, c=colors(j))
        ax.set_xlabel(latex_labels.get(label_name, None))
        ax.set_ylabel(r"$\textrm{mean binary fraction}$")

        ax.set_xlim(edge[0], edge[-1])
        ax.set_ylim(-0.025, 1.025)

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        if j == 0:
            ax.plot([edge[0] - 1, 1 + edge[-1]], [0, 0],  c="#666666", lw=0.5, linestyle=":", zorder=-1)
            ax.plot([edge[0] - 1, 1 + edge[-1]], [1, 1],  c="#666666", lw=0.5, linestyle=":", zorder=-1)


axes[0].legend(frameon=False)
for ax in axes:
    ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

plt.show()
fig.tight_layout()
plt.draw()
plt.show()

fig.savefig("binary-fraction.pdf", dpi=300)
