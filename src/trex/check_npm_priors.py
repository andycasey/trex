
import itertools
import yaml
import numpy as np
import sys
import os
import h5py as h5
from tqdm import tqdm
from scipy.stats import binned_statistic_dd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import utils

assert __name__ == "__main__"

config_path = sys.argv[1]

with open(config_path, "r") as fp:
    config = yaml.load(fp, Loader=yaml.Loader)

M = 2
N_x = 2048
N_grid_default = 30
N_grid = dict()
epsilon = 1e-3

x_max = dict(ast=50, rv=100)

debug = True

def get_mu_multiple(mu_single, sigma_single, sigma_multiple, mu_multiple_scalar):
    return np.log(mu_single + mu_multiple_scalar * sigma_single) + sigma_multiple**2


def ln_prior(theta, mu_single, sigma_single, sigma_multiple):
    if sigma_multiple < sigma_single/(mu_single + sigma_single) \
    or sigma_single < 0.125 * mu_single:
        return -np.inf
    return 0


# Load sources: we will need these later on
pwd = os.path.dirname(config_path)
data_path = os.path.join(pwd, config["data_path"])
data = h5.File(data_path, "r")
sources = data["sources"]

for model_name, model_config in config["models"].items():

    bounds = model_config.get("bounds", None)
    if bounds is None:
        print(f"Skipping '{model_name}' model because there are no parameter bounds set!")
        continue

    print(f"Running '{model_name}' model with debug = {debug}!")

    mu_multiple_scalar = model_config["mu_multiple_scalar"]
    predictor_label_name = model_config["predictor_label_name"]

    # Check the bounds of the predictor label.
    x = sources[predictor_label_name][()]
    xm = x_max.get(model_name, np.nanmax(x))
    xi = np.linspace(epsilon, xm, N_x)

    # Build up a grid of these parameters:
    thetas = np.linspace(*bounds["theta"], N_grid.get("theta", N_grid_default))
    mu_singles = np.linspace(*bounds["mu_single"], N_grid.get("mu_single", N_grid_default))
    sigma_singles = np.linspace(*bounds["sigma_single"], N_grid.get("sigma_single", N_grid_default))
    sigma_multiples = np.linspace(*bounds["sigma_multiple"], N_grid.get("sigma_multiple", N_grid_default))

    # Clip thetas so we don't end up with -inf's!
    thetas = np.clip(thetas, epsilon, 1 - epsilon)

    grid = np.array(list(itertools.product(thetas, mu_singles, sigma_singles, sigma_multiples)))
    support_flag = np.zeros((grid.shape[0], ), dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for ii, (theta, mu_single, sigma_single, sigma_multiple) in enumerate(tqdm(grid[::-1])):

            # Do in reverse order because the last edge cases are more problematic.
            i = -(ii + 1)

            # Check prior.
            if not np.isfinite(ln_prior(theta, mu_single, sigma_single, sigma_multiple)):
                # Don't evaluate support at places outside our prior space.
                continue

            mu_multiple = get_mu_multiple(mu_single, sigma_single, sigma_multiple, mu_multiple_scalar)

            ln_s = np.log(theta) + utils.normal_lpdf(xi, mu_single, sigma_single)
            ln_m = np.log(1 - theta) + utils.lognormal_lpdf(xi, mu_multiple, sigma_multiple)

            # Add sigmoid
            sigmoid_weight = (1.0/sigma_single) * np.log((2 * np.pi * sigma_single)**0.5 * np.exp(0.5 * M**2) - 1)
            sigmoid = 1/(1 + np.exp(-sigmoid_weight * (xi - mu_single)))

            ln_m = np.log(np.exp(ln_m) * sigmoid)

            def plot_it():
                fig, axes = plt.subplots(2, sharex=True)
                axes[0].plot(xi, ln_s, c="tab:blue")
                axes[0].plot(xi, ln_m, c="tab:red")

                axes[1].plot(xi, np.exp(ln_s), c="tab:blue")
                axes[1].plot(xi, np.exp(ln_m), c="tab:red")
                return fig


            # Check support at both ends.
            try:
                j = xi.searchsorted(mu_single)
                if not np.all(ln_s[:j] > ln_m[:j]):
                    support_flag[i] += 1

            except:
                support_flag[i] += 1

                if debug:
                    plot_it()
                    raise

            try:
                k = j + np.where(ln_m[j:] > ln_s[j:])[0][0]
                if not np.all(ln_m[k:] > ln_s[k:]):
                    support_flag[i] += 2

            except (IndexError, ):
                # This means we are probably not modelling large enough x values such that we don't 
                # find the turnover where ln_m > ln_s. Not necessarily an exception we should raise.
                # In this scenario it is likely that support_flag += 4 will occur as well, as the
                # distance sigma is probably > 5.
                support_flag[i] += 2

            except:
                if debug:
                    plot_it()
                    raise


            # Where is the turnover point?
            try:
                distance_sigma = (xi[k] - mu_single)/sigma_single

            except:
                support_flag[i] += 4

            else:
                if distance_sigma > 5:
                    support_flag[i] += 4

            """
            if sigma_multiple == sigma_multiples[-1]:
                if np.random.uniform() < 0.005:
                    fig = plot_it()
                    fig.axes[0].set_title(f"index={i}; theta: {theta:.1f}; mu_single = {mu_single:.1f}; sigma_single = {sigma_single:.1f}; sigma_multiple = {sigma_multiple:.1f}", fontsize=8)

            if support_flag[i] & 1:
                fig = plot_it()
                fig.axes[0].set_title(f"{model_name} index:{i}; flag:{support_flag[i]}; theta:{theta:.1f}; mu_single:{mu_single:.1f}; sigma_single:{sigma_single:.1f}; sigma_multiple:{sigma_multiple:.1f}", fontsize=8)
            """
            """
            if support_flag[i] & 2 and model_name == "rv":
                fig = plot_it()
                fig.axes[0].set_title(f"{model_name} index:{i}; flag:{support_flag[i]}; theta:{theta:.1f}; mu_single:{mu_single:.1f}; sigma_single:{sigma_single:.1f}; sigma_multiple:{sigma_multiple:.1f}", fontsize=8)
                raise a
            """


    def bin_flags(x_idx, y_idx, flags, x_label, y_label):
        bins = tuple([N_grid.get(label, N_grid_default) for label in (x_label, y_label)])
        return binned_statistic_dd(grid[:, [x_idx, y_idx]], flags, 
                                   statistic="sum", bins=bins)


    def show_support_regions(grid, flags, parameter_names=None, cmap="viridis", fig=None, debug=False, **kwargs):

        if parameter_names is None:
            parameter_names = ("theta", "mu_single", "sigma_single", "sigma_multiple")
        K = len(parameter_names) - 1

        # Do a first-pass to get (vmin, vmax)
        vmin, vmax = (+np.inf, -np.inf)
        for x_idx, x_label in enumerate(parameter_names):
            for y_idx, y_label in enumerate(parameter_names):
                if y_idx <= x_idx: continue

                H, *_ = bin_flags(x_idx, y_idx, flags, x_label, y_label)
                vmin = np.nanmin([vmin, H.min()])
                vmax = np.nanmax([vmin, H.max()])


        if fig is None:
            scale = 1.5
            fig, axes = plt.subplots(K, K, figsize=(9.5/scale, 7/scale))
        else:
            axes = fig.axes

        for x_idx, x_label in enumerate(parameter_names):
            for y_idx, y_label in enumerate(parameter_names):
                if y_idx <= x_idx: 
                    continue

                try:
                    ax = axes[y_idx - 1, x_idx]

                except:
                    continue

                H, bin_edges, bin_number = bin_flags(x_idx, y_idx, flags, x_label, y_label)

                image = ax.imshow(H.T,
                                  extent=(grid[:, x_idx].min(), grid[:, x_idx].max(),
                                          grid[:, y_idx].max(), grid[:, y_idx].min()),
                                  aspect="auto", interpolation="nearest",
                                  vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_xlim(np.sort(ax.get_xlim()))
                ax.set_ylim(np.sort(ax.get_ylim()))

                ax.xaxis.set_major_locator(MaxNLocator(6))
                ax.yaxis.set_major_locator(MaxNLocator(6))


        if not debug:
            for ax in np.array(axes).flatten():
                if not ax.get_xlabel():
                    ax.set_visible(False)

                if not ax.is_last_row():
                    ax.set_xticks([])
                    ax.set_xlabel(None)

                if not ax.is_first_col():
                    ax.set_ylabel(None)
                    ax.set_yticks([])

        fig.tight_layout()

        return fig


    path = lambda model_name, suffix: f"check_npm_priors_{model_name}_{suffix}.png"
    fig_all = show_support_regions(grid, support_flag)
    fig_all.suptitle(f"{model_name}")
    fig_all.savefig(path(model_name, "all"), dpi=150)

    fig_low_support = show_support_regions(grid, support_flag & 1, cmap="Blues")
    fig_low_support.suptitle(f"{model_name} model support at low values")
    fig_low_support.savefig(path(model_name, "low_support"), dpi=150)

    fig_high_support = show_support_regions(grid, support_flag & 2, cmap="Greens")
    fig_high_support.suptitle(f"{model_name} model support at high values")
    fig_high_support.savefig(path(model_name, "high_support"), dpi=150)

print("Created numerous figures matching 'check_npm_priors_{model_name}_{description}.png'")