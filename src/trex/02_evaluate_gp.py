
"""
Evaluate the GP predictions.
"""

import yaml
import h5py as h5

import os
import numpy as np
import logging
from scipy import optimize as op
from hashlib import md5
import sys
from tqdm import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt

from time import time
import george

import mpl_utils as mpl

logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    logger.propagate = False

logger.setLevel(logging.INFO)


if __name__ == "__main__":

    if sys.argv[1].lower().endswith(".yml") or sys.argv[1].lower().endswith(".yaml"):

        config_path = sys.argv[1]

        with open(config_path, "r") as fp:
            config = yaml.load(fp, Loader=yaml.Loader)

        pwd = os.path.dirname(config_path)

        random_seed = int(config["random_seed"])
        np.random.seed(random_seed)

        logger.info(f"Config path: {config_path} with seed {random_seed}")

        # Generate a unique hash.
        config_copy = deepcopy(config)
        for k in config_copy.pop("ignore_keywords_when_creating_hash", []):
            if k in config_copy:
                del config_copy[k]

            else:
                if "/" in k:
                    k1, k2, k3 = k.split("/")
                    del config_copy[k1][k2][k3]


        unique_hash = md5((f"{config_copy}").encode("utf-8")).hexdigest()[:5]
        logger.info(f"Unique hash: {unique_hash}")

        # Check results path now so we don't die later.
        results_path = os.path.join(pwd, config["results_path"].format(unique_hash=unique_hash))

    else:
        # Load in the results path.
        results_path = sys.argv[1]

    results_dir = os.path.dirname(results_path)

    results = h5.File(results_path, "a")

    config = yaml.load(results.attrs["config"], Loader=yaml.Loader)

    random_seed = int(config["random_seed"])
    np.random.seed(random_seed)

    # Load data.
    pwd = os.path.dirname(results.attrs["config_path"]).decode("utf-8")
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")
    sources = data["sources"]
    
    # TODO: store in config
    batch_size = 1000

    '''
    # Specify the subset of sources where we will evaluate the GP.
    mask = np.ones(len(sources["source_id"]), dtype=bool)

    for model_name, model_config in config["models"].items():

        predictor_label_name = model_config["predictor_label_name"]
        mask *= np.isfinite(sources[predictor_label_name][()])
        for ln in model_config["kdtree_label_names"]:
            mask *= np.isfinite(sources[ln][()])

    # Specify a random subset.
    source_indices = np.where(mask)[0]
    #source_indices = np.random.choice(np.where(mask)[0], 100_000, replace=False)
    '''

    # Common subset of source ids to run on.
    N = len(sources["source_id"])
    science_source_ids = np.loadtxt("../../data/catalogs/source_ids.txt", dtype=">i8")
    is_finite_mask = np.ones(N, dtype=bool)
    all_label_names = []
    for model_name, model_config in config["models"].items():
        all_label_names.extend([model_config["predictor_label_name"]] \
                            + list(model_config["kdtree_label_names"]))

        data_bounds = model_config.get("data_bounds", None)
        if data_bounds:
            for parameter_name, (upper, lower) in data_bounds.items():
                lower, upper = np.sort([lower, upper])
                is_finite_mask *= (upper >= sources[parameter_name][()]) \
                                 * (sources[parameter_name][()] >= lower)

    for ln in np.unique(all_label_names):
        is_finite_mask *= np.isfinite(sources[ln][()])

    # Run on a random 100,000 in  addition to the science source ids
    subset_size = config.get("subset_size", np.sum(is_finite_mask))
    subset_indices = np.random.choice(np.where(is_finite_mask)[0], subset_size, replace=False)
    is_science_source = np.in1d(sources["source_id"][()], science_source_ids)

    joint_data_mask = np.zeros(N, dtype=bool)
    joint_data_mask[is_science_source] = True
    #joint_data_mask *= is_finite_mask # only allow science targets with finite values
    joint_data_mask[subset_indices] = True # add in the random subset indices.

    print(f"WARNING: Using Joint data mask and only evaluating on {np.sum(joint_data_mask)} sources")

    source_indices = np.where(joint_data_mask)[0]

    for model_name, model_config in config["models"].items():

        lns = list(model_config["kdtree_label_names"]) 
        parameter_names = list(results[f"models/{model_name}/gp_model"].keys())

        logger.info(f"Model name {model_name} has GPs for parameters: {parameter_names}")

        group = results[f"models/{model_name}"]
        if "gp_predictions" in group:
            del group["gp_predictions"]
        
        gp_predictions_group = group.create_group("gp_predictions", track_order=True)
        gp_predictions_group.create_dataset("source_id", data=sources["source_id"][()][source_indices])
        gp_predictions_group.create_dataset("source_indices", data=source_indices)


        # Do it for all things that meet the data mask (e.g., finite)    
        xp = np.vstack([sources[ln][()] for ln in lns]).T[source_indices]


        for parameter_name in parameter_names:
            logger.info(f"Running GP predictions for {model_name}: {parameter_name}")

            # Construct the kernel.
            model_info = group[f"gp_model/{parameter_name}"]

            X = model_info["X"][()]
            Y = model_info["Y"][()]

            print(parameter_name, np.min(Y), np.max(Y), np.percentile(Y, [5, 16, 50, 84, 95]))

            kernel_class = getattr(george.kernels, model_info.attrs["kernel"])
            metric = np.var(X, axis=0)
            kernel = kernel_class(metric=metric, ndim=metric.size)

            gp = george.GP(kernel, 
                           mean=np.mean(Y), fit_mean=True,
                           white_noise=np.log(np.std(Y)), fit_white_noise=True)
            for p in gp.parameter_names:
                gp.set_parameter(p, model_info.attrs[p])

            gp.compute(X)

            N = xp.shape[0]

            logger.info(f"Making predictions for {N} sources")

            num_batches = int(np.ceil(N / batch_size))

            # Set up the data set.
            if parameter_name in gp_predictions_group:
                del gp_predictions_group[parameter_name]

            d = gp_predictions_group.create_dataset(parameter_name, data=np.nan * np.ones((N, 2)))

            with tqdm(total=N) as pbar:
                for i in range(num_batches):
                    si, ei = i * batch_size, (i + 1) * batch_size

                    finite_in_batch = np.all(np.isfinite(xp[si:ei]), axis=1)
                    xc = np.copy(xp[si:ei])
                    xc[~np.isfinite(xc)] = 1.0

                    d[si:ei, :] = np.vstack(gp.predict(Y, xc, return_cov=False, return_var=True)).T
                    d[si:ei][~finite_in_batch] = np.nan
                    pbar.update(batch_size)

        # Ensure predictions are valid.
        #results[f"{model_name}/gp_predictions/theta"][()][:, 0] = np.clip(results[f"{model_name}/gp_predictions/theta"][()][:, 0], 0, 1)


    # Close file.
    results.close()

    # Plot predictions.
    results = h5.File(results_path, "r")


    for model_name in config["models"].keys():

        source_indices = results[f"models/{model_name}/gp_predictions/source_indices"][()]

        bp_rp = sources["bp_rp"][()][source_indices]
        absolute_g_mag = sources["absolute_g_mag"][()][source_indices]
        phot_g_mean_mag = sources["phot_g_mean_mag"][()][source_indices]

        kwds = dict(bins=100, function="mean", interpolation="none", full_output=True)

        for parameter_name in ("theta", "mu_single", "sigma_single", "sigma_multiple"):

            p, p_var = results[f"models/{model_name}/gp_predictions/{parameter_name}"][()].T

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            _, im = mpl.plot_binned_statistic(bp_rp, absolute_g_mag, p, ax=axes[0], **kwds)
            mpl.plot_binned_statistic(bp_rp, phot_g_mean_mag, p, ax=axes[1], **kwds)

            axes[0].set_ylabel('absolute g mag')
            axes[1].set_ylabel('phot g mean mag')

            for ax in axes:    
                ax.set_xlabel('bp - rp')

            cbar = plt.colorbar(im)
            cbar.set_label(f"{model_name}:{parameter_name}")

            fig.tight_layout()

            # Save figures.
            fig.savefig(os.path.join(results_dir, f"gp-evaluations-{model_name}-{parameter_name}.png"), dpi=300)


            fig, axes = plt.subplots(1, 2, figsize=(8, 4))

            _, im = mpl.plot_binned_statistic(bp_rp, absolute_g_mag, p_var**0.5, ax=axes[0], **kwds)
            mpl.plot_binned_statistic(bp_rp, phot_g_mean_mag, p_var**0.5, ax=axes[1], **kwds)

            axes[0].set_ylabel('absolute g mag')
            axes[1].set_ylabel('phot g mean mag')

            for ax in axes:    
                ax.set_xlabel('bp - rp')

            cbar = plt.colorbar(im)
            cbar.set_label(f"uncertainty({model_name}:{parameter_name})")

            fig.tight_layout()

            # Save figures.
            fig.savefig(os.path.join(results_dir, f"gp-evaluations-{model_name}-{parameter_name}-uncert.png"), dpi=300)


    
