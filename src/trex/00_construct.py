

""" Self-calibrate the radial and astrometric jitter in Gaia. """

import h5py as h5
import logging
import multiprocessing as mp
import numpy as np
import os
import pickle
import sys
import tqdm
import yaml
from hashlib import md5
from time import (sleep, time)
from astropy.io import fits
from scipy import optimize as op
from scipy.special import logsumexp
from scipy import stats# import binned_statistic_dd
from shutil import copyfile
from glob import glob
from copy import deepcopy

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

import george

import npm_utils as npm
import mpl_utils as mpl
import stan_utils as stan
import utils


logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    logger.propagate = False



if __name__ == "__main__":


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
    results_dir = os.path.dirname(os.path.realpath(results_path))
    os.makedirs(results_dir, exist_ok=True)

    # Save a copy of the config file to the results path.
    copyfile(config_path, os.path.join(results_dir, os.path.basename(config_path)))
    logger.info(f"Copied config file to {results_dir}/")


    # Load data.
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")

    # Require finite entries for all predictors across all models.
    sources = data["sources"]
    M = config["number_of_sources_for_gaussian_process"]
    N = sources["source_id"].size

    # Create results file.
    with h5.File(results_path, "w") as results:

        # Add config.
        results.attrs.create("config", np.string_(yaml.dump(config)))
        results.attrs.create("config_path", np.string_(config_path))

    # Load model and check optimization keywords

    """
    model_path = os.path.join(pwd, config["model_path"])
    model = stan.load_stan_model(model_path, verbose=False)
    """

    # Make sure that some entries have the right type.
    default_opt_kwds = config.get("optimisation_kwds", {})
    for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if key in default_opt_kwds:
            default_opt_kwds[key] = float(default_opt_kwds[key])


    logger.info(f"Optimization keywords:\n{utils.repr_dict(default_opt_kwds)}")

    default_bounds = dict()
    """
    theta=[0.5, 1],
                          mu_single=[0.5, 15],
                          sigma_single=[0.05, 10],
                          sigma_multiple=[0.2, 1.6])
    """

    # Plotting
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Sampling.
    model_path = os.path.join(pwd, config["model_path"])
    model = stan.load_stan_model(model_path, verbose=False)

    model_results = dict()
    for model_name, model_config in config["models"].items():
        if model_name in model_results: continue

        logger.info(f"Running model '{model_name}' with config:\n{utils.repr_dict(model_config)}")

        bounds = default_bounds.copy()
        bounds.update(model_config["bounds"])

        stan_bounds = dict()
        for k, (lower, upper) in bounds.items():
            stan_bounds[f"bound_{k}"] = [lower, upper]

        # Set up a KD-tree.
        lns = list(model_config["kdtree_label_names"]) + [model_config["predictor_label_name"]]

        # Create data mask and save it.
        data_mask = np.ones(N, dtype=bool)
        all_label_names = [model_config["predictor_label_name"]] \
                        + list(model_config["kdtree_label_names"])
        for ln in np.unique(all_label_names):
            data_mask *= np.isfinite(sources[ln][()])


        data_bounds = model_config.get("data_bounds", None)
        if data_bounds:
            for parameter_name, (upper, lower) in data_bounds.items():
                lower, upper = np.sort([lower, upper])
                data_mask *= (upper >= sources[parameter_name][()]) \
                           * (sources[parameter_name][()] >= lower)

                logger.info(f"Restricting to sources with {parameter_name}: [{lower:.1f}, {upper:.1f}]")

        data_indices = np.where(data_mask)[0]


        Z = np.vstack([sources[ln][()] for ln in lns]).T
        X, Y = Z[data_mask, :-1], Z[data_mask, -1]
        S = sources["source_id"][()][data_mask]

        coreset_method = model_config.get("coreset_method", "random")
        logger.info(f"Generating coreset using {coreset_method} method")


        if coreset_method == "random":
            npm_indices = np.random.choice(data_indices.size, M, replace=False)

        elif coreset_method == "grid":

            num_bins = int(np.ceil(M**(1.0/X.shape[1])))

            M = num_bins**X.shape[1]

            bins = np.percentile(X, np.linspace(0, 100, 1 + num_bins), axis=0).T

            # Select random from each permutation?
            H, _, binnumber = stats.binned_statistic_dd(X, 1, statistic="count", bins=bins)

            npm_indices = np.zeros(M, dtype=int)

            for i, bn in enumerate(tqdm.tqdm(set(binnumber))):
                npm_indices[i] = np.random.choice(np.where(binnumber == bn)[0], size=1)

        else:
            raise NotImplementedError("unrecognised coreset method")


        for i, label_name in enumerate(all_label_names[1:]):
            v = X[npm_indices, i]
            print(f"{label_name} min: {np.min(v)}  max: {np.max(v)}")


        
        logger.info(f"Building K-D tree with N = {X.shape[0]}, D = {X.shape[1]}...")
        kdt, scales, offsets = npm.build_kdtree(X,
                relative_scales=model_config["kdtree_relative_scales"])

        kdt_kwds = dict(offsets=offsets, scales=scales, full_output=True)
        kdt_kwds.update(minimum_radius=model_config["kdtree_minimum_radius"],
                        maximum_radius=model_config["kdtree_maximum_radius"],
                        minimum_points=model_config["kdtree_minimum_points"],
                        maximum_points=model_config["kdtree_maximum_points"])

        # Optimize the non-parametric model for those sources.
        npm_results = np.zeros((M, 5))
        done = np.zeros(M, dtype=bool)


        mu_multiple_scalar = config.get("mu_multiple_scalar", 3)

        def normal_lpdf(y, mu, sigma):
            ivar = sigma**(-2)
            return 0.5 * (np.log(ivar) - np.log(2 * np.pi) - (y - mu)**2 * ivar)

        def lognormal_lpdf(y, mu, sigma):
            ivar = sigma**(-2)
            return - 0.5 * np.log(2 * np.pi) - np.log(y * sigma) \
                   - 0.5 * (np.log(y) - mu)**2 * ivar

        def plot_pdf(y, w, mu_single, sigma_single, sigma_multiple):

            x = np.linspace(0, 10, 100)

            mu_multiple = np.log(mu_single + mu_multiple_scalar * sigma_single) + sigma_multiple**2

            log_ps = np.log(w) + normal_lpdf(x, mu_single, sigma_single)
            log_pm = np.log(1 - w) + lognormal_lpdf(x, mu_multiple, sigma_multiple)

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].plot(x, np.exp(log_ps))
            axes[0].plot(x, np.exp(log_pm))
            axes[0].hist(y, bins=30, facecolor="#666666", alpha=0.5, normed=True)

            axes[1].plot(x, log_ps)
            axes[1].plot(x, log_pm)

            return fig



        def optimize_mixture_model(index, inits=None, debug=False):

            # Select indices and get data.
            d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

            y = Y[nearby_idx]
            ball = X[nearby_idx]

            # Restrict/
            max_y = config.get("max_y", None)
            if max_y is not None:
                keep = (y < max_y)
                y, ball = y[keep], ball[keep]


            #if inits is None:
            #    inits = npm._get_1d_initialisation_point(
            #        y, scalar=mu_multiple_scalar, bounds=bounds)

            # Update meta dictionary with things about the data.
            meta = dict(N=nearby_idx.size,
                        y_percentiles=np.percentile(y, [16, 50, 84]),
                        ball_ptps=np.ptp(ball, axis=0),
                        ball_medians=np.median(ball, axis=0),
                        init_points=inits,
                        kdt_indices=nearby_idx)

            data_dict = dict(y=y, N=y.size, mu_multiple_scalar=mu_multiple_scalar)
            data_dict.update(stan_bounds)

            p_opts = []
            ln_probs = []
            for j, init_dict in enumerate(inits):

                opt_kwds = dict(init=init_dict, data=data_dict, as_vector=False)
                opt_kwds.update(default_opt_kwds)

                with stan.suppress_output(suppress_output=(not debug)) as sm:

                    try:
                        p_opt = model.optimizing(**opt_kwds)

                    except:
                        logger.exception(f"Exception occurred when optimizing index {index}"\
                                          f" from {init_dict}:")
                    else:
                        if p_opt is not None:
                            p_opts.append(p_opt["par"])
                            ln_probs.append(p_opt["value"])

                            w, mu_s, sigma_s, sigma_m = p_opt["par"]["theta"], p_opt["par"]["mu_single"], p_opt["par"]["sigma_single"], p_opt["par"]["sigma_multiple"],

                            fig = plot_pdf(y, w, mu_s, sigma_s, sigma_m)

                            raise a

                try:
                    p_opt

                except UnboundLocalError:
                    logger.warning("Stan failed. STDOUT & STDERR:")
                    logger.warning("\n".join(sm.outputs))

                else:
                    if p_opt is None:
                        stdout, stderr = sm.outputs
                        logger.warning("Stan only returned p_opt = None")
                        logger.warning(f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}")

            if len(p_opts) < 1:
                logger.warning(f"Optimization on index {index} did not converge"\
                                "from any initial point trialled. Consider "\
                                "relaxing the optimization tolerances! If this "\
                                "occurs regularly then something is very wrong!")

                return (index, None, meta)

            else:
                # evaluate best.
                idx = np.argmax(ln_probs)
                p_opt = p_opts[idx]
                meta["init_idx"] = idx

                return (index, p_opt, meta)


        index = 5451123
        d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

        y = Y[nearby_idx]
        ball = X[nearby_idx]


        keep = (y < 20)
        y, ball = y[keep], ball[keep]

        inits = [dict(theta=0.9, mu_single=np.median(y), sigma_single=0.1, sigma_multiple=0.25)]
        foo = optimize_mixture_model(index, inits=inits)

        raise a


        raise a

