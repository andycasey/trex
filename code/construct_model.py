

"""
Self-calibrate the radial and astrometric jitter in Gaia.

This script takes in a config file and fits a subset of the jitter using a normal-lognormal mixture
model and saves the output of the results. Those outputs are then to be used in a gaussian process.
"""

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
from shutil import copyfile
from glob import glob

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import corner

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
    config_copy = config.copy()
    for k in config_copy.pop("ignore_keywords_when_creating_hash", []):
        if k in config_copy: 
            del config_copy[k]

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


    # Get a list of all relevant label names
    all_label_names = []
    for model_name, model_config in config["models"].items():
        all_label_names.append(model_config["predictor_label_name"])
        all_label_names.extend(model_config["kdtree_label_names"])

    all_label_names = list(np.unique(all_label_names))     

    # Require finite entries for all predictors across all models.
    sources = data["sources"]
    M = config["number_of_sources_for_gaussian_process"]
    N = sources["source_id"].size

    data_mask = np.ones(N, dtype=bool)
        
    data_bounds = config.get("data_bounds", None)
    if data_bounds:
        for parameter_name, (upper, lower) in data_bounds.items():
            lower, upper = np.sort([lower, upper])
            data_mask *= (upper >= sources[parameter_name][()]) \
                       * (sources[parameter_name][()] >= lower)

            logger.info(f"Restricting to sources with {parameter_name}: [{lower:.1f}, {upper:.1f}]")



    # Check for finite values.
    for ln in all_label_names:
        data_mask *= np.isfinite(sources[ln][()])
        
    data_indices = np.where(data_mask)[0]
    npm_indices = np.random.choice(data_indices.size, M, replace=False)
    
    # Create results file.
    with h5.File(results_path, "w") as results:

        # Add config.
        results.attrs.create("config", np.string_(yaml.dump(config)))
        results.attrs.create("config_path", np.string_(config_path))

        # Add indices.
        g = results.create_group("indices")
        g.create_dataset("data_indices", data=data_indices)
        g.create_dataset("npm_indices", data=npm_indices)


    # Load model and check optimization keywords
    model_path = os.path.join(pwd, config["model_path"])
    model = stan.load_stan_model(model_path, verbose=False)

    # Make sure that some entries have the right type.
    default_opt_kwds = config.get("optimisation_kwds", {})
    for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if key in default_opt_kwds:
            default_opt_kwds[key] = float(default_opt_kwds[key])

    logger.info(f"Optimization keywords:\n{utils.repr_dict(default_opt_kwds)}")

    default_bounds = dict(theta=[0.5, 1],
                          mu_single=[0.5, 15],
                          sigma_single=[0.05, 10],
                          sigma_multiple=[0.2, 1.6])


    # Plotting
    plot_mixture_model_figures = config.get("plot_mixture_model_figures", False)
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Sampling.
    sampling = config.get("sample_mixture_model", False)

    if plot_mixture_model_figures:
        if config["multiprocessing"]:
            raise ValueError("you want to plot figures and do multiprocessing?")

        global fig, axes
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))

        logger.info("Set globals")


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

        Z = np.vstack([sources[ln][()] for ln in lns]).T
        X, Y = Z[data_mask, :-1], Z[data_mask, -1]
        S = sources["source_id"][()][data_mask]

        N, D = X.shape


        logger.info(f"Building K-D tree with N = {N}, D = {D}...")
        kdt, scales, offsets = npm.build_kdtree(X, 
                relative_scales=model_config["kdtree_relative_scales"])

        kdt_kwds = dict(offsets=offsets, scales=scales, full_output=True)
        kdt_kwds.update(
            minimum_radius=model_config["kdtree_minimum_radius"],
            maximum_radius=model_config["kdtree_maximum_radius"],
            minimum_points=model_config["kdtree_minimum_points"],
            maximum_points=model_config["kdtree_maximum_points"])

        # Optimize the non-parametric model for those sources.
        npm_results = np.zeros((M, 5))
        done = np.zeros(M, dtype=bool)

        if plot_mixture_model_figures:

            kwds = dict(function="count", bins=250, cmap="Greys", norm=LogNorm())
            mpl.plot_binned_statistic(X[:, 0], X[:, 1], X[:, 1], ax=axes[0], **kwds)
            mpl.plot_binned_statistic(X[:, 0], X[:, 2], X[:, 2], ax=axes[1], **kwds)

            for ax in axes[:2]:
                ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

        mu_multiple_scalar = model_config["mu_multiple_scalar"]

        # TODO: put scalar to the config file.
        def optimize_mixture_model(index, inits=None, debug=False):

            suppress = config.get("suppress_stan_output", True)

            # Select indices and get data.
            d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

            y = Y[nearby_idx]
            ball = X[nearby_idx]

            if inits is None:
                inits = npm._get_1d_initialisation_point(
                    y, scalar=mu_multiple_scalar, bounds=bounds)

            # Update meta dictionary with things about the data.
            meta = dict(max_log_y=np.log(np.max(y)),
                        N=nearby_idx.size,
                        y_percentiles=np.percentile(y, [16, 50, 84]),
                        ball_ptps=np.ptp(ball, axis=0),
                        ball_medians=np.median(ball, axis=0),
                        init_points=inits,
                        kdt_indices=nearby_idx)

            data_dict = dict(y=y,
                             N=y.size,
                             scalar=mu_multiple_scalar)
            data_dict.update(stan_bounds)

            p_opts = []
            ln_probs = []
            for j, init_dict in enumerate(inits):

                opt_kwds = dict(
                    init=init_dict,
                    data=data_dict,
                    as_vector=False)
                opt_kwds.update(default_opt_kwds)

                # Do optimization.
                # TODO: Suppressing output is always dangerous.
                with stan.suppress_output(suppress) as sm:
                    try:
                        p_opt = model.optimizing(**opt_kwds)

                    except:
                        logger.exception(f"Exception occurred when optimizing index {index}"\
                                          f" from {init_dict}:")
                    else:
                        if p_opt is not None:
                            p_opts.append(p_opt["par"])
                            
                            ln_probs.append(utils.ln_prob(y, 1, *utils._pack_params(**p_opt["par"]), bounds=bounds))

                            assert abs(ln_probs[-1] - p_opt["value"]) < 1e-8

                        
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

                """
                # Calculate uncertainties.
                op_bounds = ()
                def nlp(p):
                    w, mu_s, sigma_s, sigma_m = p
                    mu_m = np.log(mu_s + mu_multiple_scalar * sigma_s) + sigma_m**2

                    if not (bounds["theta"][1] >= w >= bounds["theta"][0]) \
                    or not (bounds["mu_single"][1] >= mu_s >= bounds["mu_single"][0]) \
                    or not (bounds["sigma_multiple"][1] >= sigma_m >= bounds["sigma_multiple"][0]):
                        return np.inf

                    return -utils.ln_likelihood(y, w, mu_s, sigma_s, mu_m, sigma_m)


                op_bounds = [bounds["theta"],
                             bounds["mu_single"],
                             bounds["sigma_single"],
                             bounds["sigma_multiple"],
                ]

                #x0 = utils._pack_params(**p_opt)
                x0 = (p_opt["theta"], p_opt["mu_single"], p_opt["sigma_single"], p_opt["sigma_multiple"])
                p_opt2 = op.minimize(nlp, x0, bounds=op_bounds, method="L-BFGS-B")
                """


                # Create a three-panel figure showing:

                # (1) a log-density of the HRD + the selected ball points
                # (2) a log-density of colour vs apparent magnitude + the selected ball points
                # (3) the jitter + fitted parameters 

                if sampling:

                    chains = 2 # TODO: move to config file.
                    sampling_kwds = dict(data=opt_kwds["data"], init=[p_opt] * chains, chains=chains)
                    try:
                        samples = model.sampling(**sampling_kwds)

                    except:
                        None

                    else:
                        extracted = samples.extract()
                        chains = np.array([extracted[k] for k in samples.flatnames]).T

                        latex_labels = dict(theta=r"$w$",
                                            mu_single=r"$\mu_\mathrm{single}$",
                                            sigma_single=r"$\sigma_\mathrm{single}$",
                                            mu_multiple=r"$\mu_\mathrm{multiple}$",
                                            sigma_multiple=r"$\sigma_\mathrm{multiple}$")

                        corner_fig = corner.corner(chains, labels=[latex_labels[k] for k in samples.flatnames])

                        source_id = S[index]
                        figure_path = os.path.join(figures_dir, f"{model_name}-{source_id}-samples.png")
                        corner_fig.savefig(figure_path, dpi=150)

                        chains_path = os.path.join(figures_dir, f"{model_name}-{source_id}-chains.pkl")

                        dump = dict(names=samples.flatnames,
                                    chains=chains,
                                    y=y,
                                    ball=ball,
                                    X=X[index])

                        with open(chains_path, "wb") as fp:
                            pickle.dump(dump, fp)

                        plt.close("all")




                if plot_mixture_model_figures:

                    source_id = S[index]

                    figure_path = os.path.join(figures_dir, f"{model_name}-{source_id}.png")

                    
                    x_upper = 2 * config["models"][model_name]["bounds"]["mu_single"][1]
                    bins = np.linspace(0, x_upper, 51)

                    xi = np.linspace(0, x_upper, 1000)
                    y_s = utils.norm_pdf(xi, p_opt["mu_single"], p_opt["sigma_single"], p_opt["theta"])
                    y_m = utils.lognorm_pdf(xi, p_opt["mu_multiple"], p_opt["sigma_multiple"], p_opt["theta"])

                    items_for_deletion = [
                        axes[0].scatter(ball.T[0], ball.T[1], c="tab:blue", s=1, zorder=10, alpha=0.5),
                        axes[1].scatter(ball.T[0], ball.T[2], c="tab:blue", s=1, zorder=10, alpha=0.5),

                        axes[2].hist(y, bins=bins, facecolor="#cccccc", density=True, zorder=-1)[-1],
                        axes[2].axvline(Y[index], c="#666666"),

                        axes[2].plot(xi, y_s, c="tab:blue"),
                        axes[2].fill_between(xi, np.zeros_like(y_s), y_s, facecolor="tab:blue", alpha=0.25),

                        axes[2].plot(xi, y_m, c="tab:red"),
                        axes[2].fill_between(xi, np.zeros_like(y_m), y_m, facecolor="tab:red", alpha=0.25),
                    ]


                    # Ax limits.

                    axes[0].set_xlim(-0.5, 5)
                    axes[0].set_ylim(10, -15)

                    axes[1].set_xlim(-0.5, 5)
                    axes[1].set_ylim(15, 3)

                    axes[2].set_xlim(0, x_upper)
                    axes[2].set_yticks([])

                    fig.tight_layout()


                    fig.savefig(figure_path, dpi=150)


                    for item in items_for_deletion:
                        try:
                            item.set_visible(False)

                        except AttributeError:
                            for _ in item:
                                if hasattr(_, "set_visible"):
                                    _.set_visible(False)


                if debug:

                    # Create 
                    raise a

                return (index, p_opt, meta)


        def sp_swarm(*sp_indices, **kwargs):

            logger.info("Running single processor swarm")

            with tqdm.tqdm(sp_indices, total=len(sp_indices)) as pbar:

                for j, index in enumerate(sp_indices):
                    if done[j]: continue

                    _, result, meta = optimize_mixture_model(index, **kwargs)

                    pbar.update()

                    done[j] = True
                    
                    if result is not None:
                        npm_results[j] = utils._pack_params(**result)
                         
            return None



        def mp_swarm(*mp_indices, in_queue=None, out_queue=None, seed=None):

            np.random.seed(seed)

            swarm = True

            while swarm:

                try:
                    j, index = in_queue.get_nowait()

                except mp.queues.Empty:
                    logger.info("Queue is empty")
                    break

                except StopIteration:
                    logger.warning("Swarm is bored")
                    break

                except:
                    logger.exception("Unexpected exception:")
                    break

                else:
                    if index is None and init is False:
                        swarm = False
                        break

                    try:
                        _, result, meta = optimize_mixture_model(index)

                    except:
                        logger.exception(f"Exception when optimizing on {index}")
                        out_queue.put((j, index, None, dict()))
                    
                    else:
                        out_queue.put((j, index, result, meta))

            return None



        if not config.get("multiprocessing", False):
            sp_swarm(*npm_indices)

            raise a

        else:
            P = config.get("processes", mp.cpu_count())

            with mp.Pool(processes=P) as pool:

                manager = mp.Manager()

                in_queue = manager.Queue()
                out_queue = manager.Queue()

                swarm_kwds = dict(in_queue=in_queue,
                                  out_queue=out_queue)


                logger.info("Dumping everything into the queue!")
                for j, index in enumerate(npm_indices):
                    in_queue.put((j, index, ))

                j = []
                for _ in range(P):
                    j.append(pool.apply_async(mp_swarm, [], kwds=swarm_kwds))


                with tqdm.tqdm(total=M) as pbar:

                    while not np.all(done):

                        # Check for output.
                        try:
                            r = out_queue.get(timeout=30)

                        except mp.queues.Empty:
                            logger.info("No npm_results")
                            break

                        else:
                            j, index, result, meta = r

                            done[j] = True
                            if result is not None:
                                npm_results[j] = utils._pack_params(**result)

                            pbar.update(1)

        # Do not use bad results.

        # Bad results include:
        # - Things that are so clearly discrepant in every parameter.
        # - Things that are on the edge of the boundaries of parameter space.


        tol_sigma = model_config["tol_sum_sigma"]
        tol_proximity = model_config["tol_proximity"]

        parameter_names = (
            "theta", 
            "mu_single", "sigma_single", 
            "mu_multiple", "sigma_multiple")

        lower_bounds = np.array([model_config["bounds"].get(k, [-np.inf])[0] for k in parameter_names])
        upper_bounds = np.array([model_config["bounds"].get(k, [+np.inf])[-1] for k in parameter_names])

        for iteration in range(3): # MAGIC HACK

            sigma = np.abs(npm_results - np.median(npm_results, axis=0)) \
                  / np.std(npm_results, axis=0)
            sigma = np.sum(sigma, axis=1)
            
            # Only care about indices 1 and 2
            lower_bounds[3:] = -np.inf
            upper_bounds[3:] = +np.inf
            lower_bounds[0] = -np.inf
            upper_bounds[0] = +np.inf

            not_ok_bound = np.any(
                (np.abs(npm_results - lower_bounds) <= tol_proximity) \
              + (np.abs(npm_results - upper_bounds) <= tol_proximity), axis=1)

            not_ok_sigma = sigma > tol_sigma

            not_ok = not_ok_bound + not_ok_sigma
            not_ok = not_ok_sigma

            done[not_ok] = False
            sp_swarm(*npm_indices[not_ok], 
                     inits=[np.median(npm_results[~not_ok], axis=0), "random"],
                     debug=False)

            print(f"There were {sum(not_ok_sigma)} results discarded for being outliers")
            print(f"There were {sum(not_ok_bound)} results discarded for being close to the edge")
            print(f"There were {sum(not_ok)} results discarded in total")

        # Save results.
        with h5.File(results_path, "a") as results:

            # ast/mixture_model
            g = results.create_group(f"{model_name}/mixture_model")

            for i, parameter_name in enumerate(parameter_names):
                g.create_dataset(parameter_name, data=npm_results.T[i])

            g.create_dataset("is_ok", data=~not_ok)

        logger.info(f"Saved results of {model_name} model to {results_path}")

        # Make some figures
        import matplotlib.pyplot as plt
        for i in range(5):
            f, a = plt.subplots(1, 2)
            ax = a[0]
            ax.set_title(f"{model_name} {i}")

            kwds = dict(vmin=lower_bounds[i] if np.isfinite(lower_bounds[i]) else None,
                        vmax=upper_bounds[i] if np.isfinite(upper_bounds[i]) else None)
            kwds = dict()

            scat = ax.scatter(X[npm_indices, 0], X[npm_indices, 1], c=npm_results.T[i], s=1,
                **kwds)

            ax.scatter(X[npm_indices, 0][not_ok], X[npm_indices, 1][not_ok], s=10, facecolor="none", edgecolor="k", zorder=-1,
                **kwds)

            a[1].scatter(X[npm_indices, 0], X[npm_indices, 2], c=npm_results.T[i], s=1, **kwds)
            a[1].scatter(X[npm_indices, 0][not_ok], X[npm_indices, 2][not_ok], s=10, facecolor="none", edgecolor="k", zorder=-1,
                **kwds)

            for ax in a:
                ax.set_ylim(ax.get_ylim()[::-1])
            cbar = plt.colorbar(scat)

