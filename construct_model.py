
""" Self-calibrate the radial and astrometric jitter in Gaia astrometry. """

import h5py as h5
import logging
import multiprocessing as mp
import numpy as np
import os
import pickle
import sys
import tqdm
import yaml
from time import (sleep, time)
from astropy.io import fits
from scipy import optimize as op
from scipy.special import logsumexp

import george

import npm_utils as npm
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

    random_seed = int(config["random_seed"])
    np.random.seed(random_seed)

    logger.info(f"Config path: {config_path} with seed {random_seed}")

    # Check results path now so we don't die later.
    results_path = config["results_path"]
    directory_path = os.path.dirname(os.path.realpath(results_path))
    os.makedirs(directory_path, exist_ok=True)


    # Load data.
    data = fits.open(config["data_path"])[1].data

    # Get a list of all relevant label names
    all_label_names = []
    for model_name, model_config in config["models"].items():
        all_label_names.append(model_config["predictor_label_name"])
        all_label_names.extend(model_config["kdtree_label_names"])

    all_label_names = list(np.unique(all_label_names))     

    if config.get("check_finite", True) \
    and not np.all([np.isfinite(data[ln]) for ln in all_label_names]):
        raise ValueError("all predictor label names must be finite")

    model = stan.load_stan_model(config["model_path"], verbose=False)

    # Make sure that some entries have the right type.
    default_opt_kwds = config.get("optimisation_kwds", {})
    for key in ("tol_obj", "tol_grad", "tol_rel_grad", "tol_rel_obj"):
        if key in default_opt_kwds:
            default_opt_kwds[key] = float(default_opt_kwds[key])

    logger.info(f"Optimization keywords:\n{utils.repr_dict(default_opt_kwds)}")

    default_bounds = dict(bound_theta=[0.5, 1],
                          bound_mu_single=[0.5, 15],
                          bound_sigma_single=[0.05, 10],
                          bound_sigma_multiple=[0.2, 1.6])

    N = len(data)
    M = config["number_of_gaussian_process_sources"]
    indices = np.random.choice(N, M, replace=False)

    S = config.get("number_of_science_sources", -1)
    if S < 0:
        S = N
        science_indices = np.arange(N)

    else:
        science_indices = np.random.choice(N, S, replace=False)

    model_results = dict()
    for model_name, model_config in config["models"].items():
        if model_name in model_results or model_name == "rv_uncorrected": continue

        logger.info(f"Running model '{model_name}' with config:\n{utils.repr_dict(model_config)}")

        bounds = default_bounds.copy()
        for k, (lower, upper) in model_config["bounds"].items():
            bounds[f"bound_{k}"] = [lower, upper]

        # Set up a KD-tree.
        dtype = '<f4'
        lns = list(model_config["kdtree_label_names"]) \
            + [model_config["predictor_label_name"]]

        Z = np.array(data.view(np.recarray)[lns]\
                         .astype([(ln, dtype) for ln in lns])\
                         .view(dtype).reshape((-1, len(lns))))
        X, Y = Z[:, :-1], Z[:, -1]

        N, D = X.shape

        logger.info(f"Building K-D tree...")
        kdt, scales, offsets = npm.build_kdtree(X, 
                relative_scales=model_config["kdtree_relative_scales"])

        kdt_kwds = dict(offsets=offsets, scales=scales, full_output=True)
        kdt_kwds.update(
            minimum_radius=model_config["kdtree_minimum_radius"],
            maximum_radius=model_config.get("kdtree_maximum_radius", None),
            minimum_points=model_config["kdtree_minimum_points"],
            maximum_points=model_config["kdtree_maximum_points"],
            minimum_density=model_config.get("kdtree_minimum_density", None))

        # Optimize the non-parametric model for those sources.
        results = np.zeros((M, 5))
        done = np.zeros(M, dtype=bool)

        def optimize_mixture_model(index, inits=None, scalar=5, debug=False):

            suppress = config.get("suppress_stan_output", True)
            #print(debug, suppress)
            #if debug:
            #    suppress = False

            # Select indices and get data.
            d, nearby_idx, meta = npm.query_around_point(kdt, X[index], **kdt_kwds)

            y = Y[nearby_idx]
            ball = X[nearby_idx]

            if inits is None:
                inits = npm._get_1d_initialisation_point(
                    y, scalar=scalar, bounds=model_config["bounds"])

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
                             scalar=scalar)
            data_dict.update(bounds)

            p_opts = []
            ln_probs = []
            for j, init_dict in enumerate(inits):

                opt_kwds = dict(
                    init=init_dict,
                    data=data_dict)
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
                            p_opts.append(p_opt)
                            ln_probs.append(utils.ln_prob(y, 1, *utils._pack_params(**p_opt)))

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

                if debug:

                    theta, mu_single, sigma_single, mu_multiple, sigma_multiple = np.hstack(p_opt.values())

                    fig, ax = plt.subplots()
                    ax.hist(y, bins=50)
                    ax.axvline(Y[index])

                    xi = np.linspace(0, 20, 1000)

                    y_s = len(y) * utils.norm_pdf(xi, mu_single, sigma_single, theta)
                    y_m = len(y) * utils.lognorm_pdf(xi, mu_multiple, sigma_multiple, theta)

                    ax.plot(xi, y_s, c="tab:blue")
                    ax.plot(xi, y_m, c="tab:red")

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
                        results[j] = utils._pack_params(**result)
                         
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
            sp_swarm(*indices)

        else:
            P = mp.cpu_count()

            with mp.Pool(processes=P) as pool:

                manager = mp.Manager()

                in_queue = manager.Queue()
                out_queue = manager.Queue()

                swarm_kwds = dict(in_queue=in_queue,
                                  out_queue=out_queue)


                logger.info("Dumping everything into the queue!")
                for j, index in enumerate(indices):
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
                            logger.info("No results")
                            break

                        else:
                            j, index, result, meta = r

                            done[j] = True
                            if result is not None:
                                results[j] = utils._pack_params(**result)

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

            sigma = np.abs(results - np.median(results, axis=0)) \
                  / np.std(results, axis=0)
            sigma = np.sum(sigma, axis=1)

            
            # Only care about indices 1 and 2
            lower_bounds[3:] = -np.inf
            upper_bounds[3:] = +np.inf

            not_ok_bound = np.any(
                (np.abs(results - lower_bounds) <= tol_proximity) \
              + (np.abs(results - upper_bounds) <= tol_proximity), axis=1)

            not_ok_sigma = sigma > tol_sigma

            not_ok = not_ok_bound + not_ok_sigma

            done[not_ok] = False
            sp_swarm(*indices[not_ok], 
                     inits=[np.median(results[~not_ok], axis=0), "random"],
                     debug=False)

            print(f"There were {sum(not_ok_sigma)} results discarded for being outliers")
            print(f"There were {sum(not_ok_bound)} results discarded for being close to the edge")
            print(f"There were {sum(not_ok)} results discarded in total")


        import matplotlib.pyplot as plt
        for i in range(5):
            fig, axes = plt.subplots(1, 2)
            ax = axes[0]
            ax.set_title(f"{model_name} {i}")

            kwds = dict(vmin=lower_bounds[i] if np.isfinite(lower_bounds[i]) else 0.5,
                        vmax=upper_bounds[i] if np.isfinite(upper_bounds[i]) else 1.5)

            scat = ax.scatter(X[indices, 0], X[indices, 1], c=results.T[i], s=1,
                **kwds)

            ax.scatter(X[indices, 0][not_ok], X[indices, 1][not_ok], s=10, facecolor="none", edgecolor="k", zorder=-1,
                **kwds)

            axes[1].scatter(X[indices, 0], X[indices, 2], c=results.T[i], s=1, **kwds)
            axes[1].scatter(X[indices, 0][not_ok], X[indices, 2][not_ok], s=10, facecolor="none", edgecolor="k", zorder=-1,
                **kwds)

            for ax in axes:
                ax.set_ylim(ax.get_ylim()[::-1])
            cbar = plt.colorbar(scat)


        model_indices = indices[~not_ok]
        results = results[~not_ok]

        # Run the gaussian process on the single star estimates.
        gp_block_size = 10000
        G = 5 # number of kernel hyperparameters
        gp_predict_indices = (0, 1, 2, 3, 4)
        gp_parameters = np.zeros((len(gp_predict_indices), G))
        gp_predictions = np.nan * np.ones((X.shape[0], 2 * len(gp_predict_indices)))

        x = X[model_indices]
            
        for i, index in enumerate(gp_predict_indices):

            y = results[:, index]

            metric = np.var(x, axis=0)
            kernel = george.kernels.Matern32Kernel(metric, ndim=x.shape[1])

            gp = george.GP(kernel, 
                           mean=np.mean(y), fit_mean=True,
                           white_noise=np.log(np.std(y)), fit_white_noise=True)

            assert len(gp.parameter_names) == G

            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(y, quiet=True)

            gp.compute(x)
            logger.info("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
            logger.info("initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

            p0 = gp.get_parameter_vector()

            t_init = time()
            result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
            t_opt = time() - t_init


            gp.set_parameter_vector(result.x)
            logger.info("Result: {}".format(result))
            logger.info("Final logL = {:.2f}".format(gp.log_likelihood(y)))
            logger.info("Took {:.0f} seconds to optimize".format(t_opt))

            gp_parameters[i] = result.x

            # Predict the quantity and the variance.
            B = int(np.ceil(S / gp_block_size))

            logger.info(f"Predicting {model_name} {index}")

            with tqdm.tqdm(total=S) as pb:
                for b in range(B):
                    s, e = (b * gp_block_size, (b + 1)*gp_block_size)
                    sb = science_indices[s:1+e]

                    p, p_var = gp.predict(y, X[sb], return_var=True)
                    gp_predictions[sb, 2*i] = p
                    gp_predictions[sb, 2*i + 1] = p_var

                    pb.update(e - s)

            """
            p, p_var = gp.predict(y, X[randn], return_var=True)
            gp_predictions[randn, 2*i] = p
            gp_predictions[randn, 2*i + 1] = p_var
            
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            scat = ax.scatter(X.T[0][randn], X.T[1][randn], 
                c=gp_predictions[:, 2*i][randn], s=1)
            cbar = plt.colorbar(scat)

            ax.set_title(f"{index} mu")

            fig, ax = plt.subplots()
            scat = ax.scatter(X.T[0][randn], X.T[1][randn], 
                c=np.sqrt(gp_predictions[:, 2*i + 1][randn]), s=1)
            cbar = plt.colorbar(scat)

            ax.set_title(f"{index} sigma")

            raise a
            """

        model_results[model_name] = [model_indices, results, gp_parameters, gp_predictions]

        # Save predictions so far?
        """
        logger.info(f"Saved progress to {results_path}")
        with open(results_path, "wb") as fp:
            pickle.dump(dict(config=config, models=model_results), fp)
        """

    with h5.File(results_path, "w") as h:

        group = h.create_group("models")

        for model_name in results["models"].keys():

            sub_group = group.create_group(model_name)

            dataset_names = (
                "data_indices", 
                "mixture_model_results", 
                "gp_parameters", 
                "gp_predictions"
            )
            for i, dataset_name in enumerate(dataset_names):
                d = sub_group.create_dataset(dataset_name, 
                                             data=results["models"][model_name][i])
    
    # Calculate PDFs and estimates, etc.

    with open(f"{results_path}.meta", "w") as fp:
        fp.write(yaml.dump(config))
