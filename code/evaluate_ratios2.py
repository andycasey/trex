
import yaml
import h5py as h5

import os
import numpy as np
import logging
import sys
import warnings
from scipy import (optimize as op, integrate, special)
import multiprocessing as mp
from hashlib import md5
from tqdm import tqdm

from time import time
import george

import utils
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



def _cross_match(A_source_ids, B_source_ids):

    A = np.array(A_source_ids, dtype=np.long)
    B = np.array(B_source_ids, dtype=np.long)

    ai = np.where(np.in1d(A, B))[0]
    bi = np.where(np.in1d(B, A))[0]
    
    a_idx, b_idx = (ai[np.argsort(A[ai])], bi[np.argsort(B[bi])])

    # Sanity checks
    assert a_idx.size == b_idx.size
    assert np.all(A[a_idx] == B[b_idx])
    return (a_idx, b_idx)



def clipped_predictions(theta, mu_single, sigma_single, sigma_multiple, bounds):
    theta = np.clip(theta, *bounds["theta"])
    mu_single = np.clip(mu_single, *bounds["mu_single"])
    sigma_single = np.clip(sigma_single, *bounds["sigma_single"])
    sigma_multiple = np.clip(sigma_single, *bounds["sigma_multiple"])

    return np.array([theta, mu_single, sigma_single, sigma_multiple])



def calc_p_single(y, theta, mu_single, sigma_single, sigma_multiple, mu_multiple_scalar):

    with warnings.catch_warnings(): 
        # I'll log whatever number I want python you can't tell me what to do
        warnings.simplefilter("ignore") 

        mu_multiple = np.log(mu_single + mu_multiple_scalar * sigma_single) + sigma_multiple**2
        
        ln_s = np.log(theta) + utils.normal_lpdf(y, mu_single, sigma_single)
        ln_m = np.log(1-theta) + utils.lognormal_lpdf(y, mu_multiple, sigma_multiple)

        # FIX BAD SUPPORT.

        # This is a BAD MAGIC HACK where we are just going to flip things.
        """
        limit = mu_single - 2 * sigma_single
        bad_support = (y <= limit) * (ln_m > ln_s)
        ln_s_bs = np.copy(ln_s[bad_support])
        ln_m_bs = np.copy(ln_m[bad_support])
        ln_s[bad_support] = ln_m_bs
        ln_m[bad_support] = ln_s_bs
        """
        ln_s = np.atleast_1d(ln_s)
        ln_m = np.atleast_1d(ln_m)

        lp = np.array([ln_s, ln_m]).T

        #assert np.all(np.isfinite(lp))

        p_single = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

    return (p_single, ln_s, ln_m)


SMALL = 1e-5
N_draws = 256
percentiles = [2.5, 16, 50, 8, 97.5]
OVERWRITE = True
BF_CLIP = (0, 1e2)

if __name__ == "__main__":

    # Load in the results path.
    results_path = sys.argv[1]
    results_dir = os.path.dirname(results_path)

    if not results_path.endswith(".h5"):
        raise ValueError("this is for the new style format!!")


    results = h5.File(results_path, "a")

    config = yaml.load(results.attrs["config"], Loader=yaml.Loader)

    np.random.seed(int(config["random_seed"]))

    # Load data.
    pwd = os.path.dirname(results.attrs["config_path"]).decode("utf-8")
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")
    sources = data["sources"]

    # Create a results group where we will combine results from different models.
    if "results" in results:
        if not OVERWRITE:
            raise ValueError("results already exist in this file and not overwriting")

        del results["results"]

    group = results.create_group("results", track_order=True)

    # Get unique source ids.
    model_names = list(config["models"].keys())
    source_ids = np.hstack([results[f"models/{mn}/gp_predictions/source_id"][()] \
                            for mn in model_names])
    source_indices = np.hstack([results[f"models/{mn}/gp_predictions/source_indices"][()] \
                                for mn in model_names])
    
    source_ids, unique_indices = np.unique(source_ids, return_index=True)
    source_indices = source_indices[unique_indices]

    group.create_dataset("source_id", data=source_ids)
    group.create_dataset("source_indices", data=source_indices)

    M, N = (len(model_names), len(source_ids))
    kw = dict(shape=(N, ), dtype=float, fillvalue=np.nan)

    # Because we track the order when things are created, let's do it in a sensible way.

    for model_name in model_names:
        predictor_label_name = config["models"][model_name]["predictor_label_name"]
        group.create_dataset(predictor_label_name, data=sources[predictor_label_name][()][source_indices])


    for model_name in model_names:
        for suffix in ("theta", "mu_single", "sigma_single", "sigma_multiple"):
            group.create_dataset(f"{model_name}_{suffix}",
                                 shape=(N, 2), dtype=float, fillvalue=np.nan)
  
    ignore = []
    names = []
    names.extend(["K", "K_err"])
    for name in names:
        if name not in ignore:
            group.create_dataset(name, **kw)


    # Let's calculate shit.
    M = len(model_names)
    P = len(percentiles)
    shape = (N, M + 1, P)
    p_single_percentiles = np.empty(shape)


    # Since we are calculating percentiles and not using draws, we either have to store the draws
    # between calculating all the models, or we have to do it on a per star basis.
    # Both kinda suck. We'll do it on a per-star basis.


    # Get all bounds.
    M = len(model_names)

    # Build some big ass arrays
    ys = np.empty((N, M))
    ys[:] = np.nan

    bounds = []

    gp_predictions = np.empty((N, M, 4, 2))
    gp_predictions[:] = np.nan

    source_indices = results["results"]["source_indices"][()]

    for m, (model_name, model_config) in enumerate(config["models"].items()):

        predictor_label_name = model_config["predictor_label_name"]

        model_source_indices = results[f"models/{model_name}/gp_predictions/source_indices"][()]

        a_idx, b_idx = _cross_match(source_indices, model_source_indices)
        ys[a_idx, m] = sources[predictor_label_name][()][model_source_indices]

        gp_predictions[a_idx, m, 0] = results[f"models/{model_name}/gp_predictions/theta"][()]
        gp_predictions[a_idx, m, 1] = results[f"models/{model_name}/gp_predictions/mu_single"][()]
        gp_predictions[a_idx, m, 2] = results[f"models/{model_name}/gp_predictions/sigma_single"][()]
        gp_predictions[a_idx, m, 3] = results[f"models/{model_name}/gp_predictions/sigma_multiple"][()]

        b = model_config["bounds"]
        b.update(theta=[0, 1])
        b["theta"] = np.array(b["theta"]) + [+SMALL, -SMALL]
        bounds.append(b)


    def calc_probs(ys, gp_predictions, bounds, N_draws, percentiles):


        M = len(ys)
        shape = (M + 1, N_draws)
        obj_p_singles = np.empty(shape)
        obj_ln_s = np.empty(shape)
        obj_ln_m = np.empty(shape)

        # gp_predictions has shape M, 4, 2
        mu = gp_predictions[:, :, 0]
        var = gp_predictions[:, :, 1]

        for m, y in enumerate(ys):

            # Do draws for all models
            _slice = (m, slice(0, None))
            if not np.all(np.isfinite(np.hstack([y, mu[m], var[m]]))):
                obj_p_singles[_slice] = np.nan
                obj_ln_s[_slice] = np.nan
                obj_ln_s[_slice] = np.nan
                continue

            draws = clipped_predictions(
                *np.random.multivariate_normal(mu[m], np.diag(var[m]), size=N_draws).T, 
                bounds[m])

            obj_p_singles[_slice], obj_ln_s[_slice], obj_ln_m[_slice] \
                = calc_p_single(y, *draws, mu_multiple_scalar=3)

        # Now calculate the joint probabilities
        obj_ln_s[-1, :] = np.nansum(obj_ln_s[:-1, :], axis=0)
        obj_ln_m[-1, :] = np.nansum(obj_ln_m[:-1, :], axis=0)

        lp = np.array([obj_ln_s[-1, :], obj_ln_m[-1, :]])

        obj_p_singles[-1, :] = np.exp(lp[0] - special.logsumexp(lp, axis=0))

        # Then calculate percentiles ofthose probabilities.
        obj_p_singles[~np.isfinite(obj_p_singles)] = -1
        p = np.percentile(obj_p_singles, percentiles, axis=1).T
        p[p == -1] = np.nan
        return p


    #foo = calc_probs(ys[0], gp_predictions[0], bounds, N_draws, percentiles)

    def mp_swarm(*_, bounds, in_queue=None, out_queue=None):

        #print(f"in mp_swarm {in_queue}")
        while True:

            try:
                i, y, gpp = in_queue.get_nowait()

            except mp.queues.Empty:
                #print("empty")
                logger.info("Queue is empty")
                break

            except StopIteration:
                #print("stopped")
                logger.warning("Swarm is bored")
                break

            except:
                #print("other")
                logger.exception("Unexpected exception:")
                break

            else:
                
                #print(f"in mp swarm {i} {y} {gpp} {in_queue} {out_queue}")
                try:
                    p = calc_probs(y, gpp, bounds, 256, [2.5, 16, 50, 8, 97.5])
                except:
                    logger.exception("failured")
                    break

                out_queue.put((i, p))
                #print("done")

        #print("out")



    P = 8
    with mp.Pool(processes=P) as pool:

        manager = mp.Manager()

        in_queue = manager.Queue()
        out_queue = manager.Queue()

        swarm_kwds = dict(in_queue=in_queue,
                          out_queue=out_queue,
                          bounds=bounds)
        
        for i, (y, gpp) in enumerate(tqdm(zip(ys, gp_predictions), total=N, desc="Dumping")):
            in_queue.put((i, y, gpp))

        j = []
        for _ in range(P):
            j.append(pool.apply_async(mp_swarm, [], kwds=swarm_kwds))


        with tqdm(total=N, desc="Collecting") as pbar:
            count = 0

            while N >= count:

                # Check for output.
                try:
                    r = out_queue.get(timeout=30)

                except mp.queues.Empty:
                    logger.info("No npm_results")
                    break

                else:
                    #print("got one")
                    i, p = r
                    p_single_percentiles[i, :] = p
                    count += 1

                    pbar.update(1)

    # Store the p_single_percentiles and related attributes
    results.create_dataset("results/p_single_percentiles", data=p_single_percentiles)
    results["results"].attrs.update(percentiles=percentiles,
                                    model_names=list(model_names) + ["joint"])

    # We will need the RAM later..
    del p_single_percentiles, ys, gp_predictions


    """
    for n in tqdm(range(N)):




        # TODO: store the predictor variable in the models/{nodel_name}/gp_predictions?

        # Create empty arrays to store the p_singles for this objct.
        shape = (M + 1, N_DRAWS)
        obj_p_singles = np.empty(shape)
        obj_ln_s = np.empty(shape)
        obj_ln_m = np.empty(shape)

        for m, (model_name, model_config) in enumerate(config["models"].items()):

            predictor_label_name = model_config["predictor_label_name"]

            source_index = results[f"models/{model_name}/gp_predictions/source_indices"][n]
            y = sources[predictor_label_name][source_index]

            # Get the expectations and variances of the gaussian processes
            theta, theta_var = results[f"models/{model_name}/gp_predictions/theta"][n]
            mu_single, mu_single_var = results[f"models/{model_name}/gp_predictions/mu_single"][n]
            sigma_single, sigma_single_var = results[f"models/{model_name}/gp_predictions/sigma_single"][n]
            sigma_multiple, sigma_multiple_var = results[f"models/{model_name}/gp_predictions/sigma_multiple"][n]

            bounds = model_config["bounds"]
            bounds.update(theta=[0, 1])
            bounds["theta"] = np.array(bounds["theta"]) + [+SMALL, -SMALL]

            # Do a single prediction at the expectation value of the GP.
            #args = clipped_predictions(theta, mu_single, sigma_single, sigma_multiple, bounds)
            #p_single_percentiles[n, m, 0], *_ = calc_p_single(y, *args, mu_multiple_scalar=model_config["mu_multiple_scalar"])

            if N_DRAWS > 0:

                # Now do it for all draws.
                mu = np.array([theta, mu_single, sigma_single, sigma_multiple])
                cov = np.diag([theta_var, mu_single_var, sigma_single_var, sigma_multiple_var])

                draws = clipped_predictions(*np.random.multivariate_normal(mu, cov, size=N_DRAWS).T, bounds)

                _slice = (m, slice(0, None))
                obj_p_singles[_slice], obj_ln_s[_slice], obj_ln_m[_slice] = \
                    calc_p_single(y, *draws, mu_multiple_scalar=model_config["mu_multiple_scalar"])


        # OK, now let's calculate joint probabilities
        obj_ln_s[-1, :] = np.sum(obj_ln_s[:-1, :], axis=0)
        obj_ln_m[-1, :] = np.sum(obj_ln_m[:-1, :], axis=0)

        lp = np.array([obj_ln_s[-1, :], obj_ln_m[-1, :]])

        obj_p_singles[-1, :] = np.exp(lp[0] - special.logsumexp(lp, axis=0))

        # Then calculate percentiles of those probabilities.
        p_single_percentiles[n, :, :] = np.percentile(obj_p_singles, percentiles, axis=1).T


    raise a



    # OK let's start calculating shit
    for m, (model_name, model_config) in enumerate(config["models"].items()):
        logger.info(f"Running {model_name} model")

        predictor_label_name = model_config["predictor_label_name"]
       
        model_source_indices = results[f"models/{model_name}/gp_predictions/source_indices"][()]
        y = sources[predictor_label_name][()][model_source_indices]

        # Get predictions from our GPs
        theta, theta_var = results[f"models/{model_name}/gp_predictions/theta"][()].T
        mu_single, mu_single_var = results[f"models/{model_name}/gp_predictions/mu_single"][()].T
        sigma_single, sigma_single_var = results[f"models/{model_name}/gp_predictions/sigma_single"][()].T
        sigma_multiple, sigma_multiple_var = results[f"models/{model_name}/gp_predictions/sigma_multiple"][()].T

        bounds = model_config["bounds"]
        #bounds.setdefault("bounds", [0, 1])
        bounds.update(theta=[0, 1])
        bounds["theta"] = np.array(bounds["theta"]) + [+SMALL, -SMALL]

        def clipped_predictions(theta, mu_single, sigma_single, sigma_multiple):
            theta = np.clip(theta, *bounds["theta"])
            mu_single = np.clip(mu_single, *bounds["mu_single"])
            sigma_single = np.clip(sigma_single, *bounds["sigma_single"])
            sigma_multiple = np.clip(sigma_single, *bounds["sigma_multiple"])

            return np.array([theta, mu_single, sigma_single, sigma_multiple])


        def calc_p_single(y, theta, mu_single, sigma_single, sigma_multiple):

            with warnings.catch_warnings(): 
                # I'll log whatever number I want python you can't tell me what to do
                warnings.simplefilter("ignore") 

                mu_multiple = np.log(mu_single + model_config["mu_multiple_scalar"] * sigma_single) + sigma_multiple**2
                
                ln_s = np.log(theta) + utils.normal_lpdf(y, mu_single, sigma_single)
                ln_m = np.log(1-theta) + utils.lognormal_lpdf(y, mu_multiple, sigma_multiple)

                # FIX BAD SUPPORT.

                # This is a BAD MAGIC HACK where we are just going to flip things.
                limit = mu_single - 2 * sigma_single
                bad_support = (y <= limit) * (ln_m > ln_s)
                ln_s_bs = np.copy(ln_s[bad_support])
                ln_m_bs = np.copy(ln_m[bad_support])
                ln_s[bad_support] = ln_m_bs
                ln_m[bad_support] = ln_s_bs

                lp = np.array([ln_s, ln_m]).T

                #assert np.all(np.isfinite(lp))

                p_single = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))


            return (p_single, ln_s, ln_m)


        # Calculate value even if not doing draws.
        args = clipped_predictions(theta, mu_single, sigma_single, sigma_multiple)
        p_single, ln_s, ln_m = calc_p_single(y, *args)
        
        if N_DRAWS > 1:

            # Make draws.
            p_single_draws = np.empty((y.size, len(percentiles)))

            X = np.array([theta, mu_single, sigma_single, sigma_multiple])
            D = np.array([theta_var, mu_single_var, sigma_single_var, sigma_multiple_var])

            for i, (y_, X_, D_) in enumerate(tqdm(zip(y, X.T, D.T), total=y.size, desc=f"Making GP draws for {model_name}")):

                if i > 100000:
                    continue

                draws = clipped_predictions(*np.random.multivariate_normal(X_, np.diag(D_), size=N_DRAWS).T)
                p_single_draws[i] = np.percentile(calc_p_single(y_, *draws), percentiles)


        # Translate values.
        a_idx, b_idx = _cross_match(source_indices, model_source_indices)

        # Some issue with writing to hdf5 files using indices that I cbf figuring out.
        _shape = lambda a: tuple([N] + list(a.shape[1:]))

        def _update(name, array):
            __ = np.nan * np.ones(_shape(array))
            __[a_idx] = array[b_idx]
            results[name][:] = __

        _update(f"results/p_{model_name}_single", p_single)
        _update(f"results/ll_{model_name}_single", ln_s)
        _update(f"results/ll_{model_name}_multiple", ln_m)
        

        name = f"p_{model_name}_single_percentiles"
    
        group.create_dataset(name, dtype=float, shape=_shape(p_single_draws), fillvalue=np.nan)
        _update(f"results/{name}", p_single_draws)

        _update(f"results/{model_name}_theta", np.vstack([theta, theta_var]).T)
        _update(f"results/{model_name}_mu_single", np.vstack([mu_single, mu_single_var]).T)
        _update(f"results/{model_name}_sigma_single", np.vstack([sigma_single, sigma_single_var]).T)
        _update(f"results/{model_name}_sigma_multiple", np.vstack([sigma_multiple, sigma_multiple_var]).T)




    # Calculate joint ratios.
    if len(model_names) > 1:

        raise a

        logger.info("Calculating joint properties")

        # Calculate the following things:
        # - ll_single
        # - ll_multiple
        # - p_single
        # - bf_multiple

        ln_s = np.array([results[f"results/ll_{mn}_single"][()] for mn in model_names])
        ln_m = np.array([results[f"results/ll_{mn}_multiple"][()] for mn in model_names])
        
        # Need to deal with nans in either/both.
        print("ANDY DO THIS PROPERLY")

        LOG_SMALL = -1000

        ln_s[~np.isfinite(ln_s)] = LOG_SMALL
        
        ln_m[~np.isfinite(ln_m)] = LOG_SMALL
        
        ln_s = special.logsumexp(ln_s, axis=0)
        ln_m = special.logsumexp(ln_m, axis=0)

        results["results/ll_single"][:] = ln_s
        results["results/ll_multiple"][:] = ln_m

        lp = np.array([ln_s, ln_m]).T

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            results["results/p_single"][:] = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

            results["results/bf_multiple"][:] = np.clip(np.exp(ln_m - ln_s), *BF_CLIP)

        '''
        no_data = (lp.T[0] == lp.T[1]) # knows nothing
        lp[no_data] = np.nan
        p_single[no_data] = np.nan
        '''

    """


    # Calculate radial velocity excess
    if "rv" in model_names:
        model_config = config["models"]["rv"]

        logger.info("Estimating radial velocity semi-amplitude")

        source_indices = results["results"]["source_indices"][()]

        rv_jitter = sources[model_config["predictor_label_name"]][()][source_indices]
        rv_nb_transits = sources["rv_nb_transits"][()][source_indices]

        K = np.sqrt(2) * rv_jitter

        # Add formal errors.
        N = rv_nb_transits
        K_err = K * np.sqrt(1 - (2/(N-1)) * (special.gamma(N/2)/special.gamma((N-1)/2))**2)

        results["results/K"][:] = K
        results["results/K_err"][:] = K_err

    else:
        logger.warn("Not estimating radial velocity semi-amplitude because no 'rv' model")


    # Create a temporary FITS file to explore the data.
    logger.info("Writing temporary file to explore the data.")
    
    from astropy.io import fits

    idx = np.random.choice(len(source_ids), size=1_000_000)
    idx = np.arange(len(source_ids))

    columns = []
    source_indices = results["results/source_indices"]

    ignore_column_names = ("source_indices", "p_single_percentiles")
    include_column_names = ("bp_rp", "phot_g_mean_mag", "absolute_g_mag", "l", "b", "parallax")
    for name in include_column_names:
        columns.append(fits.Column(name=name,
                                   array=sources[name][()][source_indices][idx],
                                   format=sources[name].dtype))
        logger.info(f"Added column {name}")

    for name in results["results"].keys():
        if name in ignore_column_names:
            continue

        v = results[f"results/{name}"][()][idx]
        if len(v.shape) > 1:
            columns.append(fits.Column(name=name, array=v[:, 0], format=v.dtype))
        else:
            columns.append(fits.Column(name=name, array=v, format=v.dtype))

        logger.info(f"Added column {name}")

    # Handle percentiles nicely.
    for i, model_name in enumerate(results["results"].attrs["model_names"]):
        for j, percentile in enumerate(results["results"].attrs["percentiles"]):

            name = f"p_single_{model_name}_{percentile:.1f}"
            array = results["results/p_single_percentiles"][()][idx, i, j]
            columns.append(fits.Column(name=name, array=array, format=array.dtype))

            logger.info(f"Added column {name}")

    logger.info("If we were to die, now's the time.")

    fits.BinTableHDU.from_columns(columns).writeto(os.path.join(results_dir, "explore.fits"),
                                                   overwrite=True)

    logger.info("Closing up..")
    del columns

    data.close()
    results.close()
    logger.info("Fin")
