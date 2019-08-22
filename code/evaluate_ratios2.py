
import yaml
import h5py as h5

import os
import numpy as np
import logging
import sys
import warnings
from scipy import (optimize as op, integrate, special)
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

    prefixes = ("ll", "p", "bf")
    ignore = ["p_multiple", "bf_single"] \
           + [f"bf_{mn}_single" for mn in model_names] \
           + [f"p_{mn}_multiple" for mn in model_names]

    names = []

    for prefix in prefixes:

        for model_name in model_names:
            names.extend([
                f"{prefix}_{model_name}_single",
                f"{prefix}_{model_name}_multiple"
            ])

        for suffix in ("single", "multiple"):
            names.append(f"{prefix}_{suffix}")

    names.extend(["K", "K_err"])
    for name in names:
        if name not in ignore:
            group.create_dataset(name, **kw)

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

        # Clip as needed.
        # TODO: don't code so bad
        SMALL = 1e-3
        theta_bounds = model_config["bounds"].get("theta", [0, 1])
        theta_bounds[0] = max(theta_bounds[0], SMALL)
        theta_bounds[1] = min(theta_bounds[1], 1 - SMALL)

        theta = np.clip(theta, *theta_bounds)

        if "mu_single" in model_config["bounds"]:
            mu_single = np.clip(mu_single, *model_config["bounds"]["mu_single"])
        if "sigma_single" in model_config["bounds"]:
            #sigma_single = np.clip(sigma_single, *model_config["bounds"]["sigma_single"])

            if model_name == "ast":
                sigma_single = np.clip(sigma_single, *model_config["bounds"]["sigma_single"])
            else:
                sigma_single = np.clip(sigma_single, 0.25, 5)



        if "sigma_multiple" in model_config["bounds"]:
            sigma_multiple = np.clip(sigma_multiple, *model_config["bounds"]["sigma_multiple"])

        scalar = model_config["mu_multiple_scalar"]
        with warnings.catch_warnings(): 
            # I'll log whatever number I want python you can't tell me what to do
            warnings.simplefilter("ignore") 

            mu_multiple = np.log(mu_single + scalar * sigma_single) + sigma_multiple**2
            
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

            print(f"There were {np.sum(bad_support)} in {model_name} with bad support")

            lp = np.array([ln_s, ln_m]).T

            p_single = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

            # calc bayes factor
            bf = np.exp(ln_m - ln_s)

            # Clip bayes factors.
            bf = np.clip(bf, *BF_CLIP)

        # fill in values when we have nothing.
        bad = ~np.isfinite(y)
        bf[bad] = np.nan
        p_single[bad] = np.nan
        ln_s[bad] = np.nan
        ln_m[bad] = np.nan

        # Translate values.
        a_idx, b_idx = _cross_match(source_indices, model_source_indices)

        # Some issue with writing to hdf5 files using indices that I cbf figuring out.
        def _update_sources(name, array):
            __ = np.nan * np.ones(N)
            __[a_idx] = array[b_idx]
            results[name][:] = __

        _update_sources(f"results/ll_{model_name}_single", ln_s)
        _update_sources(f"results/ll_{model_name}_multiple", ln_m)
        _update_sources(f"results/p_{model_name}_single", p_single)
        _update_sources(f"results/bf_{model_name}_multiple", bf)

        def _update_gp_predictions(name, array):
            __ = np.nan * np.ones((N, 2))
            __[a_idx] = array[:]
            results[name][:] = __

        _update_gp_predictions(f"results/{model_name}_theta", np.vstack([theta, theta_var]).T)
        _update_gp_predictions(f"results/{model_name}_mu_single", np.vstack([mu_single, mu_single_var]).T)
        _update_gp_predictions(f"results/{model_name}_sigma_single", np.vstack([sigma_single, sigma_single_var]).T)
        _update_gp_predictions(f"results/{model_name}_sigma_multiple", np.vstack([sigma_multiple, sigma_multiple_var]).T)

    # Calculate joint ratios.
    if len(model_names) > 1:

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

        """
        no_data = (lp.T[0] == lp.T[1]) # knows nothing
        lp[no_data] = np.nan
        p_single[no_data] = np.nan
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

    columns = []
    source_indices = results["results/source_indices"]

    include_column_names = ("bp_rp", "phot_g_mean_mag", "absolute_g_mag", "l", "b", "parallax")
    for name in include_column_names:
        columns.append(fits.Column(name=name,
                                   array=sources[name][()][source_indices],
                                   format=sources[name].dtype))

    for k in results["results"].keys():
        if k == "source_indices":
            continue
        v = results[f"results/{k}"]
        if len(v.shape) > 1:
            columns.append(fits.Column(name=k, array=v[:, 0], format=v.dtype))
        else:
            columns.append(fits.Column(name=k, array=v, format=v.dtype))

    fits.BinTableHDU.from_columns(columns).writeto(os.path.join(results_dir, "explore.fits"),
                                                   overwrite=True)

    logger.info("Closing up..")
    del columns

    data.close()
    results.close()
    logger.info("Fin")
