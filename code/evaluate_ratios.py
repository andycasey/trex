
import yaml
import h5py as h5

import os
import numpy as np
import logging
from scipy import (optimize as op, integrate, special)
from hashlib import md5
import sys
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


if __name__ == "__main__":

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


    for model_name, model_config in config["models"].items():
        logger.info(f"Running {model_name} model")

        data_indices = results[model_name]["data_indices"][()]
        

        lns = list(model_config["kdtree_label_names"]) 
        predictor_label_name = model_config["predictor_label_name"]

        X = np.vstack([sources[ln][()] for ln in lns]).T[data_indices]
        N = X.shape[0]

        # model_selection/likelihood/{model_name}/single
        # model_selection/likelihood/{model_name}/multiple
        # model_selection/likelihood/{model_name}/ratio_single
        # model_selection/likelihood/joint_ratio_single
        
        group_name = f"model_selection/likelihood/{model_name}"
        if group_name not in results:
            g = results.create_group(group_name)
        else:
            g = results[group_name]

        # Calculate likelihood ratios.
        for k in ("single", "multiple", "ratio_single"):
            if k in g: del g[k]

        y = sources[predictor_label_name][()][data_indices]

        w, w_var = results[f"{model_name}/gp_predictions/theta"][()].T

        ln_s = g.create_dataset("single", data=np.nan * np.ones(N))

        mu_s, mu_s_var = results[f"{model_name}/gp_predictions/mu_single"][()].T
        sigma_s, sigma_s_var = results[f"{model_name}/gp_predictions/sigma_single"][()].T

        # Evaluate log-likelihood
        ln_s[:] = np.log(w) + utils.normal_lpdf(y, mu_s, sigma_s)

        ln_m = g.create_dataset("multiple", data=np.nan * np.ones(N))
        sigma_m, sigma_m_var = results[f"{model_name}/gp_predictions/sigma_multiple"][()].T

        if "mu_multiple" in results[f"{model_name}/gp_predictions"]:
            mu_m = results[f"{model_name}/gp_predictions/mu_multiple"][()][:, 0]

            # TODO: draws in mu_m

        else:
            scalar = model_config["mu_multiple_scalar"]
            mu_m = np.log(mu_s + scalar * sigma_s) + sigma_m**2


        ln_m[:] = np.log(1-w) + utils.lognormal_lpdf(y, mu_m, sigma_m)

        # If y < mu_s then it is clearly a single star, but sometimes the log-normal can have a
        # little bit more support.

        print("ANDY FIX SUPPORT PROBLEM")


        lp = np.array([ln_s[()], ln_m[()]]).T
        
        with np.errstate(under="ignore"):
            ratio = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))    

        full_ratios = np.zeros(sources["source_id"].size, dtype=float)
        full_ratios[data_indices] = ratio
        g.create_dataset("ratio_single", data=full_ratios)



    # Calculate joint ratios.
    model_names = list(config["models"].keys())
    
    if len(model_names) > 1:

        # Calculate joint likelihood ratio.
        kw = dict(shape=(len(model_names), sources["source_id"].size), dtype=float)
        ln_s = np.nan * np.ones(**kw)
        ln_m = np.nan * np.ones(**kw)

        for i, model_name in enumerate(model_names):
            data_indices = results[f"{model_name}/data_indices"]
            ln_s[i, data_indices] = results[f"model_selection/likelihood/{model_name}/single"][()]
            ln_m[i, data_indices] = results[f"model_selection/likelihood/{model_name}/multiple"][()]

        ln_s = np.nansum(ln_s, axis=0)
        ln_m = np.nansum(ln_m, axis=0)

        lp = np.array([ln_s, ln_m]).T

        with np.errstate(under="ignore"):
            ratio = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

        dataset_name = "model_selection/likelihood/joint_ratio_single"
        if dataset_name in results:
            del results[dataset_name]

        no_info = (ln_s == 0) & (ln_m == 0)
        ratio[no_info] = np.nan

        results.create_dataset(dataset_name, data=ratio)

    # Calculate excess K
    if "rv" in model_names:

        model_config = config["models"]["rv"]

        logger.info("Calculating excess K in 'rv' model")
        
        rv_jitter = sources[model_config["predictor_label_name"]][()][data_indices]
        rv_nb_transits = sources["rv_nb_transits"][()][data_indices]

        K_est = np.sqrt(2) * rv_jitter

        # Add formal errors.
        N = rv_nb_transits
        e_K_est = K_est * np.sqrt(1 - (2/(N-1)) * (special.gamma(N/2)/special.gamma((N-1)/2))**2)

        dataset_name = "rv/gp_predictions/K"
        if dataset_name in results:
            del results[dataset_name]
        results.create_dataset(dataset_name, data=np.vstack([K_est, e_K_est]).T)
    
    #
    if False:
        logger.info("Calculating ratio draws")


        # Calculate draws.
        D = 16 # number of draws per source
        B = 10000 # batch size
        M = len(model_names)
        N = len(data_indices)
        num_batches = int(np.ceil(N / B))

        # Pre-store y.
        y = np.nan * np.zeros((N, M))
        gn = "ratios"
        if gn in results:
            del results[gn]

        ratios = results.create_group(gn)
        
        for j, model_name in enumerate(model_names):
            predictor_label_name = config["models"][model_name]["predictor_label_name"]
            y[:, j] = sources[predictor_label_name][()][data_indices]

            ratios.create_dataset(model_name, shape=(N, D), dtype=float)

        ratios.create_dataset("joint", shape=(N, D), dtype=float)

        logger.info("Evaluating ratio draws")


        SMALL = 1e-15
        for i in tqdm(range(num_batches)):

            si, ei = i * B, (i + 1) * B

            ln = np.nan * np.ones((y.shape[0], D, M, 2))
            
            for j, model_name in enumerate(model_names):

                scalar = config["models"][model_name]["mu_multiple_scalar"]

                w, w_var = results[f"{model_name}/gp_predictions/theta"][()][si:ei].T
                mu_s, mu_s_var = results[f"{model_name}/gp_predictions/mu_single"][()][si:ei].T
                sigma_s, sigma_s_var = results[f"{model_name}/gp_predictions/sigma_single"][()][si:ei].T
                sigma_m, sigma_m_var = results[f"{model_name}/gp_predictions/sigma_multiple"][()][si:ei].T
                

                # Do the draws.
                B = w.size
                w_ = np.random.normal(w, w_var**0.5, size=(D, B))
                mu_s_ = np.random.normal(mu_s, mu_s_var**0.5, size=(D, B))
                sigma_s_ = np.random.normal(sigma_s, sigma_s_var**0.5, size=(D, B))
                sigma_m_ = np.random.normal(sigma_m, sigma_m_var**0.5, size=(D, B))

                w_ = np.clip(w_, SMALL, 1 - SMALL)
                # TODO: Clip other values to their bounds?
                mu_s_ = np.abs(mu_s_)
                sigma_m_ = np.abs(sigma_m_)
                sigma_s_ = np.abs(sigma_s_)


                # TODO: don't always fix it.
                mu_m_ = np.log(mu_s_ + scalar * sigma_s_) + sigma_m_**2

                ln[:, :, j, 0] = (np.log(w_) + utils.normal_lpdf(y[si:ei, j], mu_s_, sigma_s_)).T
                ln[:, :, j, 1] = (np.log(1-w_) + utils.lognormal_lpdf(y[si:ei, j], mu_m_, sigma_m_)).T

                with np.errstate(under="ignore"):
                    results[f"ratios/{model_name}"][si:ei, :] = np.exp(ln[:, :, j, 0] - special.logsumexp(ln[:, :, j], axis=2))

                assert np.all(np.isfinite(results[f"ratios/{model_name}"][si:ei]))

                # TODO: Check if y < mu_s that it is definitely a single star.


            # Calc likelihoods all together.
            _ = special.logsumexp(ln, axis=2)
            results["ratios/joint"][si:ei, :] = np.exp(_[:, :, 0] - special.logsumexp(_, axis=2))

            """
            idx = 400

            def plotit(idx, bins=None):

                fig, axes = plt.subplots(3)
                if bins is None:
                    bins = np.linspace(0, 1, 51)

                axes[0].hist(results["ratios/rv"][idx], bins=bins)
                axes[1].hist(results["ratios/ast"][idx], bins=bins)
                axes[2].hist(results["ratios/joint"][idx], bins=bins)

                return fig 

            fig = plotit(400)

            raise a
            """

            # TODO Store likelihoods?
        
    else:
        logger.info("Not calculating ratio draws!")

    results.close()
