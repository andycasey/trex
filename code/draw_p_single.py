
""" Estimate single star probabilities for a single Gaia source. """

import yaml
import h5py as h5

import os
import numpy as np
import logging
import sys
import warnings
from scipy import (optimize as op, integrate, special)
from hashlib import md5

import matplotlib.pyplot as plt

import utils

logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
    logger.propagate = False

logger.setLevel(logging.INFO)


D = 1024 # number of draws
SMALL = 1e-5

if __name__ == "__main__":

    # Load in the results path.
    results_path = sys.argv[1]
    results_dir = os.path.dirname(results_path)

    if not results_path.endswith(".h5"):
        raise ValueError("this is for the new style format!!")


    results = h5.File(results_path, "r")

    config = yaml.load(results.attrs["config"], Loader=yaml.Loader)
    
    # Load data.
    pwd = os.path.dirname(results.attrs["config_path"]).decode("utf-8")
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")
    sources = data["sources"]

    # Get the source id
    if len(sys.argv) < 3:
        # Chose a random one that has RV (and thus probably has AST) 
        source_id = np.random.choice(results["models/rv/gp_predictions/source_id"][()])

    else:
        source_id = int(sys.argv[2])

    #source_id = 170415678011724288
    #source_id = 5969884110561654784
    #source_id = 5645608723896366464


    def main(source_id):

        if source_id is None:
            source_id = np.random.choice(results["models/rv/gp_predictions/source_id"][()])

        print(f"Source is {source_id}")

        try:
            sources_index = np.where(sources["source_id"][()] == source_id)[0][0]
            results_index = np.where(results["results/source_id"][()] == source_id)[0][0]

        except:
            raise

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


        # Calculate probabilities for all models.
        model_names = list(config["models"].keys())
        M = len(model_names)

        

        shape = (1, M + 1, 1 + D)
        p_single = np.empty(shape)
        ln_s = np.empty(shape)
        ln_m = np.empty(shape)

        p_single[:] = np.nan
        ln_s[:] = np.nan
        ln_m[:] = np.nan


        for m, (model_name, model_config) in enumerate(config["models"].items()):

            predictor_label_name = model_config["predictor_label_name"]

            # Get predictors directly from source.
            y = sources[predictor_label_name][()][sources_index]

            try:
                index = np.where(results[f"models/{model_name}/gp_predictions/source_id"][()] == source_id)[0][0]

            except IndexError:
                print(f"No data or model predictions for {model_name}")
                p_single[0, m, 0] = np.nan
                ln_s[0, m, :] = 0
                ln_m[0, m, :] = 0
                continue

            # Get predictions from our GPs
            theta, theta_var = results[f"models/{model_name}/gp_predictions/theta"][()][index].T
            mu_single, mu_single_var = results[f"models/{model_name}/gp_predictions/mu_single"][()][index].T
            sigma_single, sigma_single_var = results[f"models/{model_name}/gp_predictions/sigma_single"][()][index].T
            sigma_multiple, sigma_multiple_var = results[f"models/{model_name}/gp_predictions/sigma_multiple"][()][index].T


            bounds = model_config["bounds"]
            #bounds.setdefault("bounds", [0, 1])
            bounds.update(theta=[0, 1])
            bounds["theta"] = np.array(bounds["theta"]) + [+SMALL, -SMALL]

            # Calculuate 
            args = clipped_predictions(theta, mu_single, sigma_single, sigma_multiple, bounds)

            _slice = (0, m, 0)
            p_single[_slice], ln_s[_slice], ln_m[_slice] = calc_p_single(y, *args, mu_multiple_scalar=model_config["mu_multiple_scalar"])

            # Now calculate draws.
            mu = np.array([theta, mu_single, sigma_single, sigma_multiple])
            cov = np.diag([theta_var, mu_single_var, sigma_single_var, sigma_multiple_var])

            draws = clipped_predictions(*np.random.multivariate_normal(mu, cov, size=D).T, bounds)

            _slice = (0, m, slice(1, None))
            p_single[_slice], ln_s[_slice], ln_m[_slice] = calc_p_single(y, *draws, mu_multiple_scalar=model_config["mu_multiple_scalar"])


        # Now calculate joint probabilities
        ln_s[0, -1, :] = np.sum(ln_s[0, :2, :], axis=0)
        ln_m[0, -1, :] = np.sum(ln_m[0, :2, :], axis=0)

        lp = np.array([ln_s[0, -1, :], ln_m[0, -1, :]])

        p_single[0, -1, :] = np.exp(lp[0] - special.logsumexp(lp, axis=0))


        labels = list(model_names) + ["joint"]

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
        ax = axes[0]
        ax.hist([p_single[0, 0, 1:], p_single[0, 1, 1:], p_single[0, 2, 1:]],
                bins=25, histtype="stepfilled", alpha=0.5)

        ax.legend(labels=labels[::-1])

        ax.set_title(f"Gaia DR2 {source_id} ({SLE})")

        ax = axes[1]
        #ax.hist([ln_s[0, 0, 1:], ln_m[0, :, 1:]])
        # single
        bins = None #np.linspace(-100, -1, 50)

        kw = dict(bins=bins, histtype="stepfilled", alpha=0.5)
        ax.hist([ln_s[0, 0, 1:], ln_s[0, 1, 1:], ln_s[0, 2, 1:]], **kw)

        ax = axes[2]
        #ax.hist([ln_s[0, 0, 1:], ln_m[0, :, 1:]])
        # single
        ax.hist([ln_m[0, 0, 1:], ln_m[0, 1, 1:], ln_m[0, 2, 1:]], **kw)

        xlims = np.array([axes[1].get_xlim(), axes[2].get_xlim()])
        ylims = np.array([axes[1].get_ylim(), axes[2].get_ylim()])

        xlims = (np.min(xlims), np.max(xlims))
        ylims = (np.min(ylims), np.max(ylims))

        #for ax in axes[1:]:
        #    ax.set_xlim(xlims)
            #ax.set_ylim(ylims)

        axes[1].set_title("single")
        axes[2].set_title("ln multiple")

        plt.show()

        return p_single


    p_single = main(source_id)