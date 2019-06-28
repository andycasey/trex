
import yaml
import h5py as h5

import os
import numpy as np
import logging
from scipy import (optimize as op, integrate, special)
from hashlib import md5
import sys


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



    # Load data.
    data_path = os.path.join(pwd, config["data_path"])
    data = h5.File(data_path, "r")
    sources = data["sources"]
    

    # Load in the results path.
    results_path = os.path.join(pwd, config["results_path"].format(unique_hash=unique_hash))
    results_dir = os.path.dirname(results_path)

    results = h5.File(results_path, "a")


    data_indices = results["indices"]["data_indices"][()]
    npm_indices = results["indices"]["npm_indices"][()]

    for model_name, model_config in config["models"].items():
        logger.info(f"Running {model_name} model")

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

        w = results[f"{model_name}/gp_predictions/theta"][()][:, 0]

        ln_s = g.create_dataset("single", data=np.nan * np.ones(N))

        mu_s = results[f"{model_name}/gp_predictions/mu_single"][()][:, 0]
        sigma_s = results[f"{model_name}/gp_predictions/sigma_single"][()][:, 0]

        print(model_name, np.isfinite(w).sum(), np.isfinite(mu_s).sum(), np.isfinite(sigma_s).sum())

        # Evaluate log-likelihood
        ln_s[:] = np.log(w) + utils.normal_lpdf(y, mu_s, sigma_s)

        ln_m = g.create_dataset("multiple", data=np.nan * np.ones(N))
        mu_m = results[f"{model_name}/gp_predictions/mu_multiple"][()][:, 0]
        sigma_m = results[f"{model_name}/gp_predictions/sigma_multiple"][()][:, 0]

        ln_m[:] = np.log(1-w) + utils.lognormal_lpdf(y, mu_m, sigma_m)

        lp = np.array([ln_s[()], ln_m[()]]).T
        
        with np.errstate(under="ignore"):
            ratio = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))    

        g.create_dataset("ratio_single", data=ratio)

        # Do many draws...


        """
        # Calculate evidences.
        for i in range(20000):
            print(i)

            f = lambda x: utils.normal_lpdf(y, x[0], x[1])

            foo = integrate.quad(lambda _: utils.normal_lpdf(y[i], mu_s[i], _),
                                 sigma_s[i] - 3, sigma_s[i] + 3)



            mu_s_std = np.sqrt(results[f"{model_name}/gp_predictions/mu_single"][()][i, 1])
            sigma_s_std = np.sqrt(results[f"{model_name}/gp_predictions/sigma_single"][()][i, 1])

            mu_s_std = 5
            sigma_s_std = 5

            vals = []
            def f(s, m):
                vals.append([m, s])
                return (2 * np.pi * s**2)**(-0.5) * np.exp(-(y[i] - m)**2/(2*s**2))

            C = 10


            bar = integrate.dblquad(f,
                                    0.5, 5,
                                    lambda m: 0.025, 
                                    lambda m: 1.0)




            mu_m_std = np.sqrt(results[f"{model_name}/gp_predictions/mu_multiple"][()][i, 1])
            sigma_m_std = np.sqrt(results[f"{model_name}/gp_predictions/sigma_multiple"][()][i, 1])

            g = lambda a, b: utils.lognormal_lpdf(y[i], b, a)

            def g(m, s):
                #print("ms", m, s)
                return (y[i] * s * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((np.log(y[i]) - m)/s)**2)


            lower = np.log(mu_s[i] + 5 * sigma_s[i])
            
            bar2 = integrate.dblquad(g,
                                     0.1, 2,
                                     lambda s: lower + s**2,
                                     lambda s: np.inf)

            ratio = bar2[0]/bar[0]

            print(f"{model_name} at {i}: ratio {ratio:.1f} because {y[i]:.1f} > {mu_s[i]:.1f} + {sigma_s[i]:.1f}")

            if ratio > 10:
                raise a

            # Calculate using Laplace approximations and unconstrained prior.


        raise a 
        """


    # Calculate joint ratios.
    model_names = list(config["models"].keys())
    if len(model_names) > 1:

        # Calculate joint likelihood ratio.
        ln_s = np.sum([results[f"model_selection/likelihood/{mn}/single"][()] for mn in model_names],
                      axis=0)
        ln_m = np.sum([results[f"model_selection/likelihood/{mn}/multiple"][()] for mn in model_names],
                      axis=0)

        lp = np.array([ln_s, ln_m]).T
        with np.errstate(under="ignore"):
            ratio = np.exp(lp[:, 0] - special.logsumexp(lp, axis=1))

        dataset_name = "model_selection/likelihood/joint_ratio_single"
        if dataset_name in results:
            del results[dataset_name]

        results.create_dataset(dataset_name, data=ratio)

    results.close()
