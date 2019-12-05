
import yaml
import h5py as h5

import os
import numpy as np
import logging
from scipy import optimize as op
from hashlib import md5
import matplotlib.pyplot as plt
import sys


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

    # Prepare for figures
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    for model_name, model_config in config["models"].items():

        logger.info(f"Running {model_name} model")

        lns = list(model_config["kdtree_label_names"]) 
        group = results[f"models/{model_name}"]
        source_indices = group["source_indices"][()]
       
        if "gp_model" not in group:
            group.create_group("gp_model", track_order=True)

        gp_group = group["gp_model"]

        for j, parameter_name in enumerate(("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple")):

            if parameter_name == "mu_multiple":
                continue

            logger.info(f"Running model {model_name}:{parameter_name}")
            
            # Fit a gaussian process model.
            is_ok = ~group["is_outlier"][()]
            
            X = np.vstack([sources[ln][()] for ln in lns]).T[source_indices][is_ok]
            Y = group[parameter_name][()][is_ok]

            metric = np.var(X, axis=0)
            kernel = george.kernels.ExpSquaredKernel(metric=metric, ndim=metric.size)
            gp = george.GP(kernel, 
                           mean=np.mean(Y), fit_mean=True,
                           white_noise=np.log(np.std(Y)), fit_white_noise=True)

            def nll(p):
                gp.set_parameter_vector(p)
                ll = gp.log_likelihood(Y, quiet=True)
                return -ll if np.isfinite(ll) else 1e25

            def grad_nll(p):
                gp.set_parameter_vector(p)
                return -gp.grad_log_likelihood(Y, quiet=True)

            gp.compute(X)

            logger.info(f"Initial log likelihood = {gp.log_likelihood(Y):.2f}")
            logger.info(f"Initial grad log likelihood = {gp.grad_log_likelihood(Y)}")
            
            p0 = gp.get_parameter_vector()

            # Check for an initial guess.
            try:
                p0 = np.array(model_config["gp_initial_guess"][parameter_name])

            except KeyError:
                p0 = gp.get_parameter_vector()
                logger.warning(f"Using default initial guess: {p0}")

            else:
                if p0.size != gp.get_parameter_vector().size:
                    logger.warn("Using default initial guess because guess in config file did not match expectations!")
                    p0 = gp.get_parameter_vector()
                else:
                    logger.info(f"Using initial guess from config file: {p0}")

            t_init = time()
            result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
            t_opt = time() - t_init

            gp.set_parameter_vector(result.x)
            logger.info("Result: {}".format(result))
            logger.info("Final logL = {:.2f}".format(gp.log_likelihood(Y)))
            logger.info("Took {:.0f} seconds to optimize".format(t_opt))

            if not result.success:
                logger.warning(f"Optimization of GP for parameter {parameter_name} not successful")

            # TODO Store source_id and source_indices if we excluded some outliers..
            #group.create_dataset("source_id", )

            # Store the gp hyperparameters.
            group_name = f"models/{model_name}/gp_model/{parameter_name}"
            try:
                g = results[group_name]

            except KeyError:
                g = results.create_group(group_name, track_order=True)

            # Store the optimized hyperparameters from the gaussian process
            for key, value in gp.get_parameter_dict().items():
                g.attrs[key] = value

            # Store the name of the kernel
            g.attrs["kernel"] = type(kernel).__name__
            g.attrs["success"] = result.success
            g.attrs["message"] = result.message

            # Store the X and Y used in the GP.
            g.create_dataset("X", data=X)
            g.create_dataset("Y", data=Y)
        
        
    # Close results file
    results.close()
