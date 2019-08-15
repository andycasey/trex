
import yaml
import h5py as h5

import os
import numpy as np
import logging
from scipy import optimize as op
from hashlib import md5
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


overwrite = True

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

        data_indices = results[model_name]["data_indices"][()]
        npm_indices = results[model_name]["npm_indices"][()]

        lns = list(model_config["kdtree_label_names"]) 

        # Get the mixture model results.
        group = results[f"{model_name}/mixture_model"]

        gp_group_name = f"{model_name}/gp"
        if gp_group_name not in results:
            results.create_group(gp_group_name)


        for j, parameter_name in enumerate(("theta", "mu_single", "sigma_single", "mu_multiple", "sigma_multiple")):

            # Check to see if we should skip mu_multiple
            # TODO
            if parameter_name == "mu_multiple": continue

            logger.info(f"Running model {model_name}:{parameter_name}")

            if parameter_name in results[f"{model_name}/gp"] and not overwrite:
                logger.warn(f"GP already exists for {model_name}:{parameter_name} -- skipping")
                continue
            
            is_ok = group["is_ok"][()]

            # Fit a gaussian process model.
            X = np.vstack([sources[ln][()] for ln in lns]).T[data_indices][npm_indices][is_ok]

            # TODO: Optionally do log of the param.
            Y = group[parameter_name][()][is_ok]

            # TODO: Only include those that are 'is_ok'?


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

            # Store the gp hyperparameters.
            group_name = f"{model_name}/gp/{parameter_name}"
            try:
                g = results[group_name]

            except KeyError:
                g = results.create_group(group_name)

            # Store the optimized hyperparameters from the gaussian process
            for key, value in gp.get_parameter_dict().items():
                g.attrs[key] = value

            # Store the name of the kernel
            g.attrs["kernel"] = type(kernel).__name__
            g.attrs["success"] = result.success
            g.attrs["message"] = result.message

            # TODO: Store the X, Y used to compute the GP.
            # In principle if we aren't taking a subset of X, Y then we don't need to do this!
            for _ in "XY":
                if _ in g:
                    del g[_]
            
            g.create_dataset("X", data=X)
            g.create_dataset("Y", data=Y)

            if not result.success:
                logger.warn(f"Optimization of GP for parameter {parameter_name} not successful")

    # Close results file
    results.close()
