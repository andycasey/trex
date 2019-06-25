
"""
Evaluate the GP predictions.
"""

import yaml
import h5py as h5

import os
import numpy as np
import logging
from scipy import optimize as op
from hashlib import md5
import sys
from tqdm import tqdm


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
    

    # TODO: store in config
    return_var = True
    batch_size = 15000

    # Load in the results path.
    results_path = os.path.join(pwd, config["results_path"].format(unique_hash=unique_hash))
    results_dir = os.path.dirname(results_path)

    results = h5.File(results_path, "a")

    data_indices = results["indices"]["data_indices"][()]

    for model_name, model_config in config["models"].items():

        lns = list(model_config["kdtree_label_names"]) 
        parameter_names = list(results[f"{model_name}/gp"].keys())

        group_name = f"{model_name}/gp_predictions"
        g = results[group_name] if group_name in results else results.create_group(group_name)

        # Do it for all things that meet the data mask (e.g., finite)    
        xp = np.vstack([sources[ln][()] for ln in lns]).T[data_indices]
        if "source_id" not in g:
            g.create_dataset("source_id", data=sources["source_id"][()][data_indices])

        for parameter_name in parameter_names:

            logger.info(f"Running GP predictions for {model_name}: {parameter_name}")

            # Construct the kernel.
            model_info = results[f"{model_name}/gp/{parameter_name}"]

            X = model_info["X"][()]
            Y = model_info["Y"][()]

            kernel_class = getattr(george.kernels, model_info.attrs["kernel"])
            # TODO: magic hack
            metric = np.var(X, axis=0)
            kernel = kernel_class(metric=metric, ndim=metric.size)

            gp = george.GP(kernel, 
                           mean=np.mean(Y), fit_mean=True,
                           white_noise=np.log(np.std(Y)), fit_white_noise=True)
            for p in gp.parameter_names:
                gp.set_parameter(p, model_info.attrs[p])

            gp.compute(X)

            N = xp.shape[0]

            logger.info(f"Making predictions for {N} sources")

            num_batches = int(np.ceil(N / batch_size))

            # Set up the data set.
            if parameter_name in g:
                del g[parameter_name]

            d = g.create_dataset(parameter_name, data=np.nan * np.ones((N, 2)))

            for i in tqdm(range(num_batches)):
                si, ei = i * batch_size, (i + 1) * batch_size
                if not return_var:
                    d[si:ei, 0] = gp.predict(Y, xp[si:ei], return_cov=False, return_var=False)
                    d[si:ei, 1] = np.inf
                else:
                    d[si:ei, :] = np.vstack(gp.predict(Y, xp[si:ei], return_cov=False, return_var=True)).T

    # Close file.
    results.close()