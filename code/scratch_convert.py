

import h5py as h5
import sys

old_path = sys.argv[1]
new_path = sys.argv[1].replace(".hdf5", ".h5")

#old_path = "../results/rc.1/results-5482.hdf5"
#new_path = "../results/rc.1/results-5482.h5"

old = h5.File(old_path, "r")
new = h5.File(new_path, "w")

sources = h5.File("../data/5482.hdf5", "r")

for k in ("config", "config_path"):
    new.attrs[k] = old.attrs[k]

models = new.create_group("models", track_order=True)

for model_name in ("ast", "rv"):

    model = models.create_group(model_name, track_order=True)

    # MIXTURE MODEL
    mixture_model = model.create_group("mixture_model", track_order=True)

    data_indices = old[model_name]["data_indices"][()] # refers to SOURCES
    npm_indices = old[model_name]["npm_indices"][()] # refers to DATA_INDICES
    source_indices = data_indices[npm_indices]

    mixture_model.create_dataset("source_id", data=sources["sources"]["source_id"][()][source_indices])
    mixture_model.create_dataset("source_indices", data=source_indices)
    for k in ("theta", "mu_single", "sigma_single", "sigma_multiple"):
        mixture_model.create_dataset(k, data=old[model_name]["mixture_model"][k][()])

    is_ok = old[model_name]["mixture_model"]["is_ok"][()]
    mixture_model.create_dataset("is_outlier_or_on_edge", data=~is_ok)

    # GP MODEL
    # Should be the same sizes as sum(~is_outlier_or_on_edge)

    gp_model = model.create_group("gp_model", track_order=True)
    gp_source_indices = source_indices[is_ok]
    gp_model.create_dataset("source_id", data=sources["sources"]["source_id"][()][gp_source_indices])
    gp_model.create_dataset("source_indices", data=gp_source_indices)

    for k in ("theta", "mu_single", "sigma_single", "sigma_multiple"):
        f = gp_model.create_group(k, track_order=True)
        f.attrs.update(dict(old["{model_name}/gp/{k}"].attrs))

        for u in "XY":
            f.create_dataset(u, data=old[model_name]["gp"][k][u][()])


    # GP PREDICTIONS
    gp_predictions = model.create_group("gp_predictions", track_order=True)
    gp_predictions.create_dataset("source_id", data=sources["sources"]["source_id"][()][data_indices])
    gp_predictions.create_dataset("source_indices", data=data_indices)
    

    for k in ("theta", "mu_single", "sigma_single", "sigma_multiple"):
        gp_predictions.create_dataset(k, data=old[f"{model_name}/gp_predictions/{k}"][()])


old.close()
new.close()

sources.close()