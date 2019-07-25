
# Data model description

The results of this work are stored in HDF5 file format. The configuration
settings used are stored in attributes of the parent structure, and can be
accessed with:

```python
    import h5py as h5
    import yaml

    fp = h5.File("results.h5", "r")

    config = yaml.load(fp.attrs["config"], Loader=yaml.Loader)
```

The parent structure of the data file contains two groups: `models` and `results`.

The `models` group contains a sub-group for each model name (e.g., `rv` for
radial velocity, `ast` for astrometry). The sub-structure is as follows:
```
    models:
        MODEL_NAME:
            mixture_model:
                source_indices
                theta
                mu_single
                sigma_single
                sigma_multiple
                is_outlier
                is_on_edge

            gp_model:
                source_indices
                theta
                mu_single
                sigma_single
                sigma_multiple

            gp_predictions:
                source_indices 
                theta
                mu_single
                sigma_single
                sigma_multiple
```

The sub-groups are appropriately named:
- `mixture_model`: the two-component mixture model
- `gp_model`: the Gaussian process models that were fit to the results of the mixture model
- `gp_predictions`: the predictions from the Gaussian process models which are used for binary detection and characterisation


# SOURCE INDICES


results:
    source_id
    source_indices

    ll_rv_single
    ll_rv_multiple
    ll_ast_single
    ll_ast_multiple
    ll_single
    ll_multiple

    p_rv_single
    p_ast_single
    p_single

    bf_rv_multiple
    bf_ast_multiple
    bf_multiple

    K
    K_err
    
    # draws and other things
