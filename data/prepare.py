
import numpy as np
import os
import h5py as h5
from tqdm import tqdm
from astropy.io import fits
from astropy.table import Table




hdu = 1
memmap = True
basename = "5482"

here = os.path.dirname(__file__)

input_path = os.path.join(here, f"{basename}.fits")
output_path = os.path.join(here, f"{basename}.hdf5")

print(f"Reading in {input_path} and creating {output_path}")

ignore_parameters = [
    "designation", # A string added by CosmoHub. No wonder the file is 11 GB!
    # The 'phot_variable_flag' is all filled with 'NOT_AVAILABLE' and it will
    # kill your CPU if you don't ignore it.
    "phot_variable_flag", 
    "rv_single_epoch_variance",
    "rv_single_epoch_scatter",
    "astrometric_unit_weight_error",
    "p_sb_16",
    "p_sb_50",
    "p_sb_84",
    "absolute_g_mag",
    "absolute_rp_mag",
    "absolute_bp_mag",
    "phot_bp_variability",
    "phot_rp_variability",
    "phot_g_variability",
    "rv_single_epoch_scatter",
    "astrometric_unit_weight_error",
]

print(f"Ignoring parameters: {','.join(ignore_parameters)}")

with h5.File(output_path, "w") as h:

    group = h.create_group("sources")

    with fits.open(input_path, memmap=memmap) as image:

        print("Parameter names:")
        parameter_names = image[hdu].columns.names
        for i, parameter_name in enumerate(parameter_names):
            print(f"{i: >3.0f}: {parameter_name}")
        
        for parameter_name in tqdm(parameter_names):
            if parameter_name not in ignore_parameters: 
                group.create_dataset(parameter_name, data=image[hdu].data[parameter_name])

        # Create additional entries.
        group.create_dataset("j_ast",
                             data=np.sqrt(group["astrometric_chi2_al"][()]/(group["astrometric_n_good_obs_al"][()] - 5)))
        group.create_dataset("j_rv",
                             data=np.sqrt((2/np.pi) * group["rv_nb_transits"][()] * (group["radial_velocity_error"][()]**2 - 0.11**2)))


        mu = 5 * np.log10(group["parallax"][()]/100.0)
        S = np.sqrt(group["astrometric_n_good_obs_al"][()])

        for band in ("g", "bp", "rp"):
            group.create_dataset(f"absolute_{band}_mag", data=group[f"phot_{band}_mean_mag"][()] + mu)
            group.create_dataset(f"phot_{band}_variability",
                                 data=S * group[f"phot_{band}_mean_flux"][()] / group[f"phot_{band}_mean_flux_error"][()])

print(f"Created {output_path}")