python construct_model.py ../config.yml
python construct_gp.py ../results/rc.5/results-5482.hdf5
python evaluate_gp.py ../results/rc.5/results-5482.hdf5
python scratch_convert.py ../results/rc.5/results-5482.hdf5
python evaluate_ratios2.py ../results/rc.5/results-5482.h5
python plots2.py ../results/rc.5/