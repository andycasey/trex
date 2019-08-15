#python construct_model.py ../config.yml
python construct_gp.py ../results/d5537/results-5482.hdf5
python evaluate_gp.py ../results/d5537/results-5482.hdf5
python scratch_convert.py ../results/d5537/results-5482.hdf5
python evaluate_ratios2.py ../results/d5537/results-5482.h5
python plots2.py ../results/d5537/