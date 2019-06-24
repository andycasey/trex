
Download the data and then convert to hdf5 (which also adds columns we need):

python data/prepare.py

Edit the `config.yml` file to your liking.

Run the mixture model:

python code/construct_model.py

Run the GP:

python code/construct_gp.py
