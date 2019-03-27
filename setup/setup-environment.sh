#wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
# install that and activate
conda env create -n py37
sudo apt-get install libeigen3-dev
conda install -n py37 --file environment.yml
conda activate py37
pip install george
