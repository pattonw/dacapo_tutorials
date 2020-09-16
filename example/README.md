# DaCapo example:

## 0) Setup
It is recommended that you create a conda environment:

[`conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

## 1) Installation
DaCapo has some non-python dependencies. If boost is not yet installed, please install via

`sudo apt install libboost-all-dev`

DaCapo depends on some packages such as `waterz` that have build requirements such as `cython` and `numpy`. Install these first.

`pip install -r build-requirements.txt`

Then install the remaining requirements, including dacapo itself:

`pip install -r requirements.txt`

Finally install torch as described [here](https://pytorch.org/get-started/locally/)

## 2) Get some data
If you have access to the Janelia File System, there is some example data that you can copy using scp or your prefered copying tool from

`example/path` -> `./data/example_data/example_data.zarr`.

Otherwise you will need to obtain a dataset. The dataset is expected to have the following form:

1) Zarr container named `example_data.zarr`.
2) This container should contain 2 groups: `raw`, `validation`.
3) Both `raw`, and `validation` should contain two arrays, `raw`, `gt`.

## 2.1) Visualize your data
If you would like to inspect your data before training on it, use

`python visualize_data.py data/example_data/example_data.zarr`

## 3) Data Storage
Modify the mongodb storage in `dacapo.conf`:

after `mongo_db_host` provide the url to a mongodb to which you have access

after `mongo_db_name` provide a name to store your dacapo data. For example: `dacapo_v0.1`

## 4) Run DaCapo
`python run.py -r example.conf`

## 4.1) Visualize training data
DaCapo saves snapshots during training as well as validation results. To visualize these use

`python visualize_data.py runs/{run_name}/...`

## 5) Visualize Results
After DaCapo has finished running, you can visualize your results. You can use the `plot_results.ipynb` jupyter notebook (make sure you use a kernel with your virtual environment) or you can run `python plot_results.py`.