import numpy as np
import random
import zarr
from skimage import data
from skimage import filters

# make sure we all see the same
np.random.seed(19623)
random.seed(19623)

# open a sample image (channels first)
raw_data = data.astronaut().transpose(2, 0, 1)

# create some dummy "ground-truth" to train on
gt_data = filters.gaussian(raw_data[0], sigma=3.0) > 0.75

# store image in zarr container
f = zarr.open("example_data.zarr", "w")
f["raw"] = raw_data
f["raw"].attrs["resolution"] = (1, 1)
f["ground_truth"] = gt_data
f["ground_truth"].attrs["resolution"] = (1, 1)
