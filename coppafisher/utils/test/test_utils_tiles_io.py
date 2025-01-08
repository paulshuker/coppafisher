import os
import tempfile

import numpy as np

from coppafisher.utils import tiles_io


def test_save_load_image():
    temp_dir = tempfile.TemporaryDirectory("coppafisher_utils_tiles_io")
    directory = os.path.join(temp_dir.name, "unit_test_dir")

    if not os.path.isdir(directory):
        os.mkdir(directory)
    rng = np.random.RandomState(0)
    array_1_shape = (3, 3, 4)
    array_1 = rng.rand(*array_1_shape).astype(dtype=np.float16)
    array_1_path = os.path.join(directory, "array_1.zarr")

    tiles_io._save_image(array_1, array_1_path)

    array_1_returned = tiles_io._load_image(array_1_path)
    assert np.allclose(array_1_returned, array_1)

    temp_dir.cleanup()


# TODO: get_npy_tile_ind unit tests.
