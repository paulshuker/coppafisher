import tempfile
from os import path

import dask.array
import numpy as np

from coppafisher.extract import raw_numpy, raw_reader
from coppafisher.setup.notebook_page import NotebookPage


def test_raw_numpy() -> None:
    n_tiles = 6
    n_rounds = 3
    n_channels = 5
    im_y = 11
    im_x = 13
    im_z = 7

    rng = np.random.RandomState(0)
    all_data = rng.randint(100, size=(n_tiles, n_rounds, n_channels, im_y, im_x, im_z))

    npy_dir = tempfile.TemporaryDirectory(suffix="coppafisher")
    npy_dir_name = npy_dir.name

    dask_chunks = (1, n_channels, im_y, im_x, im_z)

    for r in range(n_rounds):
        r_dir = path.join(npy_dir_name, f"{r}")
        image_dask = dask.array.from_array(all_data[:, r], chunks=dask_chunks)
        dask.array.to_npy_stack(r_dir, image_dask)

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.tilepos_yx = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], int)
    nbp_basic.tilepos_yx_nd2 = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], int)

    nbp_file = NotebookPage("file_names")
    nbp_file.input_dir = npy_dir_name
    nbp_file.round = tuple([f"{r}" for r in range(n_rounds)])
    nbp_file.anchor = "anchor"
    nbp_file.raw_extension = ".npy"

    tile = 1
    round = 2
    channels = [0, 1]

    reader = raw_numpy.NumpyReader()
    assert isinstance(reader, raw_reader.RawReader)
    image = reader.read(nbp_basic, nbp_file, tile, round, channels, z_planes="all")

    assert type(image) is np.ndarray
    assert image.ndim == 4
    assert image.shape == (len(channels), im_y, im_x, im_z)
    for c_i, c in enumerate(channels):
        for z in range(5):
            assert (image[c_i, :, :, z] == all_data[tile, round, c, :, :, z]).all()

    channels = [1, 0, 4]
    reader = raw_numpy.NumpyReader()
    image = reader.read(nbp_basic, nbp_file, tile, round, channels, z_planes="all")

    assert type(image) is np.ndarray
    assert image.ndim == 4
    assert image.shape == (len(channels), im_y, im_x, im_z)
    for c_i, c in enumerate(channels):
        for z in range(5):
            assert (image[c_i, :, :, z] == all_data[tile, round, c, :, :, z]).all()

    npy_dir.cleanup()
