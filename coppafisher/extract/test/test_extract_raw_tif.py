import tempfile
from os import path

import numpy as np
import tifffile

from coppafisher.extract import raw_reader, raw_tif
from coppafisher.setup.notebook_page import NotebookPage


def test_raw_tif() -> None:
    n_tiles = 4
    n_rounds = 2
    n_channels = 5
    im_y = 8
    im_x = 9
    im_z = 7

    rng = np.random.RandomState(0)
    all_data = rng.randint(100, size=(n_rounds, n_tiles * n_channels * im_z, im_y, im_x)).astype(np.uint16)

    tif_dir = tempfile.TemporaryDirectory(suffix="coppafisher")
    tif_dir_name = tif_dir.name

    for round in range(n_rounds):
        tifffile.imwrite(path.join(tif_dir_name, f"Round_{round}.tif"), all_data[round], photometric="minisblack")

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.n_tiles = n_tiles
    nbp_basic.n_channels = n_channels
    nbp_basic.use_z = (2, 3, 4, 5, 6)
    nbp_basic.tilepos_yx = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]], int)
    nbp_basic.tilepos_yx_nd2 = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], int)

    nbp_file = NotebookPage("file_names")
    nbp_file.input_dir = tif_dir_name
    nbp_file.round = tuple([f"Round_{r}.tif" for r in range(n_rounds)])
    nbp_file.anchor = "anchor"
    nbp_file.raw_extension = ".tif"

    # This refers to tile index 2 in the raw file tile ordering.
    tile = 1
    round = 1
    channels = [1, 0, 3]

    reader = raw_tif.TifReader()
    assert isinstance(reader, raw_reader.RawReader)

    result = reader.read(nbp_basic, nbp_file, tile, round, channels)

    assert type(result) is np.ndarray
    assert result.ndim == 4
    assert result.shape == (len(channels), im_y, im_x, len(nbp_basic.use_z))

    for c_index, c in enumerate(channels):
        for z_index, z in enumerate(nbp_basic.use_z):
            assert np.allclose(
                result[c_index, :, :, z_index], all_data[round, c + n_channels * z + 2 * n_channels * im_z]
            )

    # This refers to tile index 2 in the raw file tile ordering.
    tile = 1
    round = 0
    channels = [0, 1, 3, 4]

    reader = raw_tif.TifReader()
    assert isinstance(reader, raw_reader.RawReader)

    result = reader.read(nbp_basic, nbp_file, tile, round, channels)

    assert type(result) is np.ndarray
    assert result.ndim == 4
    assert result.shape == (len(channels), im_y, im_x, len(nbp_basic.use_z))

    for c_index, c in enumerate(channels):
        for z_index, z in enumerate(nbp_basic.use_z):
            assert np.allclose(
                result[c_index, :, :, z_index], all_data[round, c + n_channels * z + 2 * n_channels * im_z]
            )

    tif_dir.cleanup()
