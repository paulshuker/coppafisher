import numbers
from typing import List, Optional, Tuple, Union

import napari
import numpy as np
from tqdm import tqdm

from ..extract import nd2, raw
from ..setup import file_names
from ..setup.notebook import Notebook


def get_raw_images(
    nb: Notebook, config_path: str, tiles: List[int], rounds: List[int], channels: List[int], use_z: List[int]
) -> np.ndarray:
    """
    This loads in raw images for the experiment corresponding to the *Notebook*.

    Args:
        nb: Notebook for experiment
        tiles: npy (as opposed to nd2 fov) tile indices to view.
            For an experiment where the tiles are arranged in a 4 x 3 (ny x nx) grid, tile indices are indicated as
            below:

            | 2  | 1  | 0  |

            | 5  | 4  | 3  |

            | 8  | 7  | 6  |

            | 11 | 10 | 9  |
        rounds: Rounds to view.
        channels: Channels to view.
        use_z: Which z-planes to load in from raw data.

    Returns:
        `raw_images` - `[len(tiles) x len(rounds) x len(channels) x n_y x n_x x len(use_z)]` uint16 array.
        `raw_images[t, r, c]` is the `[n_y x n_x x len(use_z)]` image for tile `tiles[t]`, round `rounds[r]` and channel
        `channels[c]`.
    """
    n_tiles = len(tiles)
    n_rounds = len(rounds)
    n_channels = len(channels)
    n_images = n_rounds * n_tiles * n_channels
    ny = nb.basic_info.tile_sz
    nx = ny
    nz = len(use_z)

    nbp_file = file_names.get_file_names(nb.basic_info, config_path)
    raw_images = np.zeros((n_tiles, n_rounds, n_channels, ny, nx, nz), dtype=np.uint16)
    with tqdm(total=n_images) as pbar:
        pbar.set_description("Loading in raw data")
        for r in range(n_rounds):
            round_dask_array, _ = raw.load_dask(nbp_file, nb.basic_info, r=rounds[r])
            # TODO: Can get rid of these two for loops, when round_dask_array is always a dask array.
            #  At the moment though, is not dask array when using nd2_reader (On Mac M1).
            for t in range(n_tiles):
                for c in range(n_channels):
                    pbar.set_postfix({"round": rounds[r], "tile": tiles[t], "channel": channels[c]})
                    (raw_images[t, r, c],) = raw.load_image(
                        nbp_file, nb.basic_info, tiles[t], channels[c], round_dask_array, rounds[r], use_z
                    )
                    pbar.update(1)
    return raw_images


def number_to_list(var_list: List) -> Tuple:
    # Converts every value in variables to a list if it is a single number
    # Args:
    #     var_list: List of variables which need converting to list
    #
    # Returns:
    #     var_list with variables converted into list.

    for i in range(len(var_list)):
        if isinstance(var_list[i], numbers.Number):
            var_list[i] = [var_list[i]]
    return tuple(var_list)


def view_tile_layout(
    nb: Notebook,
    config_path=None,
    num_rotations: int = 0,
    tiles: Optional[Union[int, List[int]]] = None,
    channel: int = 0,
):
    """
    Function to view the tile layout in napari. Images will be middle z plane from nd2 files in the anchor round.
    Channel can be specified.
    Args:
        nb: Notebook containing at least basic info and file names.
        num_rotations: Number of 90 degree rotations to apply to each individual tile. These rotations always in the
        direction taking the y axis to the x axis.
        tiles: Which tiles to view. If `None`, will view use_tiles.
        channel: Which channel to view.
    """
    assert channel in nb.basic_info.use_channels or channel == nb.basic_info.dapi_channel, f"Invalid channel: {channel}"
    if config_path is None:
        config_path = nb.config_path

    if tiles is None:
        tiles = nb.basic_info.use_tiles

    raw_images = get_raw_images(
        nb,
        config_path,
        tiles=tiles,
        rounds=[nb.basic_info.anchor_round],
        channels=[channel],
        use_z=[nb.basic_info.nz // 2],
    )[:, 0, 0, :, :, 0]

    # First rotate the images. This makes num_rotations rotations in the direction taking the y axis to the x axis
    raw_images = np.rot90(raw_images, k=num_rotations, axes=(1, 2))
    tiles_nd2 = nd2.get_nd2_tile_ind(tiles, nb.basic_info.tilepos_yx_nd2, nb.basic_info.tilepos_yx)
    tilepos_yx = nb.basic_info.tilepos_yx.copy()
    tilepos_yx_nd2 = nb.basic_info.tilepos_yx_nd2.copy()

    # Now plot
    expected_overlap = nb.get_config()["stitch"]["expected_overlap"]
    tile_sz = nb.basic_info.tile_sz
    yx_step = tile_sz * (1 - expected_overlap)
    viewer = napari.Viewer()
    text_npy_ind = {
        "string": [f"{t}" for t in tiles],
        "color": "red",
        "size": 36,
    }
    text_nd2_ind = {
        "string": [f"{t}" for t in tiles_nd2],
        "color": "cyan",
        "size": 36,
    }
    text_npy_pos = {
        "string": [f"{tilepos_yx[t]}" for t in tiles],
        "color": "red",
        "size": 36,
    }
    text_nd2_pos = {
        "string": [f"{tilepos_yx_nd2[t]}" for t in tiles_nd2],
        "color": "cyan",
        "size": 36,
    }
    point_locs = [tilepos_yx[t] * yx_step + np.array([tile_sz // 2, tile_sz // 2]) for t in tiles]
    for t in range(len(tiles)):
        viewer.add_image(raw_images[t], name=f"Tile {tiles[t]}", translate=tilepos_yx[tiles[t]] * yx_step)
    viewer.add_points(point_locs, text=text_npy_ind, size=0, name="NPY Tile Indices")
    viewer.add_points(point_locs, text=text_nd2_ind, size=0, name="ND2 Tile Indices", visible=False)
    viewer.add_points(point_locs, text=text_npy_pos, size=0, name="NPY YX Positions", visible=False)
    viewer.add_points(point_locs, text=text_nd2_pos, size=0, name="ND2 YX Positions", visible=False)
    viewer.add_vectors(np.array([[0, 0], [0, 1]]) * tile_sz, edge_width=50, edge_color="green", name="Y Axis")
    viewer.add_vectors(np.array([[0, 0], [1, 0]]) * tile_sz, edge_width=50, edge_color="green", name="X Axis")
    axis_text = {
        "string": ["Y", "X"],
        "color": "green",
        "size": 36,
    }
    axis_label_locs = np.array([[tile_sz, -200], [-200, tile_sz]])
    viewer.add_points(axis_label_locs, text=axis_text, size=0, name="Axis Labels")
    viewer.dims.axis_labels = ["y", "x"]

    napari.run()
