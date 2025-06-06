import os
from typing import List, Optional, Tuple

import numpy as np


def get_tilepos(xy_pos: np.ndarray, tile_sz: int, expected_overlap: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Using `xy_pos` from nd2 metadata, this obtains the yx position of each tile. We label the tiles differently in the
    nd2 and npy formats. In general, there are 2 differences in npy and nd2 tile format:
    1. The same tile in npy and nd2 has a different npy and nd2 index, in nd2 it is the exact same order as the xy pos
    in the metadata, in npy it is ordered from top right to bottom left, moving left along rows and then down rows.
    2. The same tile in npy and nd2 has a different coordinate convention.

    Args:
        xy_pos: `float [n_tiles x 2]`.
            xy position of tiles in pixels. Obtained from nd2 metadata. `xy_pos[t,:]` is xy position of tile `t` where
            `t` is the nd2 tile index.
        tile_sz: xy dimension of tile in pixels.
        expected_overlap: expected overlap between tiles as a fraction of tile_sz.

    Returns:
        tilepos_yx_nd2: `int [n_tiles x 2]`.
            yx position of each tile in nd2 format. `tilepos_yx_nd2[t,:]` is yx position of tile `t` where `t` is the
            nd2 tile index.
        tilepos_yx_npy: `int [n_tiles x 2]`.
            yx position of each tile in npy format. `tilepos_yx_npy[t,:]` is yx position of tile `t` where `t` is the
            npy tile index.
    """
    # since xy_pos is in higher resolution, we need to divide by the expected difference between tiles. May not be
    # exactly the same as tile_sz due to rounding errors, so round to nearest integer
    delta = tile_sz * (1 - expected_overlap)
    xy_pos = np.round(xy_pos / delta).astype(int)

    # get the max x and y values (xy_pos was normalised to make min x and y = 0)
    max_x = np.max(xy_pos[:, 0])
    max_y = np.max(xy_pos[:, 1])

    # Now we need to loop through each tile and assign it a yx index for the nd2 and npy formats
    tilepos_yx_nd2 = np.zeros((xy_pos.shape[0], 2), dtype=int)
    tilepos_yx_npy = np.zeros((xy_pos.shape[0], 2), dtype=int)
    for t in range(xy_pos.shape[0]):
        # nd2 format
        tilepos_yx_nd2[t, 1] = max_x - xy_pos[t, 0]
        tilepos_yx_nd2[t, 0] = max_y - xy_pos[t, 1]

        # npy format
        tilepos_yx_npy[t, 1] = xy_pos[t, 0]
        tilepos_yx_npy[t, 0] = xy_pos[t, 1]

    # now we need to sort the tiles in the npy format. We want to do this decreasing in x and decreasing in y so
    # for if we have a 3 x 3 grid of tiles, we want to sort them as:
    # [2,2], [2,1], [2,0], [1,2], [1,1], [1,0], [0,2], [0,1], [0,0]
    # so we need to sort first by x and then by y
    tilepos_yx_npy = tilepos_yx_npy[tilepos_yx_npy[:, 1].argsort()]
    tilepos_yx_npy = tilepos_yx_npy[tilepos_yx_npy[:, 0].argsort(kind="mergesort")]
    tilepos_yx_npy = tilepos_yx_npy[::-1]

    return tilepos_yx_nd2, tilepos_yx_npy


def get_tile_name(
    tile_directory: str, file_base: List[str], suffix: str, r: int, t: int, c: Optional[int] = None
) -> str:
    """
    Finds the full path to tile, `t`, of particular round, `r`, and channel, `c`, in `tile_directory`.

    Args:
        tile_directory: Path to folder where tiles npy files saved.
        file_base: `str [n_rounds]`.
            `file_base[r]` is identifier for round `r`.
        suffix (str): File name suffix.
        r: Round of desired image.
        t: Tile of desired image.
        c: Channel of desired image.

    Returns:
        Full path of tile file.
    """
    if c is None:
        tile_name = os.path.join(tile_directory, "{}_t{}{}".format(file_base[r], t, suffix))
    else:
        tile_name = os.path.join(tile_directory, "{}_t{}c{}{}".format(file_base[r], t, c, suffix))
    return tile_name


def get_tile_file_names(
    tile_directory_filter: str,
    tile_directory_extract: str,
    file_base: List[str],
    n_tiles: int,
    suffix: str,
    n_channels: int = 0,
    jobs: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets array of all tile file paths which will be saved in tile directory.

    Args:
        tile_directory_filter: path to folder where filtered image files saved.
        tile_directory_unfiltered: path to folder where unfiltered, extract image files will be saved.
        file_base (`(n_rounds) list[str]`): `file_base[r]` is identifier for round `r`.
        n_tiles (int): number of tiles in data set.
        suffix (str): file name suffix. For example, for numpy file saves, will be `'.npy'`.
        n_channels (int, optional): total number of imaging channels if using 3D. `0` if using 2D pipeline as all
            channels saved in same file.
        jobs (bool, optional): set True if file were acquired using JOBs (i.e. tiles are split by laser). Default:
            false.

    Returns:
        - `(n_tiles x n_rounds (x n_channels)) ndarray[str]`. filtered image file paths such that:

            * If 2D so `n_channels = 0`, `tile_files[t, r]` is the full path to npy file containing all channels of
                tile `t`, round `r`.
            * If 3D so `n_channels > 0`, `tile_files[t, r]` is the full path to npy file containing all z-planes of
            tile `t`, round `r`, channel `c`.
        - `(n_tiles x n_rounds (x n_channels)) ndarray[str]`: extract image file paths.
    """
    if not jobs:
        n_rounds = len(file_base)
        if n_channels == 0:
            # 2D
            tile_files_filter = np.zeros((n_tiles, n_rounds), dtype=object)
            tile_files_extract = np.zeros((n_tiles, n_rounds), dtype=object)
            for r in range(n_rounds):
                for t in range(n_tiles):
                    tile_files_filter[t, r] = get_tile_name(tile_directory_filter, file_base, suffix, r, t)
                    tile_files_extract[t, r] = get_tile_name(tile_directory_extract, file_base, suffix, r, t)
        else:
            # 3D
            tile_files_filter = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)
            tile_files_extract = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)
            for r in range(n_rounds):
                for t in range(n_tiles):
                    for c in range(n_channels):
                        tile_files_filter[t, r, c] = get_tile_name(tile_directory_filter, file_base, suffix, r, t, c)
                        tile_files_extract[t, r, c] = get_tile_name(tile_directory_extract, file_base, suffix, r, t, c)
    else:
        n_lasers = 7  # TODO: should have the option to pass n_lasers as an argument for better generalisation
        n_rounds = len(file_base)
        tile_files_filter = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)
        tile_files_extract = np.zeros((n_tiles, n_rounds, n_channels), dtype=object)

        for r in range(n_rounds):

            for t in range(n_tiles):
                raw_tile_files = file_base[r][t * n_lasers : (t + 1) * n_lasers]

                for c in range(n_channels):
                    f_index = int(np.floor(c / 4))
                    t_name = os.path.join(
                        tile_directory_filter, "{}_t{}c{}{}".format(raw_tile_files[f_index], t, c, suffix)
                    )
                    t_name_unfiltered = os.path.join(
                        tile_directory_extract,
                        "{}_t{}c{}{}".format(raw_tile_files[f_index], t, c, suffix),
                    )
                    tile_files_filter[t, r, c] = t_name
                    tile_files_extract[t, r, c] = t_name_unfiltered

    return tile_files_filter, tile_files_extract
