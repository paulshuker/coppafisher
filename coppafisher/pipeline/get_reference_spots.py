import os

import numpy as np
import torch
import zarr

from .. import log
from ..setup.notebook_page import NotebookPage
from ..spot_colours import base as spot_colours_base
from ..utils import duplicates


def get_reference_spots(
    nbp_basic: NotebookPage,
    nbp_file: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_find_spots: NotebookPage,
    nbp_register: NotebookPage,
    nbp_stitch: NotebookPage,
) -> NotebookPage:
    """
    This takes each spot found on the reference round/channel and computes the corresponding intensity
    in each of the imaging rounds/channels.

    Args:
        nbp_basic: `basic_info` notebook page.
        nbp_file: `file_names` notebook page.
        nbp_filter: `filter` notebook page.
        nbp_find_spots: 'find_spots' notebook page.
        nbp_register: `register` notebook page.
        nbp_stitch: `stitch` notebook page.

    Returns:
        `NotebookPage[ref_spots]` - Page containing intensity of each reference spot on each imaging round/channel.
    """
    log.info("Get ref spots started")
    # Create a notebook page for ref_spots which stores information like local coords, tile_no of each spot and more.
    nbp = NotebookPage("ref_spots")
    # The code is going to loop through all tiles, as we expect some anchor spots on each tile but r and c should stay
    # fixed as the value of the reference round and reference channel
    r = nbp_basic.anchor_round
    c = nbp_basic.anchor_channel
    use_tiles, use_rounds, use_channels = np.array(nbp_basic.use_tiles), nbp_basic.use_rounds, nbp_basic.use_channels

    tile_origins = nbp_stitch.tile_origin.astype(np.float32)
    tile_centres = duplicates.get_tile_centres(nbp_basic.tile_sz, len(nbp_basic.use_z), tile_origins)
    tile_origins = torch.from_numpy(tile_origins)

    # all means all spots found on the reference round / channel
    all_local_yxz = np.zeros((0, 3), dtype=np.int16)
    all_local_tile = np.zeros(0, dtype=np.int16)

    # Loop through tiles and record the local_yxz spots on this tile.
    # We then append this to all_local_yxz and all_local_tile arrays.
    for t in nbp_basic.use_tiles:
        t_local_yxz = nbp_find_spots.spot_yxz[f"t{t}r{r}c{c}"][:]
        if np.shape(t_local_yxz)[0] == 0:
            continue

        # Find duplicate spots as those detected with closest tile centre not the tile they are assigned to.
        t_global_yxz = torch.from_numpy(t_local_yxz).to(torch.float32)
        t_global_yxz += tile_origins[[t]]
        is_duplicate = duplicates.is_duplicate_spot(t_global_yxz, t, tile_centres).numpy()
        del t_global_yxz
        log.debug(f"{is_duplicate.sum()} duplicate spots found on tile {t}")
        t_local_yxz = t_local_yxz[~is_duplicate]

        all_local_yxz = np.append(all_local_yxz, t_local_yxz, axis=0)
        all_local_tile = np.append(all_local_tile, np.full(t_local_yxz.shape[0], t, dtype=np.int16))

    # Only save used rounds/channels initially
    n_use_rounds, n_use_channels, n_use_tiles = len(use_rounds), len(use_channels), len(use_tiles)
    spot_colours = np.zeros((0, n_use_rounds, n_use_channels), dtype=np.float32)
    local_yxz = np.zeros((0, 3), dtype=np.int16)
    tile = np.zeros(0, dtype=np.int16)
    log.info("Reading in spot_colours for ref_round spots")
    for t in nbp_basic.use_tiles:
        in_tile = all_local_tile == t
        if np.sum(in_tile) == 0:
            continue
        log.info(f"Tile {np.where(use_tiles==t)[0][0]+1}/{n_use_tiles}")
        log.debug(f"Tile {t} has {all_local_yxz[in_tile].shape[0]} reference spots")
        colours = spot_colours_base.get_spot_colours_new_safe(
            nbp_basic,
            image=nbp_filter.images,
            flow=nbp_register.flow,
            affine=nbp_register.icp_correction,
            tile=t,
            yxz=all_local_yxz[in_tile],
            use_rounds=use_rounds,
            use_channels=use_channels,
        )
        valid = ~(np.isnan(colours).any(1).any(1))
        # Zeros in all channels for any round is an invalid spot.
        valid &= ~(np.isclose(colours, 0).all(2).any(1))
        log.debug(f"Valid ref pixel colours: {valid.sum()} out of {valid.size} for tile {t}")
        spot_colours = np.append(spot_colours, colours[valid], axis=0)
        local_yxz = np.append(local_yxz, all_local_yxz[in_tile][valid], axis=0)
        tile = np.append(tile, np.ones(valid.sum(), dtype=np.int16) * t)

    # Convert the numpy results to zarrays for saving.
    kwargs = dict(chunks=False, zarr_version=2)
    local_yxz_store = zarr.ZipStore(os.path.join(nbp_file.output_dir, "local_yxz.zarray"), mode="w")
    local_yxz = zarr.array(local_yxz, store=local_yxz_store, **kwargs)
    local_yxz_store.close()
    tile_store = zarr.ZipStore(os.path.join(nbp_file.output_dir, "tile.zarray"), mode="w")
    tile = zarr.array(tile, store=tile_store, **kwargs)
    tile_store.close()
    spot_colours_store = zarr.ZipStore(os.path.join(nbp_file.output_dir, "spot_colours.zarray"), mode="w")
    spot_colours = zarr.array(spot_colours, store=spot_colours_store, **kwargs)
    spot_colours_store.close()

    # save spot info to notebook
    nbp.local_yxz = local_yxz
    nbp.tile = tile
    nbp.colours = spot_colours
    log.info("Get ref spots complete")

    return nbp
