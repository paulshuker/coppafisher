import os

import numpy as np
import zarr
from tqdm import tqdm

from .. import log
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..stitch import base


def stitch(
    config: ConfigSection, nbp_basic: NotebookPage, nbp_file: NotebookPage, nbp_filter: NotebookPage
) -> NotebookPage:
    """
    Run tile stitching. Tiles are shifted to better align using the DAPI images.

    Args:
        config: stitch config.
        nbp_basic: `basic_info` notebook page.
        nbp_file: `file_names` notebook page.
        nbp_filter: `filter` notebook page.

    Returns:
        new `stitch` notebook page.
    """
    log.debug("Stitch started")
    nbp = NotebookPage("stitch", {config.name: config.to_dict()})

    # TODO: Make non-adjacent tiles have shifts and scores of nan instead of zero to distinguish from true zero
    # shift/scores.

    # Initialize the variables.
    overlap = config["expected_overlap"]
    use_tiles, anchor_round, dapi_channel = list(nbp_basic.use_tiles), nbp_basic.anchor_round, nbp_basic.dapi_channel
    tilepos_yx = nbp_basic.tilepos_yx[use_tiles]

    # Load the tiles.
    tiles = []
    for t in tqdm(use_tiles, total=len(use_tiles), desc="Loading tiles"):
        tile = nbp_filter.images[t, anchor_round, dapi_channel]
        tiles.append(tile)
    tiles = np.array(tiles, np.float32)

    tile_origins_full, pairwise_shifts_full, pairwise_shift_scores_full = base.stitch(
        tiles, tilepos_yx, nbp_basic.use_tiles, nbp_basic.n_tiles, config["expected_overlap"]
    )

    # Fuse the tiles and save the notebook page variables.
    save_path = os.path.join(nbp_file.output_dir, "fused_dapi_image.zarr")
    _ = base.fuse_tiles(
        tiles=tiles,
        tile_origins=tile_origins_full[use_tiles],
        tilepos_yx=tilepos_yx,
        overlap=overlap,
        save_path=save_path,
    )
    nbp.dapi_image = zarr.open_array(save_path, mode="r")
    nbp.tile_origin = tile_origins_full
    nbp.shifts = pairwise_shifts_full
    nbp.scores = pairwise_shift_scores_full

    log.debug("Stitch finished")

    return nbp
