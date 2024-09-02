import numpy as np
import torch
from tqdm import tqdm

from .. import find_spots as fs
from .. import log
from ..find_spots import detect
from ..setup import NotebookPage
from ..utils import indexing


def find_spots(
    config: dict,
    nbp_basic: NotebookPage,
    nbp_filter: NotebookPage,
) -> NotebookPage:
    """
    Turn each image in the filtered tile directory into a point cloud, saving the results as `spot_details` in the
    `find_spots` notebook page.

    See `'find_spots'` section of `notebook_comments.json` file for description of the variables in the page.

    Args:
        - config (dict): Dictionary obtained from `'find_spots'` section of config file.
        - nbp_basic (NotebookPage): `basic_info` notebook page.
        - nbp_filter (NotebookPage): `filter` notebook page.

    Returns:
        (NotebookPage) nbp_find_spots: `find_spots` notebook page.
    """
    log.debug("Find spots started")

    # Phase 0: Initialisation
    nbp = NotebookPage("find_spots", {"find_spots": config})
    auto_thresh_multiplier = config["auto_thresh_multiplier"]
    if auto_thresh_multiplier <= 0:
        raise ValueError(f"The auto_thresh_multiplier in 'find_spots' config must be positive")
    n_z = np.max([1, nbp_basic.is_3d * nbp_basic.nz])
    if nbp_basic.is_3d is False:
        # set z details to None if using 2d pipeline
        config["radius_z"] = None
        config["isolation_radius_z"] = None
        max_spots = config["max_spots_2d"]
    else:
        max_spots = config["max_spots_3d"]
    INVALID_AUTO_THRESH = -1
    auto_thresh = np.full(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.n_extra_rounds, nbp_basic.n_channels),
        fill_value=INVALID_AUTO_THRESH,
        dtype=float,
    )

    # Phase 1: Load in previous results if they exist
    spot_info = fs.load_spot_info(
        nbp_basic.n_tiles,
        nbp_basic.n_rounds,
        nbp_basic.n_extra_rounds,
        nbp_basic.n_channels,
    )
    # Define use_indices as a [n_tiles x n_rounds x n_channels] boolean array where use_indices[t, r, c] is True if
    # we want to use tile `t`, round `r`, channel `c` to find spots.
    use_indices = np.zeros(
        (nbp_basic.n_tiles, nbp_basic.n_rounds + nbp_basic.use_anchor, nbp_basic.n_channels), dtype=bool
    )
    for t, r, c in indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_bad_trc=False,
    ):
        use_indices[t, r, c] = True

    uncompleted = np.logical_and(use_indices, np.logical_not(spot_info["completed"]))

    # Phase 2: Detect spots on uncompleted tiles, rounds and channels
    with tqdm(
        total=np.sum(uncompleted),
        postfix={"tile": t, "round": r, "channel": c},
        desc=f"Detecting spots on filtered images",
    ) as pbar:
        # Loop over uncompleted tiles, rounds and channels
        for t, r, c in np.argwhere(uncompleted):
            pbar.set_postfix({"tile": t, "round": r, "channel": c})
            image_trc = nbp_filter.images[t, r, c].astype(np.float32)
            image_trc = torch.asarray(image_trc)

            # Compute the image's auto threshold to detect spots.
            mid_z = image_trc.size(2) // 2
            auto_thresh[t, r, c] = float(auto_thresh_multiplier * image_trc[..., mid_z].abs().ravel().median().clip(1))

            local_yxz, spot_intensity = detect.detect_spots(
                image_trc,
                auto_thresh[t, r, c].item(),
                remove_duplicates=True,
                radius_xy=config["radius_xy"],
                radius_z=config["radius_z"],
            )
            local_yxz = local_yxz.numpy().astype(np.int16)
            spot_intensity = spot_intensity.numpy()
            # If r is a reference round, we also get info about whether the spots are isolated
            if r == nbp_basic.anchor_round:
                isolated_spots = fs.get_isolated_spots(
                    local_yxz,
                    config["isolation_radius_xy"],
                    config["isolation_radius_z"],
                )
                spot_info["isolated"] = np.append(spot_info["isolated"], isolated_spots)
            else:
                # if imaging round, only keep the highest intensity spots on each z plane
                local_yxz = fs.filter_intense_spots(local_yxz, spot_intensity, n_z, max_spots)

            # Save results to spot_info
            spot_info["spot_yxz"] = np.vstack((spot_info["spot_yxz"], local_yxz))
            spot_info["spot_no"][t, r, c] = local_yxz.shape[0]
            spot_info["completed"][t, r, c] = True
            assert spot_info["spot_yxz"].shape[0] == np.sum(
                spot_info["spot_no"]
            ), "spot_yxz and spot_no do not match. Tile {}, round {}, channel {}".format(t, r, c)
            pbar.update()

    # Phase 3: Save results to notebook page
    nbp.auto_thresh = auto_thresh
    nbp.spot_yxz = spot_info["spot_yxz"]
    nbp.spot_no = spot_info["spot_no"].astype(np.int32)
    nbp.isolated_spots = spot_info["isolated"]
    log.debug("Find spots complete")

    return nbp
