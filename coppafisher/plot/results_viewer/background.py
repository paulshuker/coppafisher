from typing import Literal

import numpy as np

from ...setup.notebook import NotebookPage


def generate_global_image(
    name: Literal["dapi", "anchor"],
    nbp_basic: NotebookPage,
    nbp_filter: NotebookPage,
    nbp_stitch: NotebookPage,
) -> np.ndarray[np.float16]:
    """
    Produce a high-resolution, filtered global background image based on stitch results. This is used for detailed
    background images in the Viewer.

    Args:
        name (str): the background image type. Can be "dapi" or "anchor".
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_filter (NotebookPage): `filter` notebook page.
        nbp_stitch (NotebookPage): `stitch` notebook page.

    Returns:
        (`(big_im_z x big_im_y x big_im_x) ndarray[float16]`): fused_image. The large, global background image. The
            image's origin is relative to `nbp_stitch.tile_origin.min(0)`.
    """
    assert type(name) is str
    assert name in ("dapi", "anchor")
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_filter) is NotebookPage
    assert type(nbp_stitch) is NotebookPage

    n_tiles = len(nbp_basic.use_tiles)
    tile_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    channel = nbp_basic.dapi_channel if name == "dapi" else nbp_basic.anchor_channel
    tile_origins_yxz: np.ndarray = nbp_stitch.tile_origin
    tile_origins_yxz = np.rint(tile_origins_yxz).astype(int)
    tile_centres_yxz = np.rint(tile_origins_yxz + [s / 2 for s in tile_shape]).astype(int)
    min_yxz = tile_origins_yxz.min(0)
    max_yxz = tile_origins_yxz.max(0) + tile_shape
    expected_overlap = nbp_stitch.associated_configs["stitch"]["expected_overlap"]

    output_shape = (max_yxz - min_yxz).tolist()
    output = np.zeros(output_shape, np.float16)
    for t in nbp_basic.use_tiles:
        t_centre = tile_centres_yxz[t]
        tile_centres_except_t = np.full((n_tiles - 1, 3), 0, int)
        tile_centres_except_t[:t] = tile_centres_yxz[:t]
        tile_centres_except_t[t:] = tile_centres_yxz[t + 1 :]

        t_image: np.ndarray = nbp_filter.images[t, nbp_basic.anchor_round, channel].astype(np.float32)

        # Taper along the x and y axes if there is an overlapping tile.
        for dim in (0, 1):
            for on_left_or_bottom in (True, False):
                # Positive for right-sided tiles, negative for left-sided tiles.
                tile_distances: np.ndarray[int] = tile_centres_except_t.copy() - t_centre[np.newaxis]
                tile_distances = tile_distances[:, dim]
                # Really close tile distances are probably aligned along that direction, so remove them.
                # TODO: This can be done more robustly by using the tilepos_yx in nbp_basic.
                tile_distances = tile_distances[np.abs(tile_distances) > (nbp_basic.tile_sz * 0.5 * expected_overlap)]
                if on_left_or_bottom:
                    tile_distances = -tile_distances[tile_distances < 0]
                else:
                    tile_distances = tile_distances[tile_distances > 0]
                if tile_distances.size == 0:
                    continue
                # Take the closest tile distance to decide on the linear taper size.
                closest_tile_distance: int = tile_distances.min().item()
                taper_size: int = (nbp_basic.tile_sz - closest_tile_distance) // 2
                if taper_size < 2:
                    # No taper required.
                    continue
                multiplier = np.linspace(0, 1, taper_size, endpoint=False, dtype=np.float32)
                if on_left_or_bottom:
                    ind_min, ind_max = 0, taper_size
                else:
                    ind_min, ind_max = nbp_basic.tile_sz - taper_size, nbp_basic.tile_sz
                    multiplier = multiplier[::-1]

                if dim == 0:
                    t_image[ind_min:ind_max] *= multiplier[:, np.newaxis, np.newaxis]
                else:
                    t_image[:, ind_min:ind_max] *= multiplier[np.newaxis, :, np.newaxis]

        t_image = t_image.astype(np.float16)

        t_origin = tile_origins_yxz[t]
        t_ind_start = t_origin - min_yxz
        t_ind_end = (t_ind_start + tile_shape).tolist()
        output[t_ind_start[0] : t_ind_end[0], t_ind_start[1] : t_ind_end[1], t_ind_start[2] : t_ind_end[2]] += t_image

    # yxz -> zyx.
    output = output.swapaxes(0, 1).swapaxes(0, 2)

    return output
