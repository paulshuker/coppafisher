from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import tqdm

from ...setup.notebook import NotebookPage


def _default_compute_overlap_weight(tile_a: np.ndarray, tile_b: np.ndarray) -> np.ndarray[np.float32]:
    assert type(tile_a) is np.ndarray
    assert type(tile_b) is np.ndarray
    assert tile_a.shape == tile_b.shape

    overlap_size = tile_a.shape[0]
    return np.linspace(0, 1, overlap_size, endpoint=True, dtype=np.float32)


def generate_global_image(
    images: list[np.ndarray],
    tiles_given: list[int],
    nbp_basic: NotebookPage,
    nbp_stitch: NotebookPage,
    output_dtype: npt.DTypeLike = np.float16,
    compute_overlap_weight: Optional[
        Callable[[np.ndarray[np.float32], np.ndarray[np.float32]], np.ndarray[np.float32]]
    ] = None,
    silent: bool = True,
) -> np.ndarray[np.float16]:
    """
    Produce a high-resolution, filtered global background image based on stitch results.

    Args:
        images (list of `(im_y x im_x x im_z) ndarray`): images[i] is the image representing tile index tiles_given[i].
            The list of emptied by the end of the function.
        tiles_given (list of int): tiles_given[i] is the tile index for images[i]. If tiles_given does not contain a
            tile in the notebook, then that tile's area is set to all zeros.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_stitch (NotebookPage): `stitch` notebook page.
        output_dtype (dtype-like, optional): the fused_image datatype. Default: float16. If this is a integer type, then
            the final pixels are rounded to integer values.
        compute_overlap_weight: (callable, optional): the function that computes the weighting that is applied to a
            tile in the overlapping region. The function input is one tile's pixel values in the overlapping region
            (`(n_overlap_size x tile_sz x len(nbp_basic.use_z)) ndarray[float32]`), then the other tile's pixel values
            in the overlapping region (`(overlap_size x tile_sz x len(nbp_basic.use_z)) ndarray[float32]`). The output
            multiplier must be a `(n_overlap_size) ndarray[float32]`. Default: a linear decrease from 1 to 0 from one
            edge to the other.
        silent (bool, optional): do not print a progress bar. Default: true.

    Returns:
        (`(big_im_z x big_im_y x big_im_x) ndarray[output_dtype]`): fused_image. The large, global background image. The
            image's origin is relative to `nbp_stitch.tile_origin.min(0)`.
    """
    assert type(images) is list
    assert all([type(image) is np.ndarray for image in images])
    assert type(tiles_given) is list
    assert all([type(tile) is int for tile in tiles_given])
    assert len(set(tiles_given)) == len(tiles_given)
    assert len(tiles_given) == len(images)
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_stitch) is NotebookPage
    if compute_overlap_weight is None:
        compute_overlap_weight = _default_compute_overlap_weight
    assert callable(compute_overlap_weight)
    assert type(silent) is bool

    tiles_given = tiles_given.copy()
    tile_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    tile_origins_yxz: np.ndarray = nbp_stitch.tile_origin[tiles_given]

    # The lowest x/y/z tile origin values are floored down for consistency with the spot yxz positions.
    minimum_tile_origin_indices = np.argmin(tile_origins_yxz, 0)
    for i in range(3):
        tile_origins_yxz[minimum_tile_origin_indices[i]] = np.floor(tile_origins_yxz[minimum_tile_origin_indices[i]])

    tile_origins_yxz = np.rint(tile_origins_yxz).astype(int)
    tile_centres_yxz = np.rint(tile_origins_yxz + [s // 2 for s in tile_shape]).astype(int)
    # Inclusive.
    min_yxz = tile_origins_yxz.min(0)
    # Exclusive.
    max_yxz = tile_origins_yxz.max(0) + tile_shape
    expected_overlap = nbp_stitch.associated_configs["stitch"]["expected_overlap"]

    output_shape = (max_yxz - min_yxz).tolist()
    output = np.zeros(output_shape, output_dtype)
    for t_i, t in enumerate(tqdm.tqdm(tiles_given, desc="Generating global image", unit="tile", disable=silent)):
        if t not in tiles_given:
            continue
        t_centre = tile_centres_yxz[t_i]
        tile_centres_except_t = np.concat((tile_centres_yxz[:t], tile_centres_yxz[t + 1 :]), axis=0)

        t_image = images.pop(0).astype(np.float32)

        # Taper along the x and y axes if there is an overlapping tile.
        for dim in (0, 1):
            for neighbour_on_left_or_bottom in (True, False):
                # Positive for right-sided tiles, negative for left-sided tiles.
                tile_distances: np.ndarray[int] = tile_centres_except_t.copy() - t_centre[np.newaxis]
                tile_distances = -tile_distances[:, dim]
                # Really close tile distances are probably aligned along that direction, so remove them.
                # TODO: This can be done more robustly by using the tilepos_yx in nbp_basic.
                tile_distances = tile_distances[np.abs(tile_distances) > (nbp_basic.tile_sz * 0.5 * expected_overlap)]
                if neighbour_on_left_or_bottom:
                    tile_distances = tile_distances[tile_distances > 0]
                else:
                    tile_distances = -tile_distances[tile_distances < 0]
                if tile_distances.size == 0:
                    continue
                # Take the closest tile distance to decide on the linear taper size.
                closest_tile_distance: int = tile_distances.min().item()
                overlap_size: int = nbp_basic.tile_sz - closest_tile_distance
                if overlap_size < 2:
                    # No taper required.
                    continue

                ind_min, ind_max = 0, overlap_size
                b_ind_min, b_ind_max = nbp_basic.tile_sz - overlap_size, nbp_basic.tile_sz
                if not neighbour_on_left_or_bottom:
                    tmp = ind_min, ind_max
                    ind_min, ind_max = b_ind_min, b_ind_max
                    b_ind_min, b_ind_max = tmp

                a_tile = t_image[:, ind_min:ind_max].transpose((1, 0, 2)) if dim else t_image[ind_min:ind_max]
                b_tile = t_image[:, ind_min:ind_max].transpose((1, 0, 2)) if dim else t_image[ind_min:ind_max]
                multiplier = compute_overlap_weight(a_tile, b_tile)
                if type(multiplier) is not np.ndarray:
                    raise TypeError(f"compute_overlap_weight returned unexpected type {type(multiplier)}")
                if multiplier.dtype != np.float32:
                    raise ValueError(
                        f"compute_overlap_weight return np.ndarray of dtype {multiplier.dtype}, expected np.float32"
                    )

                if not neighbour_on_left_or_bottom:
                    multiplier = multiplier[::-1]
                if dim == 0:
                    t_image[ind_min:ind_max] *= multiplier[:, np.newaxis, np.newaxis]
                else:
                    t_image[:, ind_min:ind_max] *= multiplier[np.newaxis, :, np.newaxis]

        if output_dtype in (np.int16, np.int32, np.int64, np.uint16, np.uint32, np.uint64):
            t_image = np.rint(t_image)
        t_image = t_image.astype(output_dtype)

        t_origin = tile_origins_yxz[t_i]
        t_ind_start = t_origin - min_yxz
        t_ind_end = t_ind_start + tile_shape
        output[t_ind_start[0] : t_ind_end[0], t_ind_start[1] : t_ind_end[1], t_ind_start[2] : t_ind_end[2]] += t_image

    # yxz -> zyx.
    output = output.swapaxes(0, 1).swapaxes(0, 2)

    return output
