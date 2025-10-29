from typing import Any, Callable, Dict, Optional

import numpy as np
import numpy.typing as npt
import tqdm

from ...setup.notebook import NotebookPage


def _default_compute_overlap(
    tile_a: np.ndarray, current_overlap_region: np.ndarray, neighbour_on_left_or_bottom: bool
) -> np.ndarray:
    assert type(tile_a) is np.ndarray
    assert type(current_overlap_region) is np.ndarray
    assert type(neighbour_on_left_or_bottom) is bool

    overlap_size = tile_a.shape[0]
    add_from_current_overlap = np.ones_like(tile_a, bool)
    add_from_current_overlap[-1] = False
    weight = np.linspace(0, 1, overlap_size, endpoint=True, dtype=np.float32)
    if not neighbour_on_left_or_bottom:
        weight = weight[::-1]
        add_from_current_overlap = add_from_current_overlap[::-1]
    tile_a = tile_a.astype(np.float32)
    tile_a *= weight[:, np.newaxis, np.newaxis]
    tile_a = tile_a.astype(current_overlap_region.dtype)

    if current_overlap_region.dtype.kind == "f":
        add_from_current_overlap &= ~np.isnan(current_overlap_region)
    else:
        add_from_current_overlap &= current_overlap_region != np.iinfo(current_overlap_region.dtype).max
    tile_a[add_from_current_overlap] += current_overlap_region[add_from_current_overlap]

    return tile_a


def generate_global_image(
    images: list[np.ndarray],
    tiles_given: list[int],
    nbp_basic: NotebookPage,
    nbp_stitch: NotebookPage,
    output_dtype: npt.DTypeLike = np.float16,
    compute_overlap: Optional[Callable[[np.ndarray, np.ndarray, bool], np.ndarray]] = None,
    compute_overlap_kwargs: Dict[str, Any] | None = None,
    silent: bool = True,
) -> np.ndarray[np.float16]:
    """
    Produce a high-resolution, filtered global background image based on stitch results.

    Args:
        images (list of `(im_y x im_x x im_z) ndarray`): images[i] is the image representing tile index tiles_given[i].
            The list of emptied by the end of the function.
        tiles_given (list of int): tiles_given[i] is the tile index for images[i]. If tiles_given does not contain a
            tile in the notebook and the tile falls into the global image's volume, then that tile's area is set to all
            nans for floating output_dtype or np.iinfo(output_dtype).max for integer output_dtype.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_stitch (NotebookPage): `stitch` notebook page.
        output_dtype (dtype-like, optional): the fused_image datatype. Default: float16. If this is a integer type, then
            the final pixels are rounded to integer values.
        compute_overlap: (callable, optional): the function that computes overlapping regions. The function input is a
            tile's pixel values in the overlapping region (`(n_overlap_size x tile_sz x len(nbp_basic.use_z))
            ndarray[images[0].dtype]`), the current overlapping region values (`(n_overlap_size x tile_sz x
            len(nbp_basic.use_z)) ndarray[output_dtype]`) which is set to nan for floats or the maximum value for
            integers when the current overlapping region is empty. The third argument is whether the tile currently
            being considered is on the left/bottom (bool). The output is a `(n_overlap_size x tile_sz x
            len(nbp_basic.use_z)) ndarray[output_dtype]`. Default: a linear taper from 0 to 1 from one overlap edge to
            the other, giving maximum weighting for a tile when the pixel is closest to its centre.
        compute_overlap_kwargs (dict, optional): additional keyword arguments passed to the compute_weight function.
            Default: none given.
        silent (bool, optional): do not display a progress bar. Default: true.

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
    if compute_overlap is None:
        compute_overlap = _default_compute_overlap
    assert callable(compute_overlap)
    if compute_overlap_kwargs is None:
        compute_overlap_kwargs = {}
    assert type(compute_overlap_kwargs) is dict
    for key in compute_overlap_kwargs:
        assert type(key) is str
    assert type(silent) is bool

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
    output = np.full(
        output_shape, np.nan if np.issubdtype(output_dtype, np.floating) else np.iinfo(output_dtype).max, output_dtype
    )

    for t_i, _ in tqdm.tqdm(enumerate(tiles_given), desc="Generating global image", unit="tile", disable=silent):
        t_image = images.pop(0)
        t_origin = tile_origins_yxz[t_i]
        t_centre = tile_centres_yxz[t_i]
        tile_centres_except_t = np.concat((tile_centres_yxz[:t_i], tile_centres_yxz[t_i + 1 :]), axis=0)

        # Taper along the x and y axes if there is an overlapping tile.
        for dim in (0, 1):
            for neighbour_on_left_or_bottom in (True, False):
                # Positive for right-sided tiles, negative for left-sided tiles.
                tile_distances: np.ndarray[int] = tile_centres_except_t.copy() - t_centre[np.newaxis]
                tile_distances = -tile_distances[:, dim]
                # Really close tile distances are probably aligned along that direction, so remove them.
                # TODO: This can be done more robustly by using the tilepos_yx in nbp_basic.
                tile_distances = tile_distances[np.abs(tile_distances) >= (nbp_basic.tile_sz * 0.5 * expected_overlap)]
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

                ind_min, ind_max = nbp_basic.tile_sz - overlap_size, nbp_basic.tile_sz
                if neighbour_on_left_or_bottom:
                    ind_min, ind_max = 0, overlap_size

                t_image_at_overlap = t_image[:, ind_min:ind_max].swapaxes(0, 1) if dim else t_image[ind_min:ind_max]

                global_ind_min: np.ndarray[int] = t_origin.copy()
                global_ind_min += [0, ind_min, 0] if dim else [ind_min, 0, 0]
                global_ind_max: np.ndarray[int] = global_ind_min.copy()
                global_ind_max += (
                    [nbp_basic.tile_sz, overlap_size, len(nbp_basic.use_z)]
                    if dim
                    else [overlap_size, nbp_basic.tile_sz, len(nbp_basic.use_z)]
                )
                current_overlap_region = output[
                    global_ind_min[0] : global_ind_max[0],
                    global_ind_min[1] : global_ind_max[1],
                    global_ind_min[2] : global_ind_max[2],
                ]
                if dim:
                    current_overlap_region = current_overlap_region.swapaxes(0, 1)
                assert current_overlap_region.shape == t_image_at_overlap.shape

                new_overlap = compute_overlap(
                    t_image_at_overlap, current_overlap_region, neighbour_on_left_or_bottom, **compute_overlap_kwargs
                )
                if type(new_overlap) is not np.ndarray:
                    raise TypeError(f"compute_overlap must return a np.ndarray, got {type(new_overlap)} instead")
                if new_overlap.shape != t_image_at_overlap.shape:
                    raise ValueError(
                        f"compute_overlap must return shape {t_image_at_overlap.shape}, got {new_overlap.shape}"
                    )
                elif new_overlap.dtype != output_dtype:
                    raise ValueError(
                        f"compute_overlap must return ndarray of dtype {output_dtype}, got {new_overlap.dtype}"
                    )

                if dim:
                    t_image[:, ind_min:ind_max] = new_overlap.swapaxes(0, 1)
                else:
                    t_image[ind_min:ind_max] = new_overlap

        t_ind_start = t_origin - min_yxz
        t_ind_end = t_ind_start + tile_shape
        output[t_ind_start[0] : t_ind_end[0], t_ind_start[1] : t_ind_end[1], t_ind_start[2] : t_ind_end[2]] = t_image

    # yxz -> zyx.
    output = output.swapaxes(0, 1).swapaxes(0, 2)

    return output
