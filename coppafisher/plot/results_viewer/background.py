import numpy as np
import numpy.typing as npt
import tqdm

from ...setup.notebook import NotebookPage


def generate_global_image(
    images: list[np.ndarray],
    tiles_given: list[int],
    nbp_basic: NotebookPage,
    nbp_stitch: NotebookPage,
    output_dtype: npt.DTypeLike = np.float16,
    silent: bool = True,
) -> np.ndarray[np.float16]:
    """
    Produce a high-resolution, filtered global background image based on stitch results.

    Args:
        images (list of `(im_y x im_x x im_z) ndarray`): a list of images for each given tile. The list of emptied by
            the end of the function.
        tiles_given (list of int): tiles_given[i] is the tile index for images[i]. The list of emptied by the end of the
            function.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_stitch (NotebookPage): `stitch` notebook page.
        output_dtype (dtype-like, optional): the fused_image datatype. Default: float16. If this is a integer type, then
            the final pixels are rounded to integer values.
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

    tiles_given = tiles_given.copy()
    tile_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    tile_origins_yxz: np.ndarray = nbp_stitch.tile_origin[nbp_basic.use_tiles]
    tile_origins_yxz = np.floor(tile_origins_yxz).astype(int)
    tile_centres_yxz = np.rint(tile_origins_yxz + [s // 2 for s in tile_shape]).astype(int)
    # Inclusive.
    min_yxz = tile_origins_yxz.min(0)
    # Exclusive.
    max_yxz = tile_origins_yxz.max(0) + tile_shape
    expected_overlap = nbp_stitch.associated_configs["stitch"]["expected_overlap"]

    output_shape = (max_yxz - min_yxz).tolist()
    output = np.zeros(output_shape, output_dtype)
    for t_i, t in enumerate(tqdm.tqdm(nbp_basic.use_tiles, desc="Fusing images", unit="tile", disable=silent)):
        t_index = tiles_given.index(t)
        tiles_given.remove(t)
        t_centre = tile_centres_yxz[t_i]
        tile_centres_except_t = np.concat((tile_centres_yxz[:t], tile_centres_yxz[t + 1 :]), axis=0)

        t_image = images.pop(t_index).astype(np.float32)

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
                multiplier = np.linspace(0, 1, overlap_size, endpoint=True, dtype=np.float32)
                if neighbour_on_left_or_bottom:
                    ind_min, ind_max = 0, overlap_size
                else:
                    ind_min, ind_max = nbp_basic.tile_sz - overlap_size, nbp_basic.tile_sz
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
