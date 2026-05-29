import warnings
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import numpy.typing as npt
import scipy
import tqdm

from ...setup.notebook import NotebookPage
from ...utils import bits


class SolvesOverlap(Protocol):
    def solve_overlap(self, images: np.ndarray, pixel_weights: np.ndarray[np.float32], **kwargs) -> np.ndarray:
        """
        Solve the overlap region between tile images.

        Args:
            images (`(region_count x r_y x r_x x r_z) ndarray`): the region_count number of images that are shared in
                region r.
            pixel_weights (`(region_count x r_y x r_x x r_z) ndarray[float32]`): pixel_weights[i] is the linear
                weightings for each pixel value for images[i] pixels. The weights range from 0 to 1. Ones are placed on
                the edges closes to the tile's centre.
            **kwargs (dict[str, any]): additional keyword arguments.

        Returns:
            (`(r_y x r_x x r_z) ndarray[images.dtype]`): solved_region. The resulting overlap region.
        """
        ...


class _LinearInterpolator:
    def solve_overlap(self, images: np.ndarray, pixel_weights: np.ndarray) -> np.ndarray:
        result = pixel_weights.copy()

        if images.shape[0] == 4:
            import napari
            v = napari.Viewer()
            v.add_image(pixel_weights.copy().transpose((0, 3, 1, 2)), name="Initial weights")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
            result /= result.sum(0, keepdims=True)
        result[np.isnan(result)] = 0
        result *= images
        result = result.sum(0)

        if images.shape[0] == 4:
            v.add_image(result.copy().transpose((2, 0, 1)), name="Final")

        if np.issubdtype(images.dtype, np.integer):
            result = np.rint(result)
        result = result.astype(images.dtype)

        return result


class _Region:
    min_yxz: np.ndarray[int]
    max_yxz: np.ndarray[int]
    # The image indices that contribute to the region.
    image_indices: List[int]

    def get_shape(self) -> Tuple[int, int, int]:
        return tuple([self.max_yxz.item(i) - self.min_yxz.item(i) for i in range(3)])

    shape: Tuple[int, int, int] = property(get_shape)

    def overlaps_with(self, min_yxz_1: np.ndarray[int], max_yxz_1: np.ndarray[int]) -> bool:
        assert min_yxz_1.shape == (3,)
        assert max_yxz_1.shape == (3,)

        return all([self._dim_overlaps(min_yxz_1.item(i), max_yxz_1.item(i), i) for i in range(3)])

    def _dim_overlaps(self, min_1: int, max_1: int, dim: int) -> bool:
        # Min in region.
        res = (min_1 >= self.min_yxz[dim] and min_1 < self.max_yxz[dim]).item()
        # Max in region.
        res = res or (max_1 > self.min_yxz[dim] and max_1 <= self.max_yxz[dim]).item()
        # Region engulfed.
        res = res or (min_1 <= self.min_yxz[dim] and max_1 >= self.max_yxz[dim]).item()
        return res


def generate_global_image(
    images: List[np.ndarray],
    tiles_given: List[int],
    nbp_basic: NotebookPage,
    nbp_stitch: NotebookPage,
    output_dtype: npt.DTypeLike = np.float16,
    unbound_value: int | float | None = None,
    overlap_solver: Optional[SolvesOverlap] = None,
    overlap_solver_kwargs: Dict[str, Any] | None = None,
    silent: bool = True,
) -> np.ndarray:
    """
    Stitch together given images.

    The images are tiles and are stitched together based on the stitch results provided for positioning each tile. By
    default, the tile overlap is resolved by using linear interpolation.

    Args:
        images (list of `(im_y x im_x x im_z) ndarray`): images[i] is the image representing tile index tiles_given[i].
        tiles_given (list of int): tiles_given[i] is the tile index for images[i]. If tiles_given does not contain a
            tile in the notebook and the tile falls into the global image's volume, then that area is set to
            unbound_value.
        nbp_basic (NotebookPage): `basic_info` notebook page.
        nbp_stitch (NotebookPage): `stitch` notebook page.
        output_dtype (dtype-like, optional): the fused_image datatype. Default: float16.
        unbound_value (number, optional): pixels are set to unbound_value when the pixel is out of bounds from all
            tiles. Default: 0.
        overlap_solver: (SolvesOverlap, optional): solver for overlapping regions in the image. See SolvesOverlap
            protocol above for implementation details. Default: take information from every overlapping tile linearly
            weighted by the pixel's distance away from the tile's centre.
        overlap_solver_kwargs (dict[str, any], optional): additional keyword arguments passed to the overlap solver.
            Default: none given.
        silent (bool, optional): do not display a progress bar. Default: true.

    Returns:
        (`(big_im_z x big_im_y x big_im_x) ndarray[output_dtype]`): fused_image. The large, global background image. The
            image's origin is relative to `nbp_stitch.tile_origin.min(0)`.

    Raises:
        NotImplementedError: if there are too many given tiles.
    """
    assert type(images) is list
    assert all([type(image) is np.ndarray for image in images])
    assert type(tiles_given) is list
    if len(tiles_given) > 256:
        raise NotImplementedError()
    assert all([type(tile) is int for tile in tiles_given])
    assert len(set(tiles_given)) == len(tiles_given)
    assert len(tiles_given) == len(images)
    assert type(nbp_basic) is NotebookPage
    assert type(nbp_stitch) is NotebookPage
    if unbound_value is None:
        unbound_value = 0
    if overlap_solver is None:
        overlap_solver = _LinearInterpolator()
    assert callable(overlap_solver.solve_overlap)
    if overlap_solver_kwargs is None:
        overlap_solver_kwargs = {}
    assert type(overlap_solver_kwargs) is dict
    for key in overlap_solver_kwargs:
        assert type(key) is str
    assert type(silent) is bool

    tile_shape = (nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    tile_origins_yxz: np.ndarray = nbp_stitch.tile_origin[tiles_given].astype(np.float32)

    # The tile origins are shifted so that min tile origin is 0 for consistency with spot yxz positions.
    minimum_tile_origin_indices = np.argmin(tile_origins_yxz, 0)
    temp = [tile_origins_yxz[minimum_tile_origin_indices[i], i].copy() for i in range(3)]
    for i in range(3):
        tile_origins_yxz[:, i] -= temp[i]
    del temp

    tile_centres_yxz = tile_origins_yxz.copy().astype(np.float32)
    tile_centres_yxz += np.array([s / 2 for s in tile_shape], np.float32)[np.newaxis]
    tile_origins_yxz = np.rint(tile_origins_yxz).astype(int)

    # Inclusive.
    min_yxz = tile_origins_yxz.min(0)
    assert np.isclose(min_yxz, 0).all()
    # Exclusive.
    max_yxz = tile_origins_yxz.max(0) + tile_shape

    output_shape = (max_yxz - min_yxz).tolist()

    # 1) Find all unique overlapping regions.
    # Each tile's region adds a 1 bit to every pixel in occupancy_grid that it occupies.
    # Then, each unique overlapping region is found.
    occupancy_grid: np.ndarray = np.zeros(output_shape, _get_int_required(len(tiles_given)))
    tile_ids = {tile: i for i, tile in enumerate(tiles_given)}
    tile_ids_inv = {i: tile for i, tile in enumerate(tiles_given)}
    for i, tile in enumerate(tiles_given):
        t_origin = tile_origins_yxz[i]
        t_max_yxz = t_origin.copy() + tile_shape
        occupancy_grid[
            t_origin[0] : t_max_yxz[0],
            t_origin[1] : t_max_yxz[1],
            t_origin[2] : t_max_yxz[2],
        ] |= (
            1 << tile_ids[tile]
        )
    non_overlaps: List[_Region] = []
    overlaps: List[_Region] = []
    for tile_bit_combination in np.unique(occupancy_grid):
        if tile_bit_combination == 0:
            continue

        # Find lower and upper bounds of region.
        # Has shape (3 x n_points).
        yxzs = np.array((occupancy_grid == tile_bit_combination).nonzero(), np.int32)
        assert yxzs.shape[0] == 3
        new_region = _Region()
        new_region.min_yxz = yxzs.min(1)
        new_region.max_yxz = yxzs.max(1) + 1
        del yxzs

        bit_positions = bits.get_bit_positions(tile_bit_combination.item())
        new_region.image_indices = [tile_ids_inv[tile_id] for tile_id in bit_positions]
        if tile_bit_combination.item().bit_count() == 1:
            non_overlaps.append(new_region)
        else:
            overlaps.append(new_region)
    del tile_ids, tile_ids_inv

    # Sort regions such that the most overlapping regions are placed last.
    if overlaps:
        overlaps.sort(key=lambda region: len(region.image_indices))
        assert len(overlaps[0].image_indices) <= len(overlaps[-1].image_indices)

    # 2) Populate the global image with tiles, including overlapping regions.
    max_distance_from_centre = np.array([s / 2 for s in tile_shape], np.float32)
    max_distance_from_centre = np.sqrt(np.square(max_distance_from_centre).sum())
    output = np.full(output_shape, unbound_value, output_dtype)
    pbar = tqdm.tqdm(desc="Generating global image", disable=silent, total=len(non_overlaps) + len(overlaps))

    for region in non_overlaps:
        assert len(region.image_indices) == 1
        tile_index = tiles_given.index(region.image_indices[0])
        tile_min_yxz = region.min_yxz - tile_origins_yxz[tile_index]
        tile_max_yxz = tile_min_yxz.copy() + region.shape
        output[
            region.min_yxz[0] : region.max_yxz[0],
            region.min_yxz[1] : region.max_yxz[1],
            region.min_yxz[2] : region.max_yxz[2],
        ] = images[tile_index][
            tile_min_yxz[0] : tile_max_yxz[0],
            tile_min_yxz[1] : tile_max_yxz[1],
            tile_min_yxz[2] : tile_max_yxz[2],
        ]
        pbar.update()

    for region in overlaps:
        region_images = []
        pixel_weights = []

        region_centre_yx = np.array([s / 2 for s in region.shape], np.float32)[:2]
        region_centre_yx += region.min_yxz[:2].astype(np.float32)

        for tile in region.image_indices:
            tile_index = tiles_given.index(tile)
            tile_min_yxz = region.min_yxz - tile_origins_yxz[tile_index]
            tile_max_yxz = tile_min_yxz.copy() + region.max_yxz - region.min_yxz
            region_images.append(
                images[tile_index][
                    tile_min_yxz[0] : tile_max_yxz[0],
                    tile_min_yxz[1] : tile_max_yxz[1],
                    tile_min_yxz[2] : tile_max_yxz[2],
                ]
            )
            assert region_images[-1].shape == region.shape

            # Place zeros at the furthest away edge(s) to the tile's centre.
            # Place ones at the closest edge(s) to the tile's centre.
            # Then fill in the blanks using linear interpolation.
            # Do this in 2d then repeat up in the z stack.
            known_values = np.full(region.shape[:2], np.nan, np.float32)

            tile_centre_to_region_yx = region_centre_yx.copy()
            tile_centre_to_region_yx -= tile_centres_yxz[tile_index, :2].copy()
            # Normalise vector.
            tile_centre_to_region_yx /= np.sqrt(np.square(tile_centre_to_region_yx).sum())

            if len(region.image_indices) == 4:
                print("")
                print(f"{tile=}")
                print(f"{tile_centre_to_region_yx=}")

            # TODO: This is working with big overlaps. But, sometimes there is a tiny slither (1 pixel width) overlap
            # volume nearby to the big 4 tile overlap that looks ugly and broken. Maybe we should simply select a tile
            # to take full control in very thin volume cases and try not to taper since it is not possible with a single
            # pixel axis.

            if np.abs(tile_centre_to_region_yx[0]) >= np.sqrt(2) / 3:
                # y is closest to tile centre, while y_other is the furthest.
                y = 0 if tile_centre_to_region_yx[0] > 0 else -1
                y_other = -1 if tile_centre_to_region_yx[0] > 0 else 0
                known_values[y] = 1
                known_values[y_other] = 0
                if len(region.image_indices) == 4:
                    print("Y")
            if np.abs(tile_centre_to_region_yx[1]) >= np.sqrt(2) / 3:
                # x is closest to tile centre, while x_other is the furthest.
                x = 0 if tile_centre_to_region_yx[1] > 0 else -1
                x_other = -1 if tile_centre_to_region_yx[1] > 0 else 0
                known_values[:, x] = 1
                known_values[:, x_other] = 0
                if len(region.image_indices) == 4:
                    print("X")

            if len(region.image_indices) == 4:
                print("")

            known_values_points = (~np.isnan(known_values)).nonzero()
            known_values = known_values[known_values_points]
            all_points = np.ones(region.shape[:2]).nonzero()

            # Squeeze dimensions since griddata does not work when a dimension is a single value.
            dim = -1
            while dim < (len(known_values_points) - 1):
                dim += 1
                if (known_values_points[dim][0] != known_values_points[dim]).any():
                    continue
                # Squeeze dimension.
                known_values_points = tuple([values for i, values in enumerate(known_values_points) if i != dim])
                all_points = tuple([values for i, values in enumerate(all_points) if i != dim])
                dim = -1

            pixel_weight = scipy.interpolate.griddata(known_values_points, known_values, all_points, fill_value=0)
            pixel_weight[np.isnan(pixel_weight)] = 0
            pixel_weight = np.array(pixel_weight, np.float32)
            pixel_weight = pixel_weight.reshape(region.shape[:2], order="C")
            # Repeat along the z stack.
            pixel_weight = np.repeat(pixel_weight[:, :, np.newaxis], region.shape[-1], 2)

            pixel_weights.append(pixel_weight)

        region_images = np.array(region_images, output_dtype)
        pixel_weights = np.array(pixel_weights, np.float32)
        solved_overlap = overlap_solver.solve_overlap(region_images, pixel_weights, **overlap_solver_kwargs)
        assert solved_overlap.shape == region_images.shape[1:]
        output[
            region.min_yxz[0] : region.max_yxz[0],
            region.min_yxz[1] : region.max_yxz[1],
            region.min_yxz[2] : region.max_yxz[2],
        ] = solved_overlap
        del region_images, pixel_weights
        pbar.update()

    pbar.close()

    # yxz -> zyx.
    output = output.swapaxes(0, 1).swapaxes(0, 2)

    return output


def _get_int_required(bit_count: int) -> npt.DTypeLike:
    if bit_count <= 32:
        return np.int32
    elif bit_count <= 64:
        return np.int64
    elif bit_count <= 128:
        return np.int128
    elif bit_count <= 256:
        return np.int256
    else:
        raise ValueError()
