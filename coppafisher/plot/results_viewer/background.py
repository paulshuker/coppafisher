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
                weightings for each pixel value for images[i] pixels. The weights range from 0 to 1. A 1 represents the
                pixel at the centre of the images tile.
            **kwargs (dict[str, any]): additional keyword arguments.

        Returns:
            (`(r_y x r_x x r_z) ndarray[images.dtype]`): solved_region. The resulting overlap region.
        """
        ...


class _LinearInterpolator:
    def solve_overlap(self, images: np.ndarray, pixel_weights: np.ndarray) -> np.ndarray:
        # At the boundary, the image with the most weighting entirely wins. Then a linear weighting is used on all other
        # values within the cuboid region by grid linear interpolation.
        pixel_weight_max_indices = np.argmax(pixel_weights, axis=0)
        known_values = np.full_like(images, np.nan, np.float32)

        # Set boundary indices
        known_values[:, :, :, 0] = 0  # y, x, 0
        known_values[:, :, :, -1] = 0  # y, x, -1

        # Set the known values at max indices for y,x boundaries
        y_indices, x_indices = np.indices((images.shape[1], images.shape[2]))
        known_values[pixel_weight_max_indices[y_indices, x_indices, 0], y_indices, x_indices, 0] = 1
        known_values[pixel_weight_max_indices[y_indices, x_indices, -1], y_indices, x_indices, -1] = 1

        # Set boundaries for y,z (x=0 and x=-1)
        known_values[:, :, 0, :] = 0  # x=0
        known_values[:, :, -1, :] = 0  # x=-1

        # Set known values for y,z boundaries
        y_indices, z_indices = np.indices((images.shape[1], images.shape[3]))
        known_values[pixel_weight_max_indices[y_indices, 0, z_indices], y_indices, 0, z_indices] = 1
        known_values[pixel_weight_max_indices[y_indices, -1, z_indices], y_indices, -1, z_indices] = 1

        # Set boundaries for x,z (y=0 and y=-1)
        known_values[:, 0, :, :] = 0  # y=0
        known_values[:, -1, :, :] = 0  # y=-1

        # Set known values for x,z boundaries
        x_indices, z_indices = np.indices((images.shape[2], images.shape[3]))
        known_values[pixel_weight_max_indices[0, x_indices, z_indices], 0, x_indices, z_indices] = 1
        known_values[pixel_weight_max_indices[-1, x_indices, z_indices], -1, x_indices, z_indices] = 1

        del pixel_weight_max_indices

        # C ordering of indices.
        known_values_points = (~np.isnan(known_values)).nonzero()
        known_values = known_values[known_values_points]

        all_points = np.ones_like(images).nonzero()
        result = scipy.interpolate.griddata(known_values_points, known_values, all_points)
        assert result.size == all_points[0].size
        result = result.reshape(images.shape, order="C")
        result /= result.sum(0)
        result *= images
        result = result.sum(0)

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
        return tuple([self.max_yxz[i] - self.min_yxz[i] for i in range(3)])

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

    tile_centres_yxz = np.rint(tile_origins_yxz + [s / 2 for s in tile_shape]).astype(int)
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

    # 2) Populate the global image with tiles, including overlapping regions.
    max_distance_from_centre = np.array([s / 2 for s in tile_shape], np.float32)
    max_distance_from_centre = np.sqrt(np.square(max_distance_from_centre).sum())
    output = np.full(output_shape, unbound_value, output_dtype)
    pbar = tqdm.tqdm(desc="Generating global image", disable=silent, total=len(non_overlaps) + len(overlaps))
    for region in overlaps:
        region_images = []
        pixel_weights = []

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
            pixel_weight = np.array(
                np.meshgrid(
                    np.linspace(0, region.shape[0] - 1, region.shape[0]),
                    np.linspace(0, region.shape[1] - 1, region.shape[1]),
                    np.linspace(0, region.shape[2] - 1, region.shape[2]),
                    indexing="ij",
                ),
                np.float32,
            )
            pixel_weight += (region.min_yxz - tile_centres_yxz[tile_index])[:, np.newaxis, np.newaxis, np.newaxis]
            # Finds distance from tile's centre.
            pixel_weight = np.sqrt(np.square(pixel_weight).sum(0))
            pixel_weight /= max_distance_from_centre
            pixel_weight = 1 - pixel_weight
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

    for region in non_overlaps:
        assert len(region.image_indices) == 1
        tile_index = tiles_given.index(region.image_indices[0])
        tile_min_yxz = region.min_yxz - tile_origins_yxz[tile_index]
        tile_max_yxz = tile_min_yxz.copy() + region.max_yxz - region.min_yxz
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
