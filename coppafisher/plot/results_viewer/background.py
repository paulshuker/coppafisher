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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "invalid value encountered in divide", RuntimeWarning)
            result /= result.sum(0, keepdims=True)
        result[np.isnan(result)] = 0
        result *= images
        result = result.sum(0)

        if np.issubdtype(images.dtype, np.integer):
            result = np.rint(result)
        result = result.astype(images.dtype)

        return result


class _Region:
    _smallest_mask: np.ndarray[bool]
    _global_shape: Tuple[int, int, int]

    min_yxz: np.ndarray[int]
    max_yxz: np.ndarray[int]

    def get_global_mask(self) -> np.ndarray[bool]:
        global_mask = self._smallest_mask.copy()
        global_mask = np.pad(global_mask, ((self.min_yxz[0], 0), (self.min_yxz[1], 0), (self.min_yxz[2], 0)))
        pad_width = tuple([(0, self._global_shape[i] - self.max_yxz[i]) for i in range(3)])
        global_mask = np.pad(global_mask, pad_width)
        assert global_mask.shape == self._global_shape
        return global_mask

    def set_global_mask(self, global_mask: np.ndarray[bool]) -> None:
        # Find lower and upper bounds of region.
        # Has shape (3 x n_points).
        yxzs = np.array(global_mask.nonzero(), np.int32)
        assert yxzs.shape[0] == 3
        self.min_yxz = yxzs.min(1)
        self.max_yxz = yxzs.max(1) + 1
        self._global_shape = tuple(global_mask.shape)
        self._smallest_mask = np.zeros((self.max_yxz - self.min_yxz).tolist(), bool)
        self._smallest_mask[global_mask[
            self.min_yxz[0]:self.max_yxz[0],
            self.min_yxz[1]:self.max_yxz[1],
            self.min_yxz[2]:self.max_yxz[2],
        ]] = True

    # True in places where the region is on the global image. Not stored in memory, but created when needed.
    global_mask: np.ndarray[bool] = property(get_global_mask, set_global_mask)

    # The image indices that contribute to the region.
    image_indices: List[int]

    def get_shape(self) -> Tuple[int, int, int]:
        return tuple((self.max_yxz - self.min_yxz).tolist())

    shape: Tuple[int, int, int] = property(get_shape)

    def get_tile_mask(self, tile_origin_yxz: np.ndarray[int], tile_shape: Tuple[int, int, int]) -> np.ndarray[bool]:
        """
        Return the mask for the tile that is inside of this region based on the global global_mask.

        Args:
            tile_origin_yxz (`(3) ndarray[int]`): the tile's bottom-left-most corner of the tile.
            tile_shape (tuple[int, int, int]): the tile's shape in y, x, and z directions.

        Returns:
            (`(tile_shape[0] x tile_shape[1] x tile_shape[2]) ndarray[bool]`): tile_mask. True in positions that are
                occupied by the global_mask.
        """
        return self.global_mask[
            tile_origin_yxz[0] : tile_origin_yxz[0] + tile_shape[0],
            tile_origin_yxz[1] : tile_origin_yxz[1] + tile_shape[1],
            tile_origin_yxz[2] : tile_origin_yxz[2] + tile_shape[2],
        ]

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
    for tile_bit_combination in tqdm.tqdm(np.unique(occupancy_grid), desc="Generating global image", disable=silent):
        tile_bit_combination = int(tile_bit_combination)
        if tile_bit_combination == 0:
            continue

        new_region = _Region()
        new_region.global_mask = occupancy_grid == tile_bit_combination
        new_region.image_indices = [tile_ids_inv[tile_id] for tile_id in bits.get_bit_positions(tile_bit_combination)]
        if tile_bit_combination.bit_count() == 1:
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

    for region in non_overlaps:
        assert len(region.image_indices) == 1
        tile_index = tiles_given.index(region.image_indices[0])
        output[region.global_mask] = images[tile_index][region.get_tile_mask(tile_origins_yxz[tile_index], tile_shape)]

    for region in overlaps:
        if 0 in region.shape:
            continue
        region_images = []
        pixel_weights = []

        region_centre_yx = np.array([s / 2 for s in region.shape], np.float32)[:2]
        region_centre_yx += region.min_yxz[:2].astype(np.float32)

        for tile in region.image_indices:
            tile_index = tiles_given.index(tile)
            region_images.append(np.zeros(region.shape, images[0].dtype))
            region_images[-1][region.get_tile_mask(region.min_yxz, region.shape)] = images[tile_index][
                region.get_tile_mask(tile_origins_yxz[tile_index], tile_shape)
            ]
            assert region_images[-1].shape == region.shape

            known_values = np.full(region.shape, np.nan, np.float32)

            tile_centre_to_region_yx = region_centre_yx.copy()
            tile_centre_to_region_yx -= tile_centres_yxz[tile_index, :2].copy()
            # Normalise vector.
            tile_centre_to_region_yx /= np.sqrt(np.square(tile_centre_to_region_yx).sum())

            # True in places where the overlapping region is. Not necessarily a cuboid.
            binary_image = np.zeros(region.shape, bool)
            binary_image[region.get_tile_mask(region.min_yxz, region.shape)] = 1
            binary_image = np.pad(binary_image, 1)
            if np.abs(tile_centre_to_region_yx[0]) >= np.sqrt(2) / 3:
                top_structure = np.zeros((3, 1, 1), bool)
                top_structure[0] = 1
                top_structure[1] = 1
                bottom_structure = np.zeros((3, 1, 1), bool)
                bottom_structure[1] = 1
                bottom_structure[2] = 1
                is_top_edge = scipy.ndimage.binary_hit_or_miss(binary_image, top_structure)
                is_top_edge = is_top_edge[1:-1, 1:-1, 1:-1]
                is_bottom_edge = scipy.ndimage.binary_hit_or_miss(binary_image, bottom_structure)
                is_bottom_edge = is_bottom_edge[1:-1, 1:-1, 1:-1]

                if tile_centre_to_region_yx[0] > 0:
                    known_values[is_top_edge] = 0
                    known_values[is_bottom_edge] = 1
                else:
                    known_values[is_top_edge] = 1
                    known_values[is_bottom_edge] = 0

            if np.abs(tile_centre_to_region_yx[1]) >= np.sqrt(2) / 3:
                left_structure = np.zeros((1, 3, 1), bool)
                left_structure[0, 1] = 1
                left_structure[0, 2] = 1
                right_structure = np.zeros((1, 3, 1), bool)
                right_structure[0, 0] = 1
                right_structure[0, 1] = 1
                is_left_edge = scipy.ndimage.binary_hit_or_miss(binary_image, left_structure)
                is_left_edge = is_left_edge[1:-1, 1:-1, 1:-1]
                is_right_edge = scipy.ndimage.binary_hit_or_miss(binary_image, right_structure)
                is_right_edge = is_right_edge[1:-1, 1:-1, 1:-1]

                if tile_centre_to_region_yx[1] > 0:
                    known_values[is_left_edge] = 1
                    known_values[is_right_edge] = 0
                else:
                    known_values[is_left_edge] = 0
                    known_values[is_right_edge] = 1

            known_values_points = (~np.isnan(known_values)).nonzero()

            # Squeeze dimensions.
            any_dim_is_squeezed = False
            dim = -1
            while dim < (len(known_values_points) - 1):
                dim += 1
                if (known_values_points[dim][0] != known_values_points[dim]).any():
                    continue
                # Squeeze dimension.
                any_dim_is_squeezed = True
                break

            if any_dim_is_squeezed:
                pixel_weights = np.zeros((len(region.image_indices),) + region.shape, np.float32)
                # TODO: Pick the tile to take the region in a smarter way.
                pixel_weights[0] = 1
                break
            else:
                pixel_weight = np.zeros(region.shape, np.float32)
                plane_all_points = np.ones(known_values.shape[:2], bool).nonzero()
                for z in tqdm.trange(region.shape[-1]):
                    plane = known_values[:, :, z].copy()
                    plane_points = (~np.isnan(plane)).nonzero()
                    plane = plane[plane_points]
                    pixel_weight[..., z] = scipy.interpolate.griddata(
                        plane_points, plane, plane_all_points, fill_value=0
                    ).reshape(region.shape[:2], order="C")

                pixel_weight[np.isnan(pixel_weight)] = 0
                pixel_weight = scipy.ndimage.gaussian_filter(pixel_weight, 50, mode="nearest", axes=(0, 1))
                pixel_weight = np.array(pixel_weight, np.float32)
                pixel_weight = pixel_weight.reshape(region.shape, order="C")

            pixel_weights.append(pixel_weight)

        region_images = np.array(region_images, output_dtype)
        pixel_weights = np.array(pixel_weights, np.float32)
        # Make pixel_weights range from 0 to 1.
        pixel_weights -= pixel_weights.min()
        pixel_weights /= pixel_weights.max()
        pixel_weights = np.square(pixel_weights)
        solved_overlap = overlap_solver.solve_overlap(region_images, pixel_weights, **overlap_solver_kwargs)
        output[region.global_mask] = solved_overlap[region.get_tile_mask(region.min_yxz, region.shape)]
        del region_images, pixel_weights

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
