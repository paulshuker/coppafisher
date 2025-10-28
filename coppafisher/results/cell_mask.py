import math as maths
import warnings
from typing import Iterable

import numpy as np
import tqdm

from ..plot.results_viewer import background
from ..setup.notebook_page import NotebookPage


def _compute_overlap_no_merge(
    tile_a: np.ndarray, current_overlap_region: np.ndarray, neighbour_on_left_or_bottom: bool
) -> np.ndarray:
    overlap_size = tile_a.shape[0]
    if current_overlap_region.dtype.kind != "f":
        tile_a = tile_a.round()
    tile_a = tile_a.astype(current_overlap_region.dtype)
    tile_a_region = np.zeros_like(tile_a, bool)
    if neighbour_on_left_or_bottom:
        tile_a_region[maths.ceil(overlap_size / 2) :] = True
    else:
        tile_a_region[: maths.ceil(overlap_size / 2)] = True

    current_overlap_region[tile_a_region] = tile_a[tile_a_region]
    return current_overlap_region


def _compute_overlap_weight_merge(
    tile_a: np.ndarray, current_overlap_region: np.ndarray, neighbour_on_left_or_bottom: bool, overlap_threshold: float
) -> np.ndarray:
    """
    See merge_cell_masks for details.
    """
    assert type(overlap_threshold) is float
    assert 0 <= overlap_threshold <= 1

    tile_a = tile_a.round().astype(current_overlap_region.dtype)
    overlap_size = tile_a.shape[0]
    if (current_overlap_region == np.iinfo(current_overlap_region.dtype).max).all():
        # There are no overlapping cells yet.
        return tile_a.copy()

    # Cells on the overlap midpoint are extended into the other tile's region unless the overlapping region already has
    # a computed overlap there. Therefore, the algorithm works on a first-come first-serve basis for simplicity. But, if
    # there is a cell already occupying space and the overlap between the cells is above overlap_threshold, then the
    # cells are combined into one cell.
    result = current_overlap_region.copy()
    midpoint = [maths.ceil(overlap_size / 2) - 1, maths.ceil(overlap_size / 2)]
    tile_a_midpoint_cell_numbers = np.unique(tile_a[midpoint].ravel())
    tile_a_midpoint_cell_numbers = tile_a_midpoint_cell_numbers[tile_a_midpoint_cell_numbers > 0]
    current_midpoint_cell_numbers = np.unique(current_overlap_region[midpoint].ravel())
    current_midpoint_cell_numbers = current_midpoint_cell_numbers[current_midpoint_cell_numbers > 0]

    if neighbour_on_left_or_bottom:
        tile_a_cell_numbers = np.unique(tile_a[maths.ceil(overlap_size / 2) :].ravel())
    else:
        tile_a_cell_numbers = np.unique(tile_a[: maths.ceil(overlap_size / 2)].ravel())
    # The tile_a takes cell precedence in places closer to the tile_a centre.
    # But, this does not apply for any cells that touch the midpoint, which will be dealt with separately.
    for cell_num in tile_a_cell_numbers:
        if int(cell_num) == 0:
            continue
        if (cell_num == tile_a_midpoint_cell_numbers).any():
            # Deal with potential cell merges after.
            continue
        for curr_cell_num in np.unique(result[tile_a == cell_num].ravel()):
            if int(curr_cell_num) == 0:
                continue
            if (curr_cell_num == current_midpoint_cell_numbers).any():
                continue
            result[result == curr_cell_num] = 0
        result[tile_a == cell_num] = cell_num

    where_merge_already_happened = np.zeros_like(tile_a, bool)
    pbar = tqdm.tqdm(desc="Merging cells", total=tile_a_midpoint_cell_numbers.size, unit="cell")
    for cell_num in tile_a_midpoint_cell_numbers:
        cell_num = int(cell_num)
        all_cell_positions: np.ndarray[bool] = tile_a == cell_num
        if where_merge_already_happened[all_cell_positions].all():
            pbar.update()
            continue
        cell_size: int = all_cell_positions.sum().item()
        overlapping_cell_nums: np.ndarray[int] = np.unique(
            result[np.logical_and(all_cell_positions, result > 0)].ravel()
        )
        result[np.logical_and(all_cell_positions, result == 0)] = cell_num
        index = 0
        while overlapping_cell_nums.size and index < overlapping_cell_nums.size:
            other_cell_num: int = overlapping_cell_nums.item(index)
            other_cell_size: int = (result == other_cell_num).sum().item()
            overlap_fraction = np.logical_and(all_cell_positions, result == other_cell_num).sum().item()
            overlap_fraction /= min(cell_size, other_cell_size)
            if overlap_fraction < overlap_threshold:
                index += 1
                continue

            # Merge the two cells.
            result[result == other_cell_num] = cell_num

            # Update where the cell is.
            all_cell_positions = result == cell_num
            where_merge_already_happened[all_cell_positions] = True
            cell_size = all_cell_positions.sum().item()
            overlapping_cell_nums = np.unique(result[np.logical_and(all_cell_positions, result > 0)].ravel())
            if overlapping_cell_nums.size == 1 and overlapping_cell_nums[0] == other_cell_num:
                break
            index = 0

        pbar.update()
    pbar.close()

    return result


def merge_cell_masks(
    cell_mask_file_paths: Iterable[str],
    cell_mask_origin_yxzs: Iterable[Iterable[float | int]],
    expected_tile_overlap: float,
    merge_cells_method: str = "",
) -> np.ndarray:
    """
    Merge chunked cell masks for PciSeq.

    The given cell masks are for separate tiles and adjacent tiles have a tile overlap.

    Args:
        cell_mask_file_paths (iterable of str): every cell mask's file path. The cell mask must be saved as .npy files
            with (im_z x im_y x im_x) shape and np.uint16 dtype.
        cell_mask_origin_yxzs (iterable of iterables of three floats): cell_mask_origin_yxzs[i] is an iterable containing
            three floats for the ith cell mask's bottom-leftmost position relative to the other cell masks.
        expected_tile_overlap (float): the approximate tile overlap expected as a fraction of the tile's length in the
            x/y direction.
        merge_cells_method (str, optional): the method used for dealing with tile overlaps. If set to "", then the pixel
            values for the tile with the closest centre are always taken and no attempt at cell merging is made. If set
            to "merge 0.5" then two cells are merged together into one cell if the overlapping region is at least 50%
            for either one of the cells. The number 0.5 can be changed to any value between 0 and 1. Default: "".

    Returns:
        (`(big_im_z x big_im_y x big_im_x) ndarray[uint16]`): merged_cell_mask. The merged cell mask.
    """
    cell_mask_file_paths_list: list[str] = []
    cell_mask_origin_yxzs_list: list[list[int]] = []
    for cell_mask_file_path in cell_mask_file_paths:
        cell_mask_file_paths_list.append(str(cell_mask_file_path))
    for cell_mask_origin_yxz in cell_mask_origin_yxzs:
        cell_mask_origin_yxzs_list.append(cell_mask_origin_yxz)
    if len(cell_mask_file_paths_list) <= 1:
        raise ValueError("Must input at least two cell masks")
    if len(cell_mask_file_paths_list) != len(cell_mask_origin_yxzs):
        raise ValueError("cell_mask_file_paths must be the same length as cell_mask_origin_yxzs")
    if type(expected_tile_overlap) is not float:
        raise TypeError(f"expected_tile_overlap must be a float, got {type(expected_tile_overlap)} instead")

    tile_masks: list[np.ndarray] = [np.load(file_path) for file_path in cell_mask_file_paths]
    if any([mask.dtype != np.uint16 for mask in tile_masks]):
        raise ValueError("All cell masks must be np.uint16 datatype")
    if any([mask.shape != tile_masks[0].shape for mask in tile_masks]):
        raise ValueError("All cell masks must be the same shape")
    # ZYX -> YXZ.
    tile_masks = [mask.swapaxes(0, 2).swapaxes(0, 1) for mask in tile_masks]
    tile_masks = [mask.astype(np.int32) for mask in tile_masks]
    # Every cell must be given a unique number, except 0 because that is the label for background.
    for i in range(1, len(tile_masks)):
        shifted_mask = tile_masks[i].copy()
        shifted_mask[shifted_mask > 0] += tile_masks[i - 1].max()
        tile_masks[i] = shifted_mask

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.use_tiles = tuple(range(len(tile_masks)))
    nbp_basic.tile_sz = tile_masks[0].shape[0]
    nbp_basic.use_z = tuple(range(tile_masks[0].shape[2]))
    nbp_stitch = NotebookPage("stitch", {"stitch": {"expected_overlap": expected_tile_overlap}})
    nbp_stitch.tile_origin = np.array(cell_mask_origin_yxzs_list, np.float32)

    merge_cells_method = merge_cells_method.lower()
    if not merge_cells_method:
        compute_overlap = _compute_overlap_no_merge
        compute_overlap_kwargs = None
    elif len(merge_cells_method.split()) == 2 and merge_cells_method.startswith("merge "):
        compute_overlap = _compute_overlap_weight_merge
        compute_overlap_kwargs = {"overlap_threshold": float(merge_cells_method.split()[1])}
    else:
        raise ValueError(f"Unknown merge_cells_method: {merge_cells_method}")

    merged_cell_mask = background.generate_global_image(
        tile_masks,
        nbp_basic.use_tiles,
        nbp_basic,
        nbp_stitch,
        np.int32,
        compute_overlap=compute_overlap,
        compute_overlap_kwargs=compute_overlap_kwargs,
        silent=True,
    )

    # Compress the cell numbers together so they are labelled 1, 2, 3, ...
    where_background_is = merged_cell_mask == 0
    merged_cell_mask[where_background_is] = merged_cell_mask.max()
    _, inverse = np.unique(merged_cell_mask, return_inverse=True, axis=None)
    inverse = inverse.astype(np.int32)
    merged_cell_mask = inverse + 1
    merged_cell_mask[where_background_is] = 0

    if merged_cell_mask[merged_cell_mask == np.iinfo(np.uint16).max].sum():
        warnings.warn(
            "Merged cell mask contains a cell number at the largest value possible. Overflow may have occurred.",
            UserWarning,
            1,
        )

    merged_cell_mask = merged_cell_mask.astype(np.uint16)
    return merged_cell_mask
