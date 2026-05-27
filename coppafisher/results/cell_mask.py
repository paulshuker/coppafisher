import warnings
from typing import Iterable

import numpy as np
import scipy
import tqdm

from ..plot.results_viewer import background
from ..setup.notebook_page import NotebookPage


def merge_cell_masks(
    cell_mask_file_paths: Iterable[str],
    cell_mask_origin_yxzs: Iterable[Iterable[float | int]],
    merge_cells_method: str = "",
) -> np.ndarray:
    """
    Merge chunked cell masks for PciSeq.

    The given cell masks are for separate tiles and adjacent tiles have a tile overlap.

    Args:
        cell_mask_file_paths (iterable of str): every cell mask's file path. The cell mask must be saved as .npy files
            with (im_z x im_y x im_x) shape and np.uint16 dtype.
        cell_mask_origin_yxzs (iterable of iterables of three floats): cell_mask_origin_yxzs[i] is an iterable
            containing three floats for the ith cell mask's bottom-leftmost position relative to the other cell masks.
        merge_cells_method (str, optional): the method used for dealing with tile overlaps. If set to "", then the pixel
            values for the tile with the closest centre are always taken and no attempt at cell merging is made. If set
            to "merge 0.5" then cells in at the midpoint between the overlapping tiles are merged together into one cell
            if the overlapping region is at least 50% for either one of the cells. The merging can cascade. Therefore,
            one cell can continuously grow and hoover up overlapping cells. The number 0.5 can be changed to any value
            between 0 and 1. Default: "".

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
    nbp_stitch = NotebookPage("stitch")
    nbp_stitch.tile_origin = np.array(cell_mask_origin_yxzs_list, np.float32)

    merge_cells_method = merge_cells_method.lower()
    if not merge_cells_method:
        overlap_solver = _NoMerge()
        overlap_solver_kwargs = None
    elif len(merge_cells_method.split()) == 2 and merge_cells_method.startswith("merge "):
        overlap_solver = _Merge()
        overlap_solver_kwargs = {"overlap_threshold": float(merge_cells_method.split()[1])}
    else:
        raise ValueError(f"Unknown merge_cells_method: {merge_cells_method}")

    merged_cell_mask = background.generate_global_image(
        tile_masks,
        nbp_basic.use_tiles,
        nbp_basic,
        nbp_stitch,
        np.int32,
        overlap_solver=overlap_solver,
        overlap_solver_kwargs=overlap_solver_kwargs,
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


class _Merge:
    def solve_overlap(
        self, images: np.ndarray, pixel_weights: np.ndarray[np.float32], overlap_threshold: float
    ) -> np.ndarray:
        assert type(overlap_threshold) is float
        assert 0 <= overlap_threshold <= 1

        # First, take cells from the tile with the largest pixel weights (nearest tile centre) except cells at the
        # boundaries where the closest tile changes.
        result = _NoMerge().solve_overlap(images, pixel_weights)
        max_weight_indices = np.argmax(pixel_weights, 0)

        def _boundary_check(image_subset: np.ndarray) -> int:
            return 0 if (image_subset[0] == image_subset).all() else 1

        is_boundary = scipy.ndimage.generic_filter(max_weight_indices, _boundary_check, mode="nearest", size=3)
        is_boundary = is_boundary.astype(bool)
        del max_weight_indices
        for cell_num in np.unique(result[is_boundary]):
            result[result == cell_num] = 0

        # Second, go through cells at the boundaries and place them if they are in unoccupied space. If they are already
        # occupied, then merging occurs if the overlap_threshold is met. If merging condition is not met, then the cell
        # is placed in unoccupied pixels only.
        unused_cell_num = result.max() + 1
        where_merge_already_happened = np.full(images.shape[1:], False, bool)
        is_boundary = is_boundary[np.newaxis]
        is_boundary = np.repeat(is_boundary, images.shape[0], axis=0)
        images_cell_numbers = [np.unique(im) for im in images]
        pbar = tqdm.tqdm(desc="Merging cells", total=sum(im.size for im in images_cell_numbers), unit="cell")
        for image, image_cell_numbers in zip(images, images_cell_numbers, strict=True):
            for cell_num in image_cell_numbers:
                cell_num = int(cell_num)
                if not cell_num or where_merge_already_happened[image == cell_num].all():
                    pbar.update()
                    continue

                index = 0
                # Place the new cell on background regions, then deal with overlapping regions.
                result[np.logical_and(image == cell_num, result == 0)] = unused_cell_num
                other_cell_nums = np.unique(result[image == cell_num])
                while other_cell_nums.size and index < other_cell_nums.size:
                    other_cell_num = int(other_cell_nums[index])
                    if not other_cell_num:
                        index += 1
                        continue

                    # Calculate the overlap fraction.
                    overlap = np.logical_and(image == cell_num, result == other_cell_num).sum()
                    overlap /= np.min([(image == cell_num).sum(), (result == other_cell_num).sum()])
                    if overlap < overlap_threshold:
                        index += 1
                        continue

                    # Merge cells.
                    result[image == cell_num] = unused_cell_num
                    result[result == other_cell_num] = unused_cell_num
                    where_merge_already_happened[result == other_cell_num] = True
                    other_cell_nums = np.concat((other_cell_nums[:index], other_cell_nums[index + 1 :]), axis=0)
                    index = 0

                unused_cell_num += 1

        pbar.close()
        return result


class _NoMerge:
    def solve_overlap(self, images: np.ndarray, pixel_weights: np.ndarray[np.float32]) -> np.ndarray:
        # Keeps the value for the image with the largest pixel weight (closest tile centre).
        max_indices = np.argmax(pixel_weights, axis=0)
        return np.take_along_axis(images, max_indices[np.newaxis, ...], axis=0)[0]
