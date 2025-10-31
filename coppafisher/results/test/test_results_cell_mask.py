import os
import tempfile

import numpy as np

from coppafisher.results import cell_mask


def test_merge_cell_masks() -> None:
    temp_dir = tempfile.TemporaryDirectory("coppafisher")

    tile_size_y = 5
    tile_size_z = 3
    tile_a = np.zeros((tile_size_z, tile_size_y, tile_size_y), np.uint16)
    tile_b = np.zeros((tile_size_z, tile_size_y, tile_size_y), np.uint16)
    tile_a_file_path = os.path.join(temp_dir.name, "tile_a.npy")
    tile_b_file_path = os.path.join(temp_dir.name, "tile_b.npy")
    np.save(tile_a_file_path, tile_a)
    np.save(tile_b_file_path, tile_b)

    merged_cell_mask = cell_mask.merge_cell_masks([tile_a_file_path, tile_b_file_path], [[0, 0, 0], [4, 4, 1]], 0.3)
    assert type(merged_cell_mask) is np.ndarray
    assert merged_cell_mask.dtype == np.uint16
    assert merged_cell_mask.shape == (tile_size_z + 1, tile_size_y + 4, tile_size_y + 4)

    tile_a[:] = 1
    tile_b[:] = 1
    np.save(tile_a_file_path, tile_a)
    np.save(tile_b_file_path, tile_b)

    merged_cell_mask = cell_mask.merge_cell_masks([tile_a_file_path, tile_b_file_path], [[0, 0, 0], [0, 2, 0]], 0.1)
    assert type(merged_cell_mask) is np.ndarray
    assert merged_cell_mask.dtype == np.uint16
    assert merged_cell_mask.shape == (tile_size_z, tile_size_y, tile_size_y + 2)
    assert (~np.isnan(merged_cell_mask)).all()
    assert (merged_cell_mask != 0).all()
    assert (merged_cell_mask == 1).any()
    assert (merged_cell_mask == 2).any()
    assert not (merged_cell_mask > 2).any()

    merged_cell_mask = cell_mask.merge_cell_masks(
        [tile_a_file_path, tile_b_file_path], np.array([[0, 0, 0], [0, 2, 0]], float), 0.1, merge_cells_method="merge 0"
    )
    assert type(merged_cell_mask) is np.ndarray
    assert merged_cell_mask.dtype == np.uint16
    assert merged_cell_mask.shape == (tile_size_z, tile_size_y, tile_size_y + 2)
    assert (~np.isnan(merged_cell_mask)).all()
    assert (merged_cell_mask != 0).all()
    assert (merged_cell_mask == 1).any()
    assert (merged_cell_mask == 2).any()
    assert not (merged_cell_mask > 2).any()

    # Place a single cell of size 6 pixels on both tile's.
    # Give them a 50% overlap.
    tile_a[:] = 0
    tile_a[1, 1:4, 2] = 1
    tile_a[2, 1:4, 2] = 1
    tile_b[:] = 0
    tile_b[1, 1:4, 1] = 3
    tile_b[2, 4:5, 1] = 3
    tile_b[2, 0, 2] = 3
    tile_b[2, 0, 1] = 3
    np.save(tile_a_file_path, tile_a)
    np.save(tile_b_file_path, tile_b)

    merged_cell_mask = cell_mask.merge_cell_masks(
        [tile_a_file_path, tile_b_file_path], [[0, 0, 0], [0, 1, 0]], 0.01, merge_cells_method="merge 0"
    )
    assert type(merged_cell_mask) is np.ndarray
    assert merged_cell_mask.dtype == np.uint16
    assert merged_cell_mask.shape == (tile_size_z, tile_size_y, tile_size_y + 1)
    assert (~np.isnan(merged_cell_mask)).all()
    assert (merged_cell_mask == 0).any()
    assert (merged_cell_mask == 1).any()
    # The cells should be merged together.
    tile_a = np.pad(tile_a, [[0, 0], [0, 0], [0, 1]])
    tile_b = np.pad(tile_b, [[0, 0], [0, 0], [1, 0]])
    assert (merged_cell_mask[np.logical_or(tile_a == 1, tile_b == 3)] == 1).all()
    assert not (merged_cell_mask > 2).any()

    merged_cell_mask = cell_mask.merge_cell_masks(
        [tile_a_file_path, tile_b_file_path], [[0, 0, 0], [0, 1, 0]], 0.01, merge_cells_method="merge 1"
    )
    assert type(merged_cell_mask) is np.ndarray
    assert merged_cell_mask.dtype == np.uint16
    assert merged_cell_mask.shape == (tile_size_z, tile_size_y, tile_size_y + 1)
    assert (~np.isnan(merged_cell_mask)).all()
    assert (merged_cell_mask == 0).any()
    assert (merged_cell_mask == 1).any()
    assert (merged_cell_mask == 2).any()
    # The cells should not be merged. Except one cell must take the overlapping region.
    assert (merged_cell_mask[np.logical_and(tile_a == 1, tile_b == 3)] > 0).all()
    assert (merged_cell_mask[np.logical_and(tile_a == 1, tile_b == 3)] <= 2).all()
    assert (merged_cell_mask[np.logical_and(tile_a == 1, tile_b != 3)] == 1).all()
    assert (merged_cell_mask[np.logical_and(tile_a != 1, tile_b == 3)] == 2).all()
    assert not (merged_cell_mask > 3).any()
