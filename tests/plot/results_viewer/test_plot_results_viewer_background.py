import numpy as np

from coppafisher.plot.results_viewer import background
from coppafisher.setup.notebook_page import NotebookPage


def test_Region() -> None:
    r1 = background._Region()
    r1.min_yxz = np.array([0, 1, 2], int)
    r1.max_yxz = r1.min_yxz + 1

    min_yxz = np.array([0, 1, 2], int)
    max_yxz = min_yxz.copy() + 1
    assert r1.overlaps_with(min_yxz, max_yxz)
    min_yxz -= 10
    assert r1.overlaps_with(min_yxz, max_yxz)
    max_yxz += 11
    assert r1.overlaps_with(min_yxz, max_yxz)
    min_yxz += 9
    max_yxz -= 12
    assert not r1.overlaps_with(min_yxz, max_yxz)
    min_yxz = r1.min_yxz.copy() + 1
    max_yxz = r1.max_yxz.copy() + 1
    assert not r1.overlaps_with(min_yxz, max_yxz)
    min_yxz = r1.min_yxz.copy()
    max_yxz = r1.max_yxz.copy() + 1
    min_yxz[2] += 4
    max_yxz[2] += 4
    assert not r1.overlaps_with(min_yxz, max_yxz)


def test_generate_global_image() -> None:
    # Tiles are shape 10x10x5 along y/x/z.
    # 3x2 tiles along y/x.
    # Each tile has a 30% (3 pixels) stitched overlap with neighbours along y and x.
    n_tiles = 6
    n_tiles_y = 3
    n_tiles_x = 2
    tile_shape_yxz = (10, 10, 4)
    # In pixels.
    actual_overlap = 3

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.tile_sz = tile_shape_yxz[0]
    nbp_basic.use_tiles = tuple(range(n_tiles))
    nbp_basic.use_z = tuple(range(tile_shape_yxz[2]))

    nbp_stitch = NotebookPage("stitch", {"stitch": {"expected_overlap": actual_overlap / tile_shape_yxz[0]}})
    tile_origins = np.zeros((n_tiles, 3), float)
    t = 0
    for x in range(n_tiles_x):
        for y in range(n_tiles_y):
            tile_origins[t] = np.array([y * tile_shape_yxz[0], x * tile_shape_yxz[1], 0], float)
            overlap_offset = np.array((y * actual_overlap, x * actual_overlap, 0), float)
            tile_origins[t] -= overlap_offset
            t += 1
    nbp_stitch.tile_origin = tile_origins

    # Every tile is filled with a unique, constant value for simplicity.
    images = [np.ones(tile_shape_yxz, np.float16) * t for t in range(n_tiles)]

    result = background.generate_global_image(images, nbp_basic.use_tiles, nbp_basic, nbp_stitch, silent=False)

    # import napari
    # v = napari.Viewer()
    # v.add_image(result, contrast_limits=(0, 10))
    # v.add_image(images[0].transpose((2, 0, 1)), translate=nbp_stitch.tile_origin[0][[2, 0, 1]], contrast_limits=(0, 10))
    # v.add_image(images[1].transpose((2, 0, 1)), translate=nbp_stitch.tile_origin[1][[2, 0, 1]], contrast_limits=(0, 10))
    # v.add_image(images[2].transpose((2, 0, 1)), translate=nbp_stitch.tile_origin[2][[2, 0, 1]], contrast_limits=(0, 10))
    # v.add_image(images[3].transpose((2, 0, 1)), translate=nbp_stitch.tile_origin[3][[2, 0, 1]], contrast_limits=(0, 10))
    # v.add_image(images[4].transpose((2, 0, 1)), translate=nbp_stitch.tile_origin[4][[2, 0, 1]], contrast_limits=(0, 10))
    # v.add_image(images[5].transpose((2, 0, 1)), translate=nbp_stitch.tile_origin[5][[2, 0, 1]], contrast_limits=(0, 10))
    # napari.run()

    assert len(images) == n_tiles
    assert type(result) is np.ndarray
    assert result.dtype == np.float16
    assert result.shape == (
        tile_shape_yxz[2],
        n_tiles_y * tile_shape_yxz[0] - (n_tiles_y - 1) * actual_overlap,
        n_tiles_x * tile_shape_yxz[1] - (n_tiles_x - 1) * actual_overlap,
    )
    assert ((result >= 0) & (result <= (n_tiles - 1))).all()
    assert np.allclose(result[:, : tile_shape_yxz[0] - actual_overlap, : tile_shape_yxz[1] - actual_overlap], 0)
    overlap_results = result[
        :, : tile_shape_yxz[0] - actual_overlap, tile_shape_yxz[1] - actual_overlap : tile_shape_yxz[1]
    ]
    assert (0 <= overlap_results).all()
    assert (overlap_results <= 3).all()

    # assert np.allclose(overlap_results[:, 3:, 0], 0)
    # assert np.allclose(overlap_results[:, 3:, 1], 0 * 0.5 + 3 * 0.5)
    # assert np.allclose(overlap_results[:, 3:, 2], 3)
    # assert np.allclose(result[:, 16, 10:17], 5)
    # assert np.allclose(result[:, 15, 10:17], 4 * 0.5 + 5 * 0.5)
    # assert np.allclose(result[:, 14, 10:17], 4)

    # Case where all tiles have no overlap.
    tile_origins = np.zeros((n_tiles, 3), float)
    t = 0
    for x in range(n_tiles_x):
        for y in range(n_tiles_y):
            tile_origins[t] = np.array([y * tile_shape_yxz[0], x * tile_shape_yxz[1], 0], float)
            t += 1
    nbp_stitch = NotebookPage("stitch", {"stitch": {"expected_overlap": 0}})
    nbp_stitch.tile_origin = tile_origins

    # Every tile is filled with a unique, constant value for simplicity.
    images = [np.ones(tile_shape_yxz, np.float16) * t for t in range(n_tiles)]

    result = background.generate_global_image(images, nbp_basic.use_tiles, nbp_basic, nbp_stitch)

    assert type(result) is np.ndarray
    assert result.shape == (tile_shape_yxz[2], tile_shape_yxz[0] * n_tiles_y, tile_shape_yxz[1] * n_tiles_x)
    t = 0
    for x in range(n_tiles_x):
        for y in range(n_tiles_y):
            x_min, x_max = x * tile_shape_yxz[1], (x + 1) * tile_shape_yxz[1]
            y_min, y_max = y * tile_shape_yxz[0], (y + 1) * tile_shape_yxz[0]
            assert np.allclose(result[:, y_min:y_max, x_min:x_max], t)
            t += 1
