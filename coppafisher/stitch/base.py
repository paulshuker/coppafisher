from typing import Tuple

import numpy as np
import skimage
from tqdm import tqdm

from ..register import preprocessing


def stitch(
    tile_images: np.ndarray,
    tilepos_yx: np.ndarray[int],
    use_tiles: list[int],
    n_all_tiles: int,
    expected_overlap: float,
) -> Tuple[np.ndarray[np.floating], np.ndarray[np.floating], np.ndarray[np.floating]]:
    """
    Stitch given tile images together. Tiles are stitched together by translating each image into a global coordinate
    system. Phase Cross Correlation is used to find the best translations.

    Args:
        tile_images (`(len(use_tiles) x im_y x im_x x im_z) ndarray[float32]`): tile_images[i] is tile use_tiles[i]'s
            image.
        tilepos_yx (`(len(use_tiles) x 2) ndarray[int]`): tilepos_yx[i] is tile use_tiles[i]'s position relative to the
            other tiles along and y and x axis.
        use_tiles (list of int): the indices of each tile.
        n_all_tiles (int): the total number of tiles in the experiment. This is `>= len(use_tiles)`.
        expected_overlap (float): the approximate overlap between adjacent tiles as a fraction of their pixel count in
            x/y.

    Returns:
        Tuple containing:
            - (`(n_all_tiles x 3) ndarray[float]`): tile_origins_full.
            - (`(n_all_tiles x n_all_tiles x 3) ndarray[float]`): pairwise_shifts_full.
            - (`(n_all_tiles x n_all_tiles) ndarray[float]`): pairwise_shift_scores_full.
    """
    n_tiles_use = len(use_tiles)

    # Build the arrays that we will use to compute the pairwise shift.
    pairwise_shifts = np.zeros((n_tiles_use, n_tiles_use, 3))
    pairwise_shift_scores = np.zeros((n_tiles_use, n_tiles_use))

    # Fill the pairwise shift and pairwise shift score matrices.
    for i, j in tqdm(np.ndindex(n_tiles_use, n_tiles_use), total=n_tiles_use**2, desc="Computing shifts between tiles"):
        # If the tiles are not adjacent, skip.
        if abs(tilepos_yx[i] - tilepos_yx[j]).sum() != 1:
            continue
        pairwise_shifts[i, j], pairwise_shift_scores[i, j] = compute_shift(
            t1=tile_images[i], t2=tile_images[j], t1_pos=tilepos_yx[i], t2_pos=tilepos_yx[j], overlap=expected_overlap
        )

    # Compute the nominal_origin_deviations using a minimisation of a quadratic loss function.
    # Instead of recording the shift between adjacent tiles to yield an n_tiles_use x n_tiles_use x 3 array as in
    # pairwise_shiftss, this is an n_all_tiles x 3 array of every tile's shift from its nominal origin.
    nominal_origin_deviations = minimise_shift_loss(shift=pairwise_shifts, score=pairwise_shift_scores)

    # Expand the pairwise shifts and pairwise shift scores from n_tiles_use x n_tiles_use x 3 to n_all_tiles x n_all_tiles x 3.
    pairwise_shifts_full, pairwise_shift_scores_full, tile_origins_full = (
        np.zeros((n_all_tiles, n_all_tiles, 3)) * np.nan,
        np.zeros((n_all_tiles, n_all_tiles)) * np.nan,
        np.zeros((n_all_tiles, 3)) * np.nan,
    )
    im_size_y, im_size_x = tile_images[0].shape[:-1]
    for i, t in enumerate(use_tiles):
        # Fill the full shift and score matrices.
        pairwise_shifts_full[t, use_tiles] = pairwise_shifts[i]
        pairwise_shift_scores_full[t, use_tiles] = pairwise_shift_scores[i]
        # Fill the tile origins.
        nominal_origin = np.array(
            [
                tilepos_yx[i][0] * im_size_y * (1 - expected_overlap),
                tilepos_yx[i][1] * im_size_x * (1 - expected_overlap),
                0,
            ]
        )
        tile_origins_full[t] = nominal_origin + nominal_origin_deviations[i]

    return tile_origins_full, pairwise_shift_scores, pairwise_shift_scores_full


def compute_shift(
    t1: np.ndarray, t2: np.ndarray, t1_pos: np.ndarray, t2_pos: np.ndarray, overlap: float
) -> Tuple[np.ndarray, float]:
    """
    Compute the boundary shift between two tiles t1 and t2. The shift is computed by comparing the overlapping regions
    of the two tiles, and using a phase cross correlation algorithm to find any deviation from the expected overlap.
    Args:
        t1: (this is the tile that will be shifted) np.ndarray, [y_size, x_size, z_size] array of the first tile
        t2: (this is the reference tile) np.ndarray, [y_size, x_size, z_size] array of the second tile
        t1_pos: [y, x] position of tile 1 (integer indices)
        t2_pos: [y, x] position of tile 2 (integer indices)
        overlap: float, expected overlap between the two tiles

    Returns:
        shift: np.ndarray, shift in pixels between the two tiles
        score: float, square of the correlation coefficient between the reference tile and the shifted tile

    """

    # crop the tiles to the overlapping regions of the two tiles
    if (t2_pos[1] - t1_pos[1] == 1) and (t2_pos[0] - t1_pos[0] == 0):  # t2 is to the right of t1
        t1 = t1[:, -int(overlap * t1.shape[1]) :]
        t2 = t2[:, : int(overlap * t2.shape[1])]
    elif (t2_pos[1] - t1_pos[1] == -1) and (t2_pos[0] - t1_pos[0] == 0):  # t2 is to the left of t1
        t1 = t1[:, : int(overlap * t1.shape[1])]
        t2 = t2[:, -int(overlap * t2.shape[1]) :]
    elif (t2_pos[1] - t1_pos[1] == 0) and (t2_pos[0] - t1_pos[0] == 1):  # t2 is below t1
        t1 = t1[-int(overlap * t1.shape[0]) :, :]
        t2 = t2[: int(overlap * t2.shape[0]), :]
    elif (t2_pos[1] - t1_pos[1] == 0) and (t2_pos[0] - t1_pos[0] == -1):  # t2 is above t1
        t1 = t1[: int(overlap * t1.shape[0]), :]
        t2 = t2[-int(overlap * t2.shape[0]) :, :]
    else:
        raise ValueError("Tiles are not adjacent")  # this should never happen
    window = skimage.filters.window("hann", shape=(t1.shape[0], t1.shape[1]))
    # extend the window in z, but as we don't have many z-planes, fade first and last quarter of the planes
    z_planes = t1.shape[-1]
    n_z_fade = z_planes // 4
    window_z = np.concatenate(
        (np.linspace(0, 1, n_z_fade), np.ones(z_planes - 2 * n_z_fade), np.linspace(1, 0, n_z_fade))
    )
    window = window[:, :, np.newaxis] * window_z[np.newaxis, np.newaxis, :]

    # compute the shift
    shift = skimage.registration.phase_cross_correlation(
        reference_image=t2 * window, moving_image=t1 * window, overlap_ratio=0.5, disambiguate=True
    )[0]
    # compute the score
    t1_shifted = preprocessing.custom_shift(t1, shift.astype(int))
    mask = t1_shifted > 0
    score = np.corrcoef(t1_shifted[mask].flatten(), t2[mask].flatten())[0, 1]

    return shift, score**2


def minimise_shift_loss(shift: np.ndarray, score: np.ndarray) -> np.ndarray:
    """
    We have ~ 2 * n_tiles shifts that have been computed between tiles and only n_tiles shifts that we can apply to the
    tiles. We need to find the n_tiles shifts that minimise the quadratic loss function between the computed shifts
    and the shifts that we can apply to the tiles. Taking the derivative of the loss function and setting it to zero
    gives us a linear system of equations that we can solve to find the optimal shifts.

    Note: The loss function is defined as:

    L(w) = sum_i sum_j score[i, j] * (w[i] - w[j] - shift[i, j])^2,
    where w is the vector of shifts that we want to find, and shift is the matrix of computed shifts. This sum is over
    all neighbouring tiles (which is achieved by setting score = 0 for non-neighbouring tiles).

    Args:
        shift: np.ndarray, [n_tiles, n_tiles, 3] array of the shifts between the tiles
        score: np.ndarray, [n_tiles, n_tiles] array of the correlation scores between the tiles

    Returns:
        shifts_final: np.ndarray, [n_tiles, 3] array of the final shifts that will be applied to the tiles

    """
    n_tiles = shift.shape[0]
    # we need to build the n_tiles x n_tiles matrix A and the n_tiles x 3 matrix b that will be used to solve the linear
    # system of equations: Ax = b, where x is our final shift matrix (n_tiles x 3)
    A = np.zeros((n_tiles, n_tiles))
    b = np.zeros((n_tiles, 3))
    # fill the A matrix (do the maths on paper to understand this)
    for i, j in np.ndindex(n_tiles, n_tiles):
        if i == j:
            A[i, j] = np.sum(score[i, :])
        else:
            A[i, j] = -score[i, j]
    # fill the b matrix
    for i in range(n_tiles):
        b[i] = np.sum(score[i, :, np.newaxis] * shift[i], axis=0)

    # solve the linear system of equations
    shifts_final = np.linalg.lstsq(A, b, rcond=None)[0]

    return shifts_final
