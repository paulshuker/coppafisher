from typing import Union

import numpy as np
import scipy
import torch


def get_isolated_spots(
    yxz_positions: Union[torch.Tensor, np.ndarray],
    distance_threshold_yx: Union[float, int],
    distance_threshold_z: Union[float, int],
) -> torch.Tensor:
    """
    Checks what given point positions are truly isolated. A point is truly isolated if the closest other point
    position is further than the given distance thresholds.

    Args:
        - yxz_positions (`(n_points x 3) ndarray[int] or tensor[int]`): y, x, and z positions for each point.
        - distance_threshold_yx (float): any positions within this distance threshold along x or y are not truly
            isolated.
        - distance_threshold_z (float): any positions within this distance threshold along z are not truly isolated.

    Returns:
        `(n_points) tensor[bool]`: true for each point considered truly isolated.
    """
    assert type(yxz_positions) is torch.Tensor or type(yxz_positions) is np.ndarray
    assert yxz_positions.ndim == 2
    assert yxz_positions.shape[0] > 0
    assert yxz_positions.shape[1] == 3
    assert type(distance_threshold_yx) is float or type(distance_threshold_yx) is int
    assert type(distance_threshold_z) is float or type(distance_threshold_z) is int

    if type(yxz_positions) is torch.Tensor:
        yxz_norm = yxz_positions.numpy()
    else:
        yxz_norm = yxz_positions.copy()
    yxz_norm = yxz_norm.astype(np.float32)
    yxz_norm[:, 2] *= distance_threshold_yx / distance_threshold_z
    kdtree = scipy.spatial.KDTree(yxz_norm)
    close_pairs = kdtree.query_pairs(r=distance_threshold_yx, output_type="ndarray")
    assert close_pairs.shape[1] == 2
    close_pairs = close_pairs.ravel()
    close_pairs = np.unique(close_pairs)
    true_isolate = np.ones(yxz_norm.shape[0], dtype=bool)
    true_isolate[close_pairs] = False
    true_isolate = torch.tensor(true_isolate)

    return true_isolate


def check_neighbour_intensity(image: np.ndarray, spot_yxz: np.ndarray, thresh: float = 0) -> np.ndarray:
    """
    Checks whether a neighbouring pixel to those indicated in ```spot_yxz``` has intensity less than ```thresh```.
    The idea is that if pixel has very low intensity right next to it, it is probably a spurious spot.

    Args:
        image: ```float [n_y x n_x x n_z]```.
            image spots were found on.
        spot_yxz: ```int [n_peaks x image.ndim]```.
            yx or yxz location of spots found.
            If axis 1 dimension is more than ```image.ndim```, only first ```image.ndim``` dimensions used
            i.e. if supply yxz, with 2d image, only yx position used.
        thresh: Spots are indicated as ```False``` if intensity at neighbour to spot location is less than this.

    Returns:
        ```float [n_peaks]```.
            ```True``` if no neighbours below thresh.
    """
    if image.ndim == 3:
        transforms = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    elif image.ndim == 2:
        transforms = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    else:
        raise ValueError(f"image has to have two or three dimensions but given image has {image.ndim} dimensions.")
    keep = np.zeros((spot_yxz.shape[0], len(transforms)), dtype=bool)
    for i, t in enumerate(transforms):
        mod_spot_yx = spot_yxz + t
        for j in range(image.ndim):
            mod_spot_yx[:, j] = np.clip(mod_spot_yx[:, j], 0, image.shape[j] - 1)
        keep[:, i] = image[tuple([mod_spot_yx[:, j] for j in range(image.ndim)])] > thresh
    return keep.min(axis=1)


def filter_intense_spots(
    local_yxz: np.ndarray, spot_intensity: np.ndarray, n_z: int, max_spots: int = 500
) -> np.ndarray:
    """
    Filters spots by intensity. For each z plane, keeps only the top max_spots spots.
    Args:
        local_yxz: [n_spots x 3] int array of yxz positions of spots
        spot_intensity: [n_spots] float array of spot intensities
        max_spots: int indicating maximum number of spots to keep per z plane
        n_z: int indicating number of z planes

    Returns:
        local_yxz: [n_spots_keep x 3] int array of yxz positions of spots
    """
    assert spot_intensity.ndim == 1
    keep = np.ones(local_yxz.shape[0], dtype=bool)
    # Loop over each z plane and keep only the top max_spots spots
    for z in range(n_z):
        # If the number of spots on this z-plane is > max_spots (500 by default for 3D) then we
        # set the intensity threshold to the 500th most intense spot and take the top 500 values
        z_spot_count = np.sum(local_yxz[:, 2] == z)
        if z_spot_count > max_spots:
            intensity_thresh = np.sort(spot_intensity[local_yxz[:, 2] == z])[-max_spots]
            keep[np.logical_and(local_yxz[:, 2] == z, spot_intensity < intensity_thresh)] = False

    return local_yxz[keep]
