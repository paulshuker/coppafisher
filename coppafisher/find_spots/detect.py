from typing import Optional, Union

import numpy as np
import scipy
import torch


def detect_spots(
    image: Union[np.ndarray, torch.Tensor],
    intensity_thresh: float,
    remove_duplicates: bool = False,
    radius_xy: Optional[int] = None,
    radius_z: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spots are detected as local maxima on the given image above intensity_thresh.

    Args:
        image (`(im_y x im_x x im_z) ndarray[int or float] or tensor[int or float]`): image to detect the local maxima.
        intensity_thresh (float): local maxima are greater than intensity_thresh.
        remove_duplicates (bool, optional): if two or more local maxima are close together, then only the greatest
            maxima value is detected. If they have identical intensities, one is chosen over the other. Default: false.
        radius_xy (int, optional): two local maxima are considered close together if their distance along x and/or y is
            less than radius_xy. Default: not given.
        radius_z (int, optional): two local maxima are considered close together if their distance along z is less than
            radius_xy. Default: not given.

    Returns:
        Tuple containing:
            - (`(n_spots x 3) ndarray[int16]`): maxima_yxz. The y, x, and z coordinate positions of local maxima.
            - (`(n_spots) ndarray[image.dtype]`): maxima_intensity. maxima_intensity[i] is the image intensity at maxima_yxz[i].
    """
    assert type(image) is np.ndarray or type(image) is torch.Tensor
    assert type(intensity_thresh) is float
    assert type(remove_duplicates) is bool
    assert image.ndim == 3
    if remove_duplicates:
        assert radius_xy > 0
        assert radius_z > 0

    # (n_spots x 3) coordinate positions of the image local maxima.
    maxima_locations = np.array(np.array(image > intensity_thresh).nonzero()).T.astype(np.int16)
    maxima_intensities = np.array(image[tuple(maxima_locations.T)])
    if remove_duplicates:
        maxima_locations_norm = maxima_locations.astype(np.float32)
        maxima_locations_norm[:, 2] *= radius_xy / radius_z
        kdtree = scipy.spatial.KDTree(maxima_locations_norm)
        # Gives a list for each maxima that contains a list of indices that are nearby neighbours, including itself.
        pairs = kdtree.query_ball_tree(kdtree, r=radius_xy)
        keep_maxima = np.array([len(pair) == 1 for pair in pairs], bool)
        completed_indices = keep_maxima.copy()
        for i, i_pairs in enumerate(pairs):
            if completed_indices[i]:
                continue
            intensity_argsorted = [i_pairs[i] for i in np.argsort(maxima_intensities[i_pairs])]
            keep_maxima[intensity_argsorted[-1]] = True
            keep_maxima[intensity_argsorted[:-1]] = False
            completed_indices[intensity_argsorted[:-1]] = True
        del completed_indices

        maxima_locations = maxima_locations[keep_maxima]
        maxima_intensities = maxima_intensities[keep_maxima]

    return maxima_locations, maxima_intensities
