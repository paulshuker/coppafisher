import math
from typing import List, Optional, Union

import numpy as np
import scipy
import torch

MAX_PARTITION: int = 2_000_000


def detect_spots(
    image: Union[np.ndarray, torch.Tensor] | List[Union[np.ndarray, torch.Tensor]],
    intensity_thresh: float,
    remove_duplicates: bool = False,
    radius_xy: Optional[int] = None,
    radius_z: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spots are detected as local maxima on the given image above intensity_thresh.

    Args:
        image (`(im_y x im_x x im_z) ndarray[int or float] or tensor[int or float]` or list of one image): image to
            detect the local maxima.
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

    Notes:
        - If the image is given as a mutable list, then the image will be popped and removed from the list once
            finished.
    """
    if type(image) is list:
        image = image.pop()
    assert type(image) is np.ndarray or type(image) is torch.Tensor
    assert type(intensity_thresh) is float
    assert type(remove_duplicates) is bool
    assert image.ndim == 3
    if remove_duplicates:
        assert radius_xy > 0
        assert radius_z > 0

    def find_locations_to_keep(locations: np.ndarray[np.int16], intensity: np.ndarray[np.float32]) -> np.ndarray[bool]:
        locations_norm = locations.copy().astype(np.float32)
        locations_norm[:, 2] *= radius_xy / radius_z
        kdtree = scipy.spatial.KDTree(locations_norm)
        # Gives a list for each maxima that contains a list of indices that are nearby neighbours, including itself.
        pairs = kdtree.query_ball_tree(kdtree, r=radius_xy)
        keep = np.array([len(pair) == 1 for pair in pairs], bool)
        for i, i_pairs in enumerate(pairs):
            if keep[i_pairs].any():
                # A near neighbour has already been kept.
                continue
            if (intensity[i] >= intensity[i_pairs]).all():
                keep[i_pairs] = False
                keep[i] = True
        return keep

    # (n_spots x 3) coordinate positions of the image local maxima.
    maxima_locations = np.array(np.array(image > intensity_thresh).nonzero()).T.astype(np.int16)
    maxima_intensities = np.array(image[tuple(maxima_locations.T)])
    del image

    if not remove_duplicates:
        return maxima_locations, maxima_intensities

    # Sometimes the KDTree query gets too large and causes a memory crash.
    # Therefore, n_spots is partitioned.
    n_partitions = math.ceil(maxima_intensities.size / MAX_PARTITION)
    maxima_location_partitions = np.array_split(maxima_locations, n_partitions)
    maxima_intensity_partitions = np.array_split(maxima_intensities, n_partitions)

    keep_maxima = np.zeros_like(maxima_intensities, bool)
    index_start = 0
    for locations_partition, maxima_intensity_partition in zip(
        maxima_location_partitions, maxima_intensity_partitions, strict=True
    ):
        keep_partition = find_locations_to_keep(locations_partition, maxima_intensity_partition)
        keep_maxima[index_start : index_start + keep_partition.size] = keep_partition
        index_start += keep_partition.size
    maxima_locations = maxima_locations[keep_maxima]
    maxima_intensities = maxima_intensities[keep_maxima]
    keep = find_locations_to_keep(maxima_locations, maxima_intensities)
    maxima_locations = maxima_locations[keep]
    maxima_intensities = maxima_intensities[keep]

    return maxima_locations, maxima_intensities
