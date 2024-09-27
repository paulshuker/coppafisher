import numpy as np
import torch

from coppafish.find_spots import detect


def test_detect_spots() -> None:
    image_shape = 3, 4, 5
    image = np.zeros(image_shape, dtype=np.int16)
    image[0, 0, 0] = 1
    intensity_thresh = 0.0
    maxima_yxz, maxima_intensity = detect.detect_spots(image, intensity_thresh, remove_duplicates=False)
    assert type(maxima_yxz) is np.ndarray
    assert maxima_yxz.shape == (1, 3), f"Got shape {maxima_yxz.shape}"
    assert (maxima_yxz[0] == 0).all()
    assert type(maxima_intensity) is np.ndarray
    assert maxima_intensity.shape == (1,), f"Got shape {maxima_intensity.shape}"
    assert maxima_intensity[0] == 1
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_xy=1, radius_z=1
    )
    assert type(maxima_yxz) is np.ndarray
    assert maxima_yxz.shape == (1, 3)
    assert (maxima_yxz[0] == 0).all()
    assert type(maxima_intensity) is np.ndarray
    assert maxima_intensity.shape == (1,)
    assert maxima_intensity[0] == 1
    # Image with one isolated maxima and two nearby maxima.
    image[0, 3, 2] = 2
    image[0, 3, 4] = 2
    image = torch.from_numpy(image)
    maxima_yxz, maxima_intensity = detect.detect_spots(image, intensity_thresh, remove_duplicates=False)
    assert maxima_yxz.shape == (3, 3)
    assert (maxima_yxz[0] == 0).all()
    assert (maxima_yxz[1] == [0, 3, 2]).all()
    assert (maxima_yxz[2] == [0, 3, 4]).all()
    assert maxima_intensity[0] == 1
    assert maxima_intensity[1] == 2
    assert maxima_intensity[2] == 2
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_z=1, radius_xy=2
    )
    assert maxima_yxz.shape == (3, 3)
    assert (maxima_yxz[0] == 0).all()
    assert (maxima_yxz[1] == [0, 3, 2]).all()
    assert (maxima_yxz[2] == [0, 3, 4]).all()
    assert maxima_intensity[0] == 1
    assert maxima_intensity[1] == 2
    assert maxima_intensity[2] == 2
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_z=2, radius_xy=1
    )
    assert maxima_yxz.shape == (2, 3)
    assert (maxima_yxz[0] == 0).all()
    # Only one of the two close maxima should be kept.
    assert (maxima_yxz[1] == [0, 3, 2]).all() or (maxima_yxz[1] == [0, 3, 4]).all()
    assert maxima_intensity[0] == 1
    assert maxima_intensity[1] == 2
    image[0, 3, 4] = 5
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_z=2, radius_xy=1
    )
    assert maxima_yxz.shape == (2, 3)
    assert (maxima_yxz[0] == 0).all()
    assert (maxima_yxz[1] == [0, 3, 4]).all()
    assert maxima_intensity[0] == 1
    assert maxima_intensity[1] == 5
