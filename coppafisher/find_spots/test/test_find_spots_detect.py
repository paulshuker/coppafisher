import timeit

import numpy as np
import pytest
import torch

from .. import detect


@pytest.mark.benchmark
def test_detect_spots_benchmark() -> None:
    def test() -> None:
        image_shape = 2304 // 4, 2304 // 4, 20
        rng = np.random.RandomState(0)
        image = rng.rand(*image_shape)
        intensity_thresh = 0.99

        detect.detect_spots(image, intensity_thresh, True, 4, 2)

    assert timeit.timeit(test, number=10, globals=locals()) <= 6.0


def test_detect_spots() -> None:
    image_shape = 11, 12, 5
    image = np.random.rand(*image_shape).astype(np.float32)
    intensity_thresh = 0.7
    radius_xy = 2
    radius_z = 1
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_xy=radius_xy, radius_z=radius_z
    )

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
    assert (maxima_yxz == 0).all(1).any(0)
    assert (maxima_yxz == [0, 3, 2]).all(1).any(0)
    assert (maxima_yxz == [0, 3, 4]).all(1).any(0)
    assert (maxima_intensity == 1).sum() == 1
    assert (maxima_intensity == 2).sum() == 2
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_z=2, radius_xy=1
    )
    assert maxima_yxz.shape == (2, 3)
    assert (maxima_yxz[1] == 0).all()
    # Only one of the two close maxima should be kept.
    assert (maxima_yxz[0] == [0, 3, 2]).all() or (maxima_yxz[0] == [0, 3, 4]).all()
    assert maxima_intensity[0] == 2
    assert maxima_intensity[1] == 1
    image[0, 3, 4] = 5
    maxima_yxz, maxima_intensity = detect.detect_spots(
        image, intensity_thresh, remove_duplicates=True, radius_z=2, radius_xy=1
    )
    assert maxima_yxz.shape == (2, 3)
    assert (maxima_yxz[1] == 0).all()
    assert (maxima_yxz[0] == [0, 3, 4]).all()
    assert maxima_intensity[1] == 1
    assert maxima_intensity[0] == 5

    image_shape = 11, 12, 13
    rng = np.random.RandomState(0)
    image = rng.rand(*image_shape).astype(np.float32)
    intensity_thresh = 0.6
    radius_xy = 1
    radius_z = 1
    maxima_yxz, maxima_intensity = detect.detect_spots(image, intensity_thresh, False, radius_xy, radius_z)

    n_spots_expected = (image > intensity_thresh).sum()
    assert maxima_yxz.shape == (n_spots_expected, 3)
    assert maxima_intensity.shape == (n_spots_expected,)
    for yxz in np.array(np.where(image > intensity_thresh)).T:
        assert (maxima_yxz == yxz).all(1).any(0)
        index = (maxima_yxz == yxz).all(1).nonzero()[0][0]
        assert np.isclose(maxima_intensity[index], image[yxz[0], yxz[1], yxz[2]])
