import numpy as np

from coppafisher.filter import radius_normalisation


def test_radius_normalise_image() -> None:
    rng = np.random.RandomState(0)
    image = rng.rand(3, 3, 10).astype(np.float32)
    radius_norm = np.ones(3, np.float32)

    output = radius_normalisation.radius_normalise_image(image, radius_norm)

    assert type(output) is np.ndarray
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert np.allclose(output, image)

    radius_norm[-1] = 0.5

    # Linear interpoaltion check.
    expected_edge_boost = np.sqrt(2) - 1
    expected_edge_boost = 0.5 * expected_edge_boost + 1 * (1 - expected_edge_boost)
    expected_edge_boost = 1 / expected_edge_boost

    output = radius_normalisation.radius_normalise_image(image, radius_norm)

    assert type(output) is np.ndarray
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert not np.allclose(image, output)
    assert np.allclose(expected_edge_boost * image[-1, -1, :], output[-1, -1, :])

    # Try with an even number of image pixels in x/y.
    image = rng.rand(4, 4, 7)
    radius_norm = np.ones(4, np.float64)

    output = radius_normalisation.radius_normalise_image(image, radius_norm)

    assert type(output) is np.ndarray
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert np.allclose(image, output)

    radius_norm[0] = 0

    output = radius_normalisation.radius_normalise_image(image, radius_norm)

    assert type(output) is np.ndarray
    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert not np.isclose(image[1:3, 1:3], output[1:3, 1:3]).any()
    assert np.allclose(image[1:3, 1:3], output[1:3, 1:3] / np.sqrt(2))
    assert np.allclose(image[:, 0], output[:, 0])
    assert np.allclose(image[0, :], output[0, :])
    assert np.allclose(image[-1, :], output[-1, :])
    assert np.allclose(image[:, -1], output[:, -1])
