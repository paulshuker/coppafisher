import numpy as np
import scipy

from coppafisher.utils import affine


def test_compose_affines() -> None:
    a = np.zeros((4, 3), int)
    a[0] = [2, 0, 0]
    a[1] = [0, 1, 0]
    a[2] = [0, 0, 3]
    a[3] = [5, 6, 7]
    b = np.zeros((4, 3), int)
    b[0] = [0, 1, 0]
    b[1] = [-1, 0, 0]
    b[2] = [0, 0, 1]
    b[3] = [10, 20, 30]
    a = a.T
    b = b.T

    result = affine.compose_affines(a, b)

    assert type(result) is np.ndarray
    assert result.shape == (3, 4)

    rng = np.random.RandomState(0)
    image_base = rng.randint(50, size=(11, 12, 13), dtype=int)
    image_0 = scipy.ndimage.affine_transform(image_base, result, mode="grid-wrap")
    image_1 = scipy.ndimage.affine_transform(image_base, a, mode="grid-wrap")
    image_1 = scipy.ndimage.affine_transform(image_1, b, mode="grid-wrap")

    assert np.allclose(image_0, image_1)
