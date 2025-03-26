import numpy as np


def compose_affines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compose two 3D affines into one affine transform.

    The affine transforms are 3 by 4 arrays. The fourth column represents a translation.

    Args:
        a (`(3 x 4) ndarray`): the second applied affine transform.
        b (`(3 x 4) ndarray`): the first applied affine transform.

    Result:
        (`(3 x 4) ndarray`): c. The composed affine. c is equivalent to applying affine a then applying b to a 3D image.
    """
    assert type(a) is np.ndarray
    assert type(b) is np.ndarray
    assert a.shape == (3, 4)
    assert b.shape == (3, 4)

    a_linear = a[:, :3]
    b_linear = b[:, :3]
    a_translation = a[:, 3]
    b_translation = b[:, 3]

    c = np.zeros((3, 4), a.dtype)
    c[:, :3] = np.matmul(a_linear, b_linear)
    c[:, 3] = np.matmul(a_linear, b_translation) + a_translation

    return c
