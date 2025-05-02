import numpy as np


def dilate(
    vertices: np.ndarray[np.floating | np.integer], scale_factor: float, origin: np.ndarray[np.floating | np.integer]
) -> np.ndarray[np.float32]:
    """
    Dilate the given 2d polygon.

    Args:
        vertices (`(n_points x 2) ndarray[number]`): every vertex of the 2d polygon.
        scale_factor (float): the dilation scale factor.
        origin (`(2) ndarray`): the origin to expand relative to.

    Returns:
        (`(n_points x 2)`): dilated_vertices. The dilated vertices.
    """
    assert type(vertices) is np.ndarray
    assert vertices.ndim == 2
    assert vertices.shape[1] == 2, "Must be a two-dimension polygon"
    assert vertices.shape[0] > 2, "A polygon must have at least three vertices"
    assert type(scale_factor) is float
    assert scale_factor > 0
    assert type(origin) is np.ndarray
    assert origin.shape == (2,)

    OUTPUT_DTYPE = np.float32

    dilated_points = vertices.astype(OUTPUT_DTYPE, copy=True)
    dilated_points -= origin[np.newaxis]
    dilated_points *= scale_factor
    dilated_points += origin[np.newaxis]

    return dilated_points


def compute_centroid(vertices: np.ndarray[np.floating | np.integer]) -> np.ndarray[np.float32]:
    """
    Estimate the centre of mass (centroid) of a 2d polygon.

    The polygon must be non-intersecting to work correctly.

    Args:
        vertices (`(n_points x 2) ndarray[number]`): every vertex of the 2d polygon.

    Returns:
        (`(2) ndarray[float32]`): centroid_position. The position of the centroid.
    """
    assert type(vertices) is np.ndarray
    assert vertices.ndim == 2
    assert vertices.shape[1] == 2, "Must be a two-dimension polygon"
    assert vertices.shape[0] > 2, "A polygon must have at least three vertices"

    OUTPUT_DTYPE = np.float32

    points = vertices.astype(OUTPUT_DTYPE, copy=True)

    # Ensure the polygon is closed (first point == last point).
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    x = points[:, 0]
    y = points[:, 1]

    # Shifted (wrapped around) array of points (x_{i+1}, y_{i+1}).
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    # Signed area.
    A = 0.5 * np.sum(x * y_next - x_next * y)

    # Compute C_x and C_y.
    C_x = np.sum((x + x_next) * (x * y_next - x_next * y)) / (6 * A)
    C_y = np.sum((y + y_next) * (x * y_next - x_next * y)) / (6 * A)

    return np.array([C_x, C_y], OUTPUT_DTYPE)
