import numpy as np

from coppafisher.utils import polygon2d


def test_dilate() -> None:
    square_vertices = np.zeros((4, 2), np.float64)
    square_vertices[1] = [0, 1]
    square_vertices[2] = [1, 1]
    square_vertices[3] = [1, 0]

    dilated_vertices = polygon2d.dilate(square_vertices, 2.0, np.array([0, 0]))
    expected_vertices = square_vertices.copy()
    expected_vertices[1] = [0, 2]
    expected_vertices[2] = [2, 2]
    expected_vertices[3] = [2, 0]

    assert type(dilated_vertices) is np.ndarray
    assert dilated_vertices.shape == square_vertices.shape
    assert np.allclose(dilated_vertices, expected_vertices)

    dilated_vertices = polygon2d.dilate(square_vertices, 2.0, np.array([0, 1]))
    expected_vertices = square_vertices.copy()
    expected_vertices[0] = [0, -1]
    expected_vertices[1] = [0, 1]
    expected_vertices[2] = [2, 1]
    expected_vertices[3] = [2, -1]

    assert type(dilated_vertices) is np.ndarray
    assert dilated_vertices.shape == square_vertices.shape
    assert np.allclose(dilated_vertices, expected_vertices)


def test_compute_centroid() -> None:
    square_vertices = np.zeros((4, 2), np.float64)
    square_vertices[1] = [0, 1]
    square_vertices[2] = [1, 1]
    square_vertices[3] = [1, 0]

    centroid_position = polygon2d.compute_centroid(square_vertices)

    assert type(centroid_position) is np.ndarray
    assert centroid_position.dtype.type == np.float32
    assert centroid_position.shape == (2,)
    assert np.allclose(centroid_position, [0.5, 0.5])

    square_vertices[1] = [0, 1]
    square_vertices[2] = [2, 1]
    square_vertices[3] = [2, 0]

    centroid_position = polygon2d.compute_centroid(square_vertices)

    assert type(centroid_position) is np.ndarray
    assert centroid_position.dtype.type == np.float32
    assert centroid_position.shape == (2,)
    assert np.allclose(centroid_position, [1, 0.5])
