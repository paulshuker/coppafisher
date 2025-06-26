import numpy as np
import scipy


def validate_radius_normalisation(radius_normalisation: np.ndarray[np.floating], tile_size: int) -> None:
    """
    Raises an error if the radius normalisation parameter is not what is expected.
    """
    assert tile_size > 0

    if type(radius_normalisation) is not np.ndarray:
        raise TypeError(f"Expected radius_normalisation to be ndarray, got {type(radius_normalisation)}")

    if radius_normalisation.ndim != 1:
        raise ValueError(f"radius_normalisation must be one-dimensional, got {radius_normalisation.ndim}")

    max_image_radius = np.ceil(np.sqrt(2 * ((tile_size - 1) / 2) ** 2)).astype(int).item() + 1
    if radius_normalisation.size != max_image_radius:
        raise ValueError(
            f"Expected radius_normalisation to be size {max_image_radius}, got {radius_normalisation.size}"
        )


def radius_normalise_image(
    image: np.ndarray[np.floating], radius_normalisation: np.ndarray[np.floating]
) -> np.ndarray[np.floating]:
    """
    Radius normalise the image.

    Divide each pixel position by the value in radius_normalisation that corresponds to the pixel's radius from the tile
    centre in x/y. This is done for every z plane.

    Args:
        image (`(im_y x im_x x im_z) ndarray[float]`): the tile image. `im_y == im_x`.
        radius_normalisation (`(max_tile_radius) ndarray[float]`): radius_normalisation[r] is the value to divide by for a pixel at radius `r` from the
            centre of the tile image.

    Returns:
        (`(im_y x im_x x im_z) ndarray[float]`): normalised_image. The radius-normalised image.

    Notes:
        - If a pixel's radius is a non-integer, linearly interpolation is applied to find the value to divide by.
        - `max_tile_radius` must be `ceil(sqrt(2 * ((im_y - 1) / 2) ** 2)) + 1`.
    """
    assert type(image) is np.ndarray
    assert image.ndim == 3
    assert image.shape[0] == image.shape[1]
    assert type(radius_normalisation) is np.ndarray
    assert radius_normalisation.ndim == 1
    assert radius_normalisation.size == np.ceil(np.sqrt(2 * (0.5 * (image.shape[0] - 1)) ** 2)).astype(int) + 1

    radius_normalisation = radius_normalisation.astype(image.dtype)

    image_centre = np.array(image.shape, image.dtype) - 1
    image_centre /= 2

    image_yx_positions = np.meshgrid(
        np.linspace(0, image.shape[0] - 1, image.shape[0]),
        np.linspace(0, image.shape[1] - 1, image.shape[1]),
        indexing="ij",
    )
    # Has shape (2, im_y, im_x).
    image_yx_positions = np.array(image_yx_positions, image.dtype)
    image_yx_positions -= image_centre[:2, np.newaxis, np.newaxis]

    # The radius of each image position.
    image_radii = np.sqrt(image_yx_positions[0] * image_yx_positions[0] + image_yx_positions[1] * image_yx_positions[1])
    image_radii = image_radii.reshape(-1, order="F")
    del image_yx_positions

    linear_spline = scipy.interpolate.make_interp_spline(
        np.arange(radius_normalisation.size), radius_normalisation, k=1
    )
    image_normalisations = linear_spline(image_radii)
    image_normalisations = image_normalisations.reshape(image.shape[:2], order="F")
    del image_radii

    output = image.copy()
    output /= image_normalisations[:, :, np.newaxis]

    return output
