import numpy as np


def pad_and_crop_image_to_origin(
    image: np.ndarray[np.floating],
    current_origin: np.ndarray[np.floating],
    global_origin: np.ndarray[np.floating],
    shape: tuple[int, int, int],
    pad_value: float = 0,
) -> np.ndarray[np.floating]:
    """
    The image is cropped to be the given shape and start at the given global origin at the first value.

    Args:
        image (`(im_z x im_y x im_x) ndarray`): the image to crop and pad.
        current_origin (`(3) ndarray`): the current origin position for the first value of the image.
        global_origin (`(3) ndarray`): the wanted global origin position for the first value of the image.
        shape (tuple of three ints): the required shape to put the image into.
        pad_value (float, optional): the value to pad the image with. Default: 0.

    Returns:
        (`shape ndarray`): padded_cropped_image. The final padded and/or cropped image.
    """
    assert image.ndim == 3
    assert current_origin.shape == (3,)
    assert global_origin.shape == (3,)
    assert len(shape) == 3

    new_image = image.copy()
    origin_difference: list[int] = np.rint(current_origin - global_origin).astype(int).tolist()
    # Ensure the image has the right shape on the starting edges.
    for dim in range(3):
        if origin_difference[dim] == 0:
            continue

        if origin_difference[dim] > 0:
            # Pad the image.
            pad_widths = []
            for _ in range(dim):
                pad_widths.append([0, 0])
            pad_widths.append([origin_difference[dim], 0])
            for _ in range(3 - dim - 1):
                pad_widths.append([0, 0])
            new_image = np.pad(new_image, pad_widths, constant_values=pad_value)
        else:
            # Crop the image.
            crop_amount = origin_difference[dim]
            if dim == 0:
                new_image = new_image[crop_amount:]
            elif dim == 1:
                new_image = new_image[:, crop_amount:]
            elif dim == 2:
                new_image = new_image[:, :, crop_amount:]

    # Now ensure the image has the right shape at the end edges.
    for dim in range(3):
        if new_image.shape[dim] == shape[dim]:
            continue

        if new_image.shape[dim] < shape[dim]:
            # Pad the image.
            pad_amount = shape[dim] - new_image.shape[dim]
            pad_widths = []
            for _ in range(dim):
                pad_widths.append([0, 0])
            pad_widths.append([0, pad_amount])
            for _ in range(3 - dim - 1):
                pad_widths.append([0, 0])
            new_image = np.pad(new_image, pad_widths, constant_values=pad_value)
        else:
            # Crop the image.
            crop_amount = new_image.shape[dim] - shape[dim]
            if dim == 0:
                new_image = new_image[:-crop_amount]
            elif dim == 1:
                new_image = new_image[:, :-crop_amount]
            elif dim == 2:
                new_image = new_image[:, :, :-crop_amount]

        assert new_image.shape[dim] == shape[dim]

    assert new_image.shape == shape
    return new_image
