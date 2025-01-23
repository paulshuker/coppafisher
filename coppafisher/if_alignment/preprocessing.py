import numpy as np
import skimage
from scipy import signal
from tqdm import tqdm


def slice1(o: int) -> slice:
    return slice(o, None) if o >= 0 else slice(0, o)


def custom_shift(array: np.ndarray, offset: np.ndarray, constant_values=0):
    """
    Compute array shifted by a certain offset.

    Args:
        array: array to be shifted.
        offset: shift value (must be int).
        constant_values: This is the value used for points outside the boundaries after shifting.

    Returns:
        new_array: array shifted by offset.
    """
    array = np.asarray(array)
    offset = np.atleast_1d(offset)
    assert len(offset) == array.ndim
    new_array = np.empty_like(array)

    new_array[tuple(slice1(o) for o in offset)] = array[tuple(slice1(-o) for o in offset)]

    for axis, o in enumerate(offset):
        new_array[(slice(None),) * axis + (slice(0, o) if o >= 0 else slice(o, None),)] = constant_values

    return new_array


def split_image(image: np.ndarray, subvolume_size: list, overlap: float = 0.1):
    """
    Split a 3D image into subvolumes of size given by subvolume_size
    Args:
        image: (np.ndarray) 3D image to split (nz, ny, nx)
        subvolume_size: Size of the subvolumes in each dimension (size_z, size_y, size_x)
        overlap: (float) Fraction of overlap between subvolumes: 0 <= overlap < 1 (default 0.1)

    Returns:
        subvolumes: (np.ndarray) List of subvolumes (n_subvolumes, size_z, size_y, size_x)
        position: (np.ndarray) List of positions of the subvolumes in the original image (n_subvolumes, 3)
    """
    assert 0 <= overlap < 1, "Overlap must be between 0 and 1"
    assert len(subvolume_size) == 3, "Subvolume size must be a list of 3 integers"
    assert all([type(i) == int for i in subvolume_size]), "Subvolume size must be a list of 3 integers"
    assert [subvolume_size[i] <= image.shape[i] for i in range(3)], "Subvolume size must be smaller than image size"

    # determine number of subvolumes in each dimension. This crops the image to the nearest multiple of subvolume_size,
    # after accounting for overlap
    im_size = np.array(image.shape)
    n_subvolumes = (im_size // ((1 - overlap) * np.array(subvolume_size))).astype(int)
    # regression will only work if there are more than 1 subvolume in each dimension
    assert all(n_subvolumes > 1), "Subvolume size too large for image size. Reduce subvolume size or increase overlap"

    # create an array of dimensions (n_subvolumes_z, n_subvolumes_y, n_subvolumes_x, size_z, size_y, size_x)
    subvol_dims = np.array([n_subvolumes, subvolume_size]).flatten()
    subvolumes = np.zeros(subvol_dims)
    positions = np.zeros(np.concatenate([n_subvolumes, [3]]))
    # populate the subvolumes array
    for z, y, x in tqdm(np.ndindex(tuple(n_subvolumes)), desc="Splitting image into subvolumes"):
        z_start, y_start, x_start = (np.array([z, y, x]) * (1 - overlap) * np.array(subvolume_size)).astype(int)
        z_end, y_end, x_end = z_start + subvolume_size[0], y_start + subvolume_size[1], x_start + subvolume_size[2]
        z_centre, y_centre, x_centre = (z_start + z_end) // 2, (y_start + y_end) // 2, (x_start + x_end) // 2
        subvolumes[z, y, x] = image[z_start:z_end, y_start:y_end, x_start:x_end]
        positions[z, y, x] = np.array([z_centre, y_centre, x_centre])

    return subvolumes, positions


def merge_subvols(position, subvol):
    """
    Suppose we have a known volume V split into subvolumes. The position of the subvolume corner
    in the coords of the initial volume is given by position. However, these subvolumes may have been shifted
    so position may be slightly different. This function finds the minimal volume containing all these
    shifted subvolumes.

    If regions overlap, we take the values from the later subvolume.
    Args:
        position: n_subvols x 3 array of positions of bottom left of subvols (zyx)
        subvol: n_subvols x z_box x y_box x x_box array of subvols

    Returns:
        merged: merged image (size will depend on amount of overlap)
    """
    position = position.astype(int)
    # set min values to 0
    position -= np.min(position, axis=0)
    z_box, y_box, x_box = subvol.shape[1:]
    centre = position + np.array([z_box // 2, y_box // 2, x_box // 2])
    # Get the min and max values of the position, use this to get the size of the merged image and initialise it
    max_pos = np.max(position, axis=0)
    merged = np.zeros((max_pos + subvol.shape[1:]).astype(int))
    neighbour_im = np.zeros_like(merged)
    # Loop through the subvols and add them to the merged image at the correct position.
    for i in range(position.shape[0]):
        subvol_i_mask = np.ix_(
            range(position[i, 0], position[i, 0] + z_box),
            range(position[i, 1], position[i, 1] + y_box),
            range(position[i, 2], position[i, 2] + x_box),
        )
        neighbour_im[subvol_i_mask] += 1
        merged[subvol_i_mask] = subvol[i]

    # identify overlapping regions
    overlapping_pixels = np.argwhere(neighbour_im > 1)
    if len(overlapping_pixels) == 0:
        return merged
    centre_dist = np.linalg.norm(overlapping_pixels[:, None, :] - centre[None, :, :], axis=2)
    # get the index of the closest centre
    closest_centre = np.argmin(centre_dist, axis=1)
    # now loop through subvols and assign overlapping pixels to the closest centre
    for i in range(position.shape[0]):
        subvol_i_pixel_ind = np.where(closest_centre == i)[0]
        subvol_i_pixel_coords_global = np.array([overlapping_pixels[j] for j in subvol_i_pixel_ind])
        # if there are no overlapping pixels, skip
        if len(subvol_i_pixel_coords_global) == 0:
            continue
        subvol_i_pixel_coords_local = subvol_i_pixel_coords_global - position[i]
        z_global, y_global, x_global = subvol_i_pixel_coords_global.T
        z_local, y_local, x_local = subvol_i_pixel_coords_local.T
        merged[z_global, y_global, x_global] = subvol[i, z_local, y_local, x_local]

    return merged


def window_image(image: np.ndarray) -> np.ndarray:
    """
    Window the image by a hann window in y and x and a Tukey window in z.

    Args:
        image: image to be windowed. (z, y, x)

    Returns:
        image: windowed image.
    """
    window_yx = skimage.filters.window("hann", image.shape[1:])
    window_z = signal.windows.tukey(image.shape[0], alpha=0.33)
    if (window_z == 0).all():
        window_z[...] = 1
    window = window_z[:, None, None] * window_yx[None, :, :]
    image = image * window

    return image
