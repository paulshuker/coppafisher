import numpy as np
import skimage
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm

from . import preprocessing


def find_shift_array(subvol_base, subvol_target, position, r_threshold):
    """
    This function takes in 2 3d images which have already been split up into 3d subvolumes. We then find the shift from
    each base subvolume to its corresponding target subvolume.
    NOTE: This function does allow matching to non-corresponding subvolumes in z.
    NOTE: This function performs flattening of the output arrays, and this is done according to np.reshape.
    Args:
        subvol_base: Base subvolume array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, z_box, y_box, x_box)
        subvol_target: Target subvolume array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, z_box, y_box, x_box)
        position: Position of centre of subvolumes in base array (n_z_subvolumes, n_y_subvolumes, n_x_subvolumes, 3)
        r_threshold: measure of shift quality. If the correlation between corrected base subvol and fixed target
        subvol is beneath this, we store shift as [nan, nan, nan]
    Returns:
        shift: 2D array, with first dimension referring to subvolume index and final dim referring to shift
        (n_z_subvolumes * n_y_subvolumes * n_x_subvolumes, 3)
        shift_corr: 2D array, with first dimension referring to subvolume index and final dim referring to
        shift_corr coef (n_z_subvolumes * n_y_subvolumes * n_x_subvolumes, 1)
    """
    assert subvol_target.shape == subvol_base.shape, "Base and target subvolumes must have the same shape"
    z_subvolumes, y_subvolumes, x_subvolumes = subvol_base.shape[0], subvol_base.shape[1], subvol_base.shape[2]
    shift = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes, 3))
    shift_corr = np.zeros((z_subvolumes, y_subvolumes, x_subvolumes))
    position = np.reshape(position, (z_subvolumes, y_subvolumes, x_subvolumes, 3))

    for y, x in tqdm(
        np.ndindex(y_subvolumes, x_subvolumes), desc="Computing subvolume shifts", total=y_subvolumes * x_subvolumes
    ):
        shift[:, y, x], shift_corr[:, y, x] = find_z_tower_shifts(
            subvol_base=subvol_base[:, y, x],
            subvol_target=subvol_target[:, y, x],
            position=position[:, y, x].copy(),
            pearson_r_threshold=r_threshold,
        )

    return np.reshape(shift, (shift.shape[0] * shift.shape[1] * shift.shape[2], 3)), np.reshape(
        shift_corr, shift.shape[0] * shift.shape[1] * shift.shape[2]
    )


def find_z_tower_shifts(subvol_base, subvol_target, position, pearson_r_threshold, z_neighbours=1):
    """
    This function takes in 2 split up 3d images along one tower of subvolumes in z and computes shifts for each of these
    to its nearest neighbour subvol
    Args:
        subvol_base: (n_z_subvols, z_box, y_box, x_box) Base subvolume array (this is a z-tower of n_z_subvols base
        subvolumes)
        subvol_target: (n_z_subvols, z_box, y_box, x_box) Target subvolume array (this is a z-tower of n_z_subvols
        target subvolumes)
        position: (n_z_subvols, 3) Position of centre of subvolumes (z, y, x)
        pearson_r_threshold: (float) threshold of correlation used in degenerate cases
        z_neighbours: (int) number of neighbouring sub-volumes to merge with the current sub-volume to compute the shift
    Returns:
        shift: (n_z_subvols, 3) shift of each subvolume in z-tower
        shift_corr: (n_z_subvols, 1) correlation coefficient of each subvolume in z-tower
    """
    position = position.astype(int)
    # for the purposes of this section, we'll take position to be the bottom left corner of the subvolume
    position = position - np.array([subvol_base.shape[1], subvol_base.shape[2], subvol_base.shape[3]]) // 2
    z_subvolumes = subvol_base.shape[0]
    z_box = subvol_base.shape[1]
    shift = np.zeros((z_subvolumes, 3))
    shift_corr = np.zeros(z_subvolumes)

    # this is quite confusing, but what we do here is:
    # 1. merge the target subvolumes in adjacent z-planes
    # 2. create an image of the same shape for the base image, but only fill this with the current subvolume
    # 3. window the whole target merged subvol from beginning to end in z
    # 4. window the base 'merged' subvol from start of current subvolume to the end of current subvolume in z
    for z in range(z_subvolumes):
        # find the start and end z indices of the subvolumes to merge
        z_start, z_end = int(max(0, z - z_neighbours)), int(min(z_subvolumes, z + z_neighbours + 1))
        # merge the target subvolumes from z_start to z_end
        merged_subvol_target = preprocessing.merge_subvols(
            position=np.copy(position[z_start:z_end]), subvol=subvol_target[z_start:z_end]
        )
        # initialise the base subvolume and the windowed base subvolume
        merged_subvol_base = np.zeros_like(merged_subvol_target)
        # populate the merged subvol base image with just the current subvolume
        relative_min_z = position[z, 0] - position[z_start, 0]
        merged_subvol_base[relative_min_z : relative_min_z + z_box] = subvol_base[z]

        # window the images (only windowing the nonzero subset of the base merged image)
        merged_subvol_base_windowed = np.zeros_like(merged_subvol_target)
        merged_subvol_target_windowed = preprocessing.window_image(merged_subvol_target)
        merged_subvol_base_windowed[relative_min_z : relative_min_z + z_box] = preprocessing.window_image(
            subvol_base[z]
        )

        # if one of these images is all zeros, we can't compute the shift, so skip
        image_exists = np.max(merged_subvol_target) != 0 and np.max(merged_subvol_base) != 0
        if not image_exists:
            shift_corr[z] = 0
            shift[z] = np.array([np.nan, np.nan, np.nan])
            continue

        # Now we have the merged subvolumes, we can compute the shift
        shift[z], _, _ = skimage.registration.phase_cross_correlation(
            reference_image=merged_subvol_target_windowed,
            moving_image=merged_subvol_base_windowed,
            upsample_factor=10,
            disambiguate=True,
            overlap_ratio=0.5,
        )
        # compute pearson correlation coefficient
        shift_base = preprocessing.custom_shift(merged_subvol_base, shift[z].astype(int))
        mask = (shift_base != 0) * (merged_subvol_target != 0)
        if np.sum(mask) == 0:
            shift_corr[z] = 0
        else:
            shift_corr[z] = np.corrcoef(shift_base[mask], merged_subvol_target[mask])[0, 1]
        if shift_corr[z] < pearson_r_threshold:
            shift[z] = np.array([np.nan, np.nan, np.nan])

    return shift, shift_corr


def find_zyx_shift(subvol_base, subvol_target, pearson_r_threshold=0.9):
    """
    This function takes in 2 3d images and finds the optimal shift from one to the other. We use a phase cross
    correlation method to find the shift.
    Args:
        subvol_base: Base subvolume array (this will contain a lot of zeroes) (n_z_pixels, n_y_pixels, n_x_pixels)
        subvol_target: Target subvolume array (this will be a merging of subvolumes with neighbouring subvolumes)
        (nz_pixels2, n_y_pixels2, n_x_pixels2) size 2 >= size 1
        pearson_r_threshold: Threshold used to accept a shift as valid (float)

    Returns:
        shift: zyx shift (3,)
        shift_corr: correlation coefficient of shift (float)
    """
    assert subvol_target.shape == subvol_base.shape, "Base and target subvolumes must have the same shape"
    shift, _, _ = skimage.registration.phase_cross_correlation(
        reference_image=subvol_target, moving_image=subvol_base, upsample_factor=10
    )
    alt_shift = np.copy(shift)
    # now anti alias the shift in z. To do this, consider that the other possible aliased z shift is the either one
    # subvolume above or below the current shift. (In theory, we could also consider the subvolume 2 above or below,
    # but this is unlikely to be the case in practice as we are already merging subvolumes)
    if shift[0] > 0:
        alt_shift[0] = shift[0] - subvol_base.shape[0]
    else:
        alt_shift[0] = shift[0] + subvol_base.shape[0]

    # Now we need to compute the correlation coefficient of the shift and the anti aliased shift
    shift_base = preprocessing.custom_shift(subvol_base, shift.astype(int))
    alt_shift_base = preprocessing.custom_shift(subvol_base, alt_shift.astype(int))
    # Now compute the correlation coefficients. First create a mask of the nonzero values
    mask = shift_base != 0
    shift_corr = np.corrcoef(shift_base[mask], subvol_target[mask])[0, 1]
    if np.isnan(shift_corr):
        shift_corr = 0.0
    mask = alt_shift_base != 0
    alt_shift_corr = np.corrcoef(alt_shift_base[mask], subvol_target[mask])[0, 1]
    if np.isnan(alt_shift_corr):
        alt_shift_corr = 0.0
    mask = subvol_base != 0
    base_corr = np.corrcoef(subvol_base[mask], subvol_target[mask])[0, 1]
    if np.isnan(base_corr):
        base_corr = 0.0

    # Now return the shift with the highest correlation coefficient
    if alt_shift_corr > shift_corr:
        shift = alt_shift
        shift_corr = alt_shift_corr
    if base_corr > shift_corr:
        shift = np.array([0, 0, 0])
        shift_corr = base_corr
    # Now check if the correlation coefficient is above the threshold. If not, set the shift to nan
    if shift_corr < pearson_r_threshold:
        shift = np.array([np.nan, np.nan, np.nan])
        shift_corr = np.nanmax([shift_corr, alt_shift_corr, base_corr])

    return shift, shift_corr


def huber_regression(shift, position, predict_shift=True):
    """
    Function to predict shift as a function of position using robust huber regressor. If we do not have >= 3 z-coords
    in position, the z-coords of the affine transform will be estimated as no scaling, and a shift of mean(shift).
    Args:
        shift: n_tiles x 3 ndarray of zyx shifts
        position: n_tiles x 2 ndarray of yx tile coords or n_tiles x 3 ndarray of zyx tile coords
        predict_shift: If True, predict shift as a function of position. If False, predict position as a function of
        position. Default is True. Difference is that if false, we add 1 to each diagonal of the transform matrix.
    Returns:
        transform: 3 x 3 matrix where each row predicts shift of z y z as a function of y index, x index and the final
        row is the offset at 0,0
        or 3
    """
    # We are going to get rid of the shifts where any of the values are nan for regression
    position = position[~np.isnan(shift[:, 0])]
    shift = shift[~np.isnan(shift[:, 0])]
    # Check if we have any shifts to predict
    if len(shift) == 0 and predict_shift:
        transform = np.zeros((3, 4))
        return transform
    elif len(shift) == 0 and not predict_shift:
        transform = np.eye(3, 4)
        return transform
    # Do robust regression
    # Check we have at least 3 z-coords in position
    if len(set(position[:, 0])) <= 2:
        z_coef = np.array([0, 0, 0])
        z_shift = np.mean(shift[:, 0])
    else:
        huber_z = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 0])
        z_coef = huber_z.coef_
        z_shift = huber_z.intercept_
    huber_y = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 1])
    huber_x = HuberRegressor(epsilon=2, max_iter=400, tol=1e-4).fit(X=position, y=shift[:, 2])
    transform = np.vstack(
        (
            np.append(z_coef, z_shift),
            np.append(huber_y.coef_, huber_y.intercept_),
            np.append(huber_x.coef_, huber_x.intercept_),
        )
    )
    if not predict_shift:
        transform += np.eye(3, 4)

    return transform


def procrustes_regression(base_points: np.ndarray, target_points: np.ndarray):
    """
    Perform procrustes analysis to find the affine transform between two sets of points. This will return the best
    orthogonal transformation between the two sets of points. This is useful for finding the rotation and translation
    between two sets of points.
    Args:
        base_points: n_points x 2 np.ndarray of points to transform
        target_points: n_points x 2 np.ndarray of points to transform to

    Returns:
        transform_procrustes: 3x4 np.ndarray, affine transform matrix
    """
    assert base_points.shape == target_points.shape, "Base and target points must have the same shape"
    assert base_points.shape[1] == 2, "Base and target points must be 2D"

    # centre the points
    base_mean, target_mean = np.mean(base_points, axis=0), np.mean(target_points, axis=0)
    base_points_centred = base_points - base_mean
    target_points_centred = target_points - target_mean

    # procrustes analysis uses svd to find optimal rotation
    U, S, Vt = np.linalg.svd(target_points_centred.T @ base_points_centred)
    R = U @ Vt
    angle = np.arccos(R[0, 0])
    shift = target_mean - base_mean
    print(f"Initial angle is {np.round(angle * 180 / np.pi, 2)} degrees and shift is {np.round(shift, 2)}")

    # This shift found takes origin at the centre of mass of the anchor points (base mean),
    # correct for this and make our shift relative to (0, 0)
    shift += (np.eye(2) - R) @ base_mean
    # convert our matrix to a 3 x 4 affine transform matrix (as this will be the starting point for 3d registration)
    transform_procrustes = np.eye(3, 4)
    transform_procrustes[1:3, 1:3] = R
    transform_procrustes[1:, 3] = shift

    return transform_procrustes
