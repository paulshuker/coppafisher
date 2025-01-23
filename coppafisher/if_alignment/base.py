import os

import napari
import nd2
import numpy as np
import numpy.typing as npt
import scipy
import skimage
import tifffile
import tqdm
import zarr

from ..extract import raw_nd2
from ..setup.notebook import Notebook
from . import preprocessing, subvol_registration


def extract_raw(nb: Notebook, read_dir: str, save_dir: str, use_tiles: list, use_channels: list) -> None:
    """
    Extract images from the given ND2 file and save them as .tif files without filtering.

    Args:
        nb (Notebook): notebook of the initial experiment.
        read_dir (str): the directory of the raw data as an ND2 file.
        save_dir (str): the directory where the images are saved.
        use_tiles (list): list of tiles to use.
        use_channels (list): list of channels to use.
    """
    if type(use_channels) == int:
        use_channels = [use_channels]
    # Check if directories exist
    assert os.path.isfile(read_dir), f"Raw data file {read_dir} does not exist"
    save_dirs = [save_dir]
    save_dirs += [os.path.join(save_dir, "if", f"channel_{c}") for c in use_channels]
    save_dirs += [os.path.join(save_dir, "seq", f"channel_{nb.basic_info.dapi_channel}")]
    for d in save_dirs:
        if not os.path.isdir(d):
            os.makedirs(d)
    del save_dirs

    # Get NPY and ND2 indices.
    tilepos_yx, tilepos_yx_nd2 = nb.basic_info.tilepos_yx, nb.basic_info.tilepos_yx_nd2
    nd2_reader = raw_nd2.Nd2Reader()
    nd2_indices = [nd2_reader.get_tile_raw_index(t, tilepos_yx_nd2, tilepos_yx) for t in range(nb.basic_info.n_tiles)]
    del nd2_reader
    # Get n_rotations.
    num_rotations = nb.extract.associated_configs["extract"]["num_rotations"]
    c_dapi = nb.basic_info.dapi_channel

    # Load ND2 file.
    with nd2.ND2File(read_dir) as f:
        nd2_file = f.to_dask()

    # 1. Collect extracted DAPI from seq images.
    for t in tqdm(use_tiles, desc="Extracting DAPI from seq images", total=len(use_tiles)):
        y, x = tilepos_yx[t]
        save_path = os.path.join(save_dir, "seq", f"channel_{c_dapi}", f"{x}_{y}.tif")
        if os.path.isfile(save_path):
            continue
        # Load raw image.
        raw_path = nb.file_names.tile_unfiltered[t][nb.basic_info.anchor_round][c_dapi]
        image_raw: npt.NDArray[np.uint16] = zarr.open_array(raw_path, "r")[:]
        # Save image in the format `x_y.tif`.
        tifffile.imwrite(save_path, image_raw)

    # 2. extract all relevant channels from the IF images
    for t in tqdm(use_tiles, desc="Extracting IF images", total=len(use_tiles)):
        for c in use_channels:
            y, x = tilepos_yx[t]
            save_path = os.path.join(save_dir, "if", f"channel_{c}", f"{x}_{y}.tif")
            if os.path.isfile(save_path):
                continue
            # Load image.
            image = np.array(nd2_file[nd2_indices[t], :, c])
            image = np.rot90(image, k=num_rotations, axes=(1, 2))[1:]
            image = image.astype(np.uint16)
            # Save image in the format x_y.tif
            tifffile.imwrite(save_path, image)


def stitch_if_and_dapi(nb: Notebook, extract_dir: str, use_channels: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Stitch the IF images and the DAPI images together using coppafisher's stich results. All relevant files must be
    inside the given extract_dir. Inside extract_dir must be two directories: "if" and "seq" that are produced by the
    function extract_raw above.

    Args:
        nb (Notebook): the notebook with at least `stitch` completed.
        extract_dir (str): the extract directory where the IF and sequence DAPI images are located.
        use_channels (list of int): the channels to use that are in the extract directory.

    Returns:
        Tuple containing:
            - (`(big_im_z x big_im_y x big_im_x) ndarray`): if_fused_image. The stitched together IF images.
            - (`(big_im_z x big_im_y x big_im_x) ndarray`): dapi_fused_image. The stitched together DAPI images.
    """
    if type(nb) is not Notebook:
        raise TypeError(f"nb must be a Notebook, got {type(nb)}")
    if type(extract_dir) is not str:
        raise TypeError(f"extract_dir must be a str, got {type(extract_dir)}")
    if type(use_channels) is not list:
        raise TypeError(f"use_channels must be a list, got {type(use_channels)}")


def register_if(
    anchor_dapi: np.ndarray,
    if_dapi: np.ndarray,
    transform_save_dir: str,
    reg_parameters: dict = None,
    downsample_factor_yx: int = 4,
) -> np.ndarray:
    """
    Register IF image to anchor image.

    anchor_dapi (`(nz x ny x nx) ndarray`): Stitched large anchor image.
    if_dapi (`(nz x ny x nx) ndarray`): Stitched large IF image.
    transform_save_dir (str): directory to save the transform as a .npy file.
    reg_parameters (dict[str, Any]`): Dictionary of registration parameters. Keys are:
        * registration_type: str, type of registration to perform (must be 'shift' or 'subvolume')
        if registration_type is 'shift':
            No additional parameters are required.
        if registration_type is 'subvolume':
            * subvolume_size: np.ndarray, size of subvolumes in each dimension (size_z, size_y, size_x).
            * overlap: float, fraction of overlap between subvolumes: 0 <= overlap < 1.
            * r_threshold: float, threshold for correlation coefficient.
    downsample_factor_yx (int): downsample factor for y and x dimensions.


    Returns:
        (np.ndarray): transform. Affine transform matrix.
    """
    # Steps are as follows:
    # 1. Manual selection of reference points for shift and rotation correction
    # 2. Local correction for z shifts (done as a global shift correction or by subvolume registration)

    if anchor_dapi.shape != if_dapi.shape:
        z_box_anchor, y_box_anchor, x_box_anchor = np.array(anchor_dapi.shape)
        z_box_if, y_box_if, x_box_if = np.array(if_dapi.shape)
        z_box, y_box, x_box = max(z_box_anchor, z_box_if), max(y_box_anchor, y_box_if), max(x_box_anchor, x_box_if)
        anchor_dapi_full, if_dapi_full = np.zeros((z_box, y_box, x_box)), np.zeros((z_box, y_box, x_box))
        anchor_dapi_full[:z_box_anchor, :y_box_anchor, :x_box_anchor] = anchor_dapi
        if_dapi_full[:z_box_if, :y_box_if, :x_box_if] = if_dapi
        anchor_dapi, if_dapi = anchor_dapi_full, if_dapi_full
        del anchor_dapi_full, if_dapi_full

    if reg_parameters is None:
        z_size, y_size, x_size = 16, 512, 512
        reg_parameters = {
            "registration_type": "subvolume",  # 'shift' or 'subvolume'
            "subvolume_size": [z_size, y_size, x_size],
            "overlap": 0.1,
            "r_threshold": 0.8,
        }

    # 1. Global correction for shift and rotation using procrustes analysis
    anchor_dapi_2d = np.max(anchor_dapi, axis=0)
    if_dapi_2d = np.max(if_dapi, axis=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi_2d, name="anchor_dapi", colormap="red", blending="additive")
    v.add_image(if_dapi_2d, name="if_dapi", colormap="green", blending="additive")
    v.add_layer(
        napari.layers.Points(
            data=np.array([]), name="anchor_dapi_points", size=1, edge_color=np.zeros((3, 4)), face_color="white"
        )
    )
    v.add_layer(
        napari.layers.Points(
            data=np.array([]), name="if_dapi_points", size=1, edge_color=np.zeros((3, 4)), face_color="white"
        )
    )
    v.show(block=True)

    # Get user input for shift and rotation
    base_points = v.layers[2].data
    target_points = v.layers[3].data
    # Calculate the original orthogonal transform
    transform_initial = subvol_registration.procrustes_regression(base_points, target_points)
    # Now apply the transform to the IF image
    if_dapi_aligned_initial = scipy.ndimage.affine_transform(if_dapi, transform_initial, order=0)

    v = napari.Viewer()
    v.add_image(anchor_dapi, name="anchor_dapi", colormap="red", blending="additive")
    v.add_image(if_dapi_aligned_initial, name="if_dapi", colormap="green", blending="additive")
    v.show(block=True)

    # 2. Local correction for shifts
    if reg_parameters["registration_type"] == "shift":
        # shift needs to be shift taking anchor to if, as the first transform was obtained this way
        shift = skimage.registration.phase_cross_correlation(
            reference_image=if_dapi_aligned_initial, moving_image=anchor_dapi
        )[0]
        transform_3d_correction = np.eye(3, 4)
        transform_3d_correction[:, 3] = shift

    elif reg_parameters["registration_type"] == "subvolume":
        # First, split the images into subvolumes
        z_size, y_size, x_size = reg_parameters["subvolume_size"]
        anchor_subvolumes, position = preprocessing.split_image(
            image=anchor_dapi, subvolume_size=[z_size, y_size, x_size], overlap=reg_parameters["overlap"]
        )
        if_subvolumes, _ = preprocessing.split_image(
            image=if_dapi_aligned_initial, subvolume_size=[z_size, y_size, x_size], overlap=reg_parameters["overlap"]
        )
        # Now loop through subvolumes and calculate the shifts
        shift, _ = subvol_registration.find_shift_array(
            anchor_subvolumes, if_subvolumes, position, r_threshold=reg_parameters["r_threshold"]
        )
        # flatten the position array
        position = position.reshape(-1, 3)

        # Use these shifts to compute a global affine transform
        transform_3d_correction = subvol_registration.huber_regression(shift, position, predict_shift=False)
    else:
        raise ValueError("Invalid registration type. Must be 'shift' or 'subvolume'")

    # plot the transformed image
    if_dapi_aligned = scipy.ndimage.affine_transform(if_dapi_aligned_initial, transform_3d_correction, order=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi, name="anchor_dapi", colormap="red", blending="additive")
    v.add_image(if_dapi_aligned, name="if_dapi", colormap="green", blending="additive")
    v.show(block=True)

    # Now compose the initial and 3d correction transforms
    transform = (np.vstack((transform_initial, [0, 0, 0, 1])) @ np.vstack((transform_3d_correction, [0, 0, 0, 1])))[
        :3, :
    ]
    # up-sample shift in yx
    transform[1:, -1] *= downsample_factor_yx
    np.save(os.path.join(transform_save_dir, "transform.npy"), transform)

    return transform
