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
from ..plot.results_viewer import background
from ..setup import file_names
from ..setup.notebook import Notebook
from ..setup.notebook_page import NotebookPage
from ..stitch import base as stitch_base
from . import postprocessing, preprocessing, subvol_registration

CUSTOM_DIR_NAME = "custom"
DAPI_DIR_NAME = "seq"
SUFFIX = ".tif"


def extract_raw(
    nb: Notebook, config_file_path: str, read_dir: str, save_dir: str, use_tiles: list, use_channels: list
) -> None:
    """
    Extract images from the given ND2 file and DAPI images.

    They are saved as .tif files without filtering.

    Args:
        nb (Notebook): notebook of the initial experiment.
        config_file (str): the config file used in the experiment.
        read_dir (str): the directory of the raw data as an ND2 file.
        save_dir (str): the directory where the images are saved.
        use_tiles (list): list of tiles to use.
        use_channels (list): list of channels to use.
    """
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"No config file at {config_file_path}")

    nbp_file_names = file_names.get_file_names(nb.basic_info, config_file_path)

    if type(use_channels) == int:
        use_channels = [use_channels]
    # Check if directories exist.
    assert os.path.isfile(read_dir), f"Raw data file {read_dir} does not exist"
    save_dirs = [save_dir]
    save_dirs += [os.path.join(save_dir, CUSTOM_DIR_NAME, f"channel_{c}") for c in use_channels]
    save_dirs += [os.path.join(save_dir, DAPI_DIR_NAME, f"channel_{nb.basic_info.dapi_channel}")]
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

    # 1. Collect extracted DAPI from seq images.
    for t in tqdm.tqdm(use_tiles, desc="Extracting DAPI from seq images", total=len(use_tiles)):
        y, x = tilepos_yx[t]
        save_path = os.path.join(save_dir, DAPI_DIR_NAME, f"channel_{c_dapi}", f"{x}_{y}{SUFFIX}")
        if os.path.isfile(save_path):
            continue
        # Load raw image.
        raw_path = nbp_file_names.tile_unfiltered[t][nb.basic_info.anchor_round][c_dapi]
        image_raw: npt.NDArray[np.uint16] = zarr.open_array(raw_path, "r")[:]
        # Save image in the format `x_y.tif`.
        tifffile.imwrite(save_path, image_raw)

    # Load ND2 file.
    with nd2.ND2File(read_dir) as f:
        nd2_file = f.to_dask()

    # 2. Extract all relevant channels from the custom images.
    for t in tqdm.tqdm(use_tiles, desc=f"Extracting {read_dir}", total=len(use_tiles)):
        t_files = nd2_file[nd2_indices[t]].compute()
        for c in use_channels:
            y, x = tilepos_yx[t]
            save_path = os.path.join(save_dir, CUSTOM_DIR_NAME, f"channel_{c}", f"{x}_{y}{SUFFIX}")
            if os.path.isfile(save_path):
                continue
            image = np.array(t_files[:, c])
            image = np.rot90(image, k=num_rotations, axes=(1, 2))
            image = image.astype(np.uint16)
            # zyx -> yxz.
            image = image.swapaxes(0, 2).swapaxes(0, 1)
            # Save image in the format x_y.tif.
            tifffile.imwrite(save_path, image)


def fuse_custom_and_dapi(nb: Notebook, extract_dir: str, channel: int) -> np.ndarray[np.float32]:
    """
    Compute the stitches required to combine the custom and anchor-DAPI images together into large images.

    The algorithm is the same as the stitching algorithm for the DAPI images during the pipeline.

    Args:
        nb (Notebook): the experiment notebook.
        extract_dir (str): the directory containing the custom images.
        channels (int): the custom image's channel index.

    Returns:
        Tuple containing:
            - (`(big_im_z x big_im_y x big_im_x) ndarray[float32]`): fused_custom_image. The large, global background
                custom image. The image's origin is relative to `nbp_stitch.tile_origin.min(0)`.
            - (`(big_im_z x big_im_y x big_im_x) ndarray[float32]`): fused_dapi_image. The large, global background
                anchor-DAPI image. The image's origin is relative to `nbp_stitch.tile_origin.min(0)`.
    """
    if type(extract_dir) is not str:
        raise TypeError(f"extract_dir must be a str, got {type(extract_dir)}")
    if type(channel) is not int:
        raise TypeError(f"channel must be an int, got {type(channel)}")
    if not os.path.isdir(extract_dir):
        raise SystemError(f"No extract_dir at {extract_dir}")

    custom_dir = os.path.join(extract_dir, CUSTOM_DIR_NAME, f"channel_{channel}")
    dapi_dir = os.path.join(extract_dir, DAPI_DIR_NAME, f"channel_{nb.basic_info.dapi_channel}")

    tile_indices = []
    tilepositions_yx = []
    custom_images = []
    dapi_images = []
    for dir_entry in os.scandir(custom_dir):
        if not dir_entry.name.endswith(SUFFIX):
            raise ValueError(f"All files must end with {SUFFIX}, but found {os.path.abspath(dir_entry.path)}")

        tilepos_x = int(dir_entry.name.split("_")[0])
        tilepos_y = int(dir_entry.name.split("_")[1][: -len(SUFFIX)])

        tilepos_yx = np.array([tilepos_y, tilepos_x], int)
        tilepositions_yx.append(tilepos_yx)
        is_tile_match: np.ndarray[bool] = (nb.basic_info.tilepos_yx == tilepos_yx).all(1)
        if is_tile_match.sum() != 1:
            raise ValueError(f"Failed to resolve {SUFFIX} file {dir_entry.path}")
        tile_index = np.flatnonzero(is_tile_match).item(0)

        tile_indices.append(tile_index)
        custom_images.append(tifffile.imread(dir_entry.path))
        dapi_images.append(tifffile.imread(os.path.join(dapi_dir, dir_entry.name)))

    tilepositions_yx = np.array(tilepositions_yx, int)
    expected_overlap = nb.stitch.associated_configs["stitch"]["expected_overlap"]

    tile_origins_custom, _, _ = stitch_base.stitch(
        custom_images,
        tilepositions_yx,
        tile_indices,
        nb.basic_info.n_tiles,
        expected_overlap,
    )

    # The DAPI stitch results are taken from the notebook. This is important so that the images are aligned with the
    # exported spot positions.
    tile_origins_dapi = nb.stitch.tile_origin

    nbp_basic = NotebookPage("basic_info")
    nbp_basic.tile_sz = nb.basic_info.tile_sz
    nbp_basic.use_tiles = tuple(tile_indices)
    nbp_basic.use_z = tuple(nb.basic_info.use_z)

    nbp_stitch = NotebookPage("stitch", {"stitch": {"expected_overlap": expected_overlap}})
    nbp_stitch.tile_origin = tile_origins_dapi
    dapi_fused_image = background.generate_global_image(
        dapi_images, tile_indices, nbp_basic, nbp_stitch, np.float32, silent=False
    )

    nbp_stitch = NotebookPage("stitch", {"stitch": {"expected_overlap": expected_overlap}})
    nbp_stitch.tile_origin = tile_origins_custom
    custom_fused_image = background.generate_global_image(
        custom_images, tile_indices, nbp_basic, nbp_stitch, np.float32, silent=False
    )

    # The custom image is cropped/padded with zeros to share the same position and shape of the DAPI fused image.
    custom_fused_image = postprocessing.pad_and_crop_image_to_origin(
        custom_fused_image, np.nanmin(tile_origins_custom, 0), np.nanmin(tile_origins_dapi, 0), dapi_fused_image.shape
    )

    return custom_fused_image, dapi_fused_image


def register_custom_image(
    anchor_dapi: np.ndarray,
    custom_image: np.ndarray,
    reg_parameters: dict,
    downsample_factor_yx: int,
) -> np.ndarray[np.float32]:
    """
    Register custom image to the given DAPI anchor image.

    The new registered and fused custom is saved as "custom_image.tif" to disk. This can then be opened in the Viewer.

    Args:
        anchor_dapi (`(nz x ny x nx) ndarray`): stitched large anchor image.
        custom_image (`(nz x ny x nx) ndarray`): stitched large custom image.
        reg_parameters (dict[str, Any]): dictionary of registration parameters. Keys are:
            * registration_type: str, type of registration to perform (must be 'shift' or 'subvolume')
            if registration_type is 'shift':
                No additional parameters are required.
            if registration_type is 'subvolume':
                * subvolume_size: np.ndarray, size of subvolumes in each dimension (size_z, size_y, size_x).
                * overlap: float, fraction of overlap between subvolumes: 0 <= overlap < 1.
                * r_threshold: float, threshold for correlation coefficient.
        downsample_factor_yx (int, optional): downsample factor for y and x dimensions.

    Returns:
        (`(3 x 4) ndarray[float]`): affine_tranform. The final calculated affine transform for the custom image.
    """
    # Steps are as follows:
    # 1. Manual selection of reference points for shift and rotation correction
    # 2. Local correction for z shifts (done as a global shift correction or by subvolume registration)

    if anchor_dapi.shape != custom_image.shape:
        z_box_anchor, y_box_anchor, x_box_anchor = np.array(anchor_dapi.shape)
        z_box_if, y_box_if, x_box_if = np.array(custom_image.shape)
        z_box, y_box, x_box = max(z_box_anchor, z_box_if), max(y_box_anchor, y_box_if), max(x_box_anchor, x_box_if)
        anchor_dapi_full, custom_full = np.zeros((z_box, y_box, x_box)), np.zeros((z_box, y_box, x_box))
        anchor_dapi_full[:z_box_anchor, :y_box_anchor, :x_box_anchor] = anchor_dapi
        custom_full[:z_box_if, :y_box_if, :x_box_if] = custom_image
        anchor_dapi, custom_image = anchor_dapi_full, custom_full
        del anchor_dapi_full, custom_full

    # 1. Global correction for shift and rotation using procrustes analysis.
    anchor_dapi_2d = np.max(anchor_dapi, axis=0)
    custom_2d = np.max(custom_image, axis=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi_2d, name="anchor_dapi", colormap="red", blending="additive")
    v.add_image(custom_2d, name="custom_image", colormap="green", blending="additive")
    v.add_layer(
        napari.layers.Points(
            data=np.array([]), name="anchor_dapi_points", size=1, border_color=np.zeros((3, 4)), face_color="white"
        )
    )
    v.add_layer(
        napari.layers.Points(
            data=np.array([]), name="custom_image_points", size=1, border_color=np.zeros((3, 4)), face_color="white"
        )
    )
    v.show(block=True)

    # Get user input for shift and rotation.
    base_points = v.layers[2].data
    target_points = v.layers[3].data
    # Calculate the original orthogonal transform.
    transform_initial = subvol_registration.procrustes_regression(base_points, target_points)
    # Now apply the transform to the custom image.
    custom_aligned_initial = scipy.ndimage.affine_transform(custom_image, transform_initial, order=0)

    v = napari.Viewer()
    v.add_image(anchor_dapi, name="anchor_dapi", colormap="red", blending="additive")
    v.add_image(custom_aligned_initial, name="custom_image", colormap="green", blending="additive")
    v.show(block=True)

    # 2. Local correction for shifts
    if reg_parameters["registration_type"] == "shift":
        # shift needs to be shift taking anchor to if, as the first transform was obtained this way
        shift = skimage.registration.phase_cross_correlation(
            reference_image=custom_aligned_initial, moving_image=anchor_dapi
        )[0]
        transform_3d_correction = np.eye(3, 4)
        transform_3d_correction[:, 3] = shift
    elif reg_parameters["registration_type"] == "subvolume":
        # First, split the images into subvolumes.
        z_size, y_size, x_size = reg_parameters["subvolume_size"]
        anchor_subvolumes, position = preprocessing.split_image(
            image=anchor_dapi, subvolume_size=[z_size, y_size, x_size], overlap=reg_parameters["overlap"]
        )
        custom_subvolumes, _ = preprocessing.split_image(
            image=custom_aligned_initial, subvolume_size=[z_size, y_size, x_size], overlap=reg_parameters["overlap"]
        )
        # Now loop through subvolumes and calculate the shifts.
        shift, _ = subvol_registration.find_shift_array(
            anchor_subvolumes, custom_subvolumes, position, r_threshold=reg_parameters["r_threshold"]
        )
        # flatten the position array.
        position = position.reshape(-1, 3)

        # Use these shifts to compute a global affine transform.
        transform_3d_correction = subvol_registration.huber_regression(shift, position, predict_shift=False)
    else:
        raise ValueError(f"Unknown registration type: {reg_parameters['registration_type']}")

    # Plot the transformed image.
    custom_aligned = scipy.ndimage.affine_transform(custom_aligned_initial, transform_3d_correction, order=0)
    v = napari.Viewer()
    v.add_image(anchor_dapi, name="anchor_dapi", colormap="red", blending="additive")
    v.add_image(custom_aligned, name="custom_image", colormap="green", blending="additive")
    v.show(block=True)

    # Now compose the initial and 3d correction transforms.
    transform = (np.vstack((transform_initial, [0, 0, 0, 1])) @ np.vstack((transform_3d_correction, [0, 0, 0, 1])))[
        :3, :
    ]
    # Upsample shift in yx.
    transform[1:, -1] *= downsample_factor_yx

    return transform


def apply_transform(image: np.ndarray, transform: np.ndarray | None, save_dir: str, name: str) -> None:
    if transform is not None:
        image = scipy.ndimage.affine_transform(image, transform, order=3)

    tifffile.imwrite(os.path.join(save_dir, name), image)

    print(f"File saved at {os.path.join(save_dir, name)}")
