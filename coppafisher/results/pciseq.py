import importlib.resources as importlib_resources
import os
from numbers import Number

import numpy as np
import zarr

from ..filter import radius_normalisation
from ..pipeline import filter_run
from ..plot.results_viewer import background
from ..results.base import MethodData
from ..setup import file_names
from ..setup.notebook import Notebook


def export_pciseq_unfiltered_dapi_image(
    nb: Notebook, config_file_path: str, radius_norm_file: str | None = None
) -> None:
    """
    Save a global, unfiltered anchor DAPI image based on stitch results as uint16.

    The dapi_image.npz file is saved into the directory where the notebook is located. The image data is stored inside
    key `"arr_0"`.

    Args:
        nb (Notebook): the experiment's coppafisher outputted Notebook, must have completed at least up to stitch.
        config_file_path (str): the experiment's config file location.
        radius_norm_file (str or none, optional): file path to the DAPI channel radius normalisation file. See
           `dapi_radius_normalisation_filepath` in coppafisher/setup/default.ini for details. Set to "default" for the
           default radius normalisation (only works when nb.basic_info.dapi_channel == 0).
    """
    nbp_file = file_names.get_file_names(nb.basic_info, config_file_path)

    dapi_image_paths = [
        nbp_file.tile_unfiltered[t][nb.basic_info.anchor_round][nb.basic_info.dapi_channel]
        for t in nb.basic_info.use_tiles
    ]
    dapi_images: list[np.np.ndarray] = []
    for dapi_path in dapi_image_paths:
        try:
            with zarr.ZipStore(dapi_path, mode="r") as dapi_store:
                dapi_images.append(zarr.open_array(dapi_store)[:])
        except (IsADirectoryError, PermissionError):
            dapi_images.append(zarr.open_array(dapi_path, "r")[:])

    radius_norm = None
    if radius_norm_file == "default" and nb.basic_info.dapi_channel == 0:
        radius_norm_file = importlib_resources.files(filter_run.DAPI_CHANNEL_DIR).joinpath(filter_run.DAPI_CHANNEL_NAME)

    if radius_norm_file is not None:
        if not os.path.isfile(radius_norm_file):
            raise FileNotFoundError(f"Failed to find dapi radius norm file at {radius_norm_file}")
        radius_norm = np.load(str(radius_norm_file))["arr_0"]
        radius_normalisation.validate_radius_normalisation(radius_norm, nb.basic_info.tile_sz)

    if radius_norm is not None:
        for i, dapi_image in enumerate(dapi_images):
            dapi_images[i] = radius_normalisation.radius_normalise_image(dapi_image.astype(np.float32), radius_norm)
            dapi_images[i] = dapi_images[i].astype(np.float16)

    fused_image = background.generate_global_image(
        dapi_images, nb.basic_info.use_tiles, nb.basic_info, nb.stitch, dapi_images[0].dtype, silent=False
    )

    file_path = os.path.join(os.path.dirname(nb.directory), "dapi_image_unfiltered.npz")
    if os.path.isfile(file_path):
        print(f"File {file_path} already exists, replacing it...")
        os.remove(file_path)
    np.savez_compressed(file_path, fused_image)

    print(f"DAPI image saved at {file_path}")


def export_pciseq_dapi_image(nb: Notebook) -> None:
    """
    Save a global, filtered anchor DAPI image based on stitch results.

    The dapi_image.npz file is saved into the directory where the notebook is located. The image data is stored inside
    key `"arr_0"`.

    Args:
        nb (Notebook): the experiment's coppafisher outputted Notebook, must have completed at least up to stitch.
    """
    dapi_images = [
        nb.filter.images[t, nb.basic_info.anchor_round, nb.basic_info.dapi_channel] for t in nb.basic_info.use_tiles
    ]
    fused_image = background.generate_global_image(dapi_images, nb.basic_info.use_tiles, nb.basic_info, nb.stitch)

    file_path = os.path.join(os.path.dirname(nb.directory), "dapi_image.npz")
    if os.path.isfile(file_path):
        print(f"File {file_path} already exists, replacing it...")
        os.remove(file_path)
    np.savez_compressed(file_path, fused_image)

    print(f"DAPI image saved at {file_path}")


def export_to_pciseq(
    nb: Notebook,
    method: str,
    score_thresh: float | None = None,
    intensity_thresh: float | None = None,
) -> str:
    """
    Saves a .csv files containing gene spot information compatible with pciSeq. The csv contains:

    - Gene: Name of gene each spot was assigned to.
    - y: y coordinate of each spot in stitched coordinate system.
    - x: x coordinate of each spot in stitched coordinate system.
    - z_stack: z coordinate of each spot in stitched coordinate system (in units of z-pixels).
    - score: the spot's score.
    - intensity: the spot's intensity.

    Only spots which pass the thresholds are saved. The spot positions are positioned relative to
    `nb.stitch.tile_origin.min(0).floor()` so the spots align with the exported dapi image.

    One .csv file is saved for each method: *omp*, *anchor*, and *prob*. The .csv file is saved into the directory
    where the notebook is located.

    Args:
        nb (Notebook): the experiment's coppafisher outputted Notebook, must have completed at least up to call spots.
        method (str): gene calling method. Can be 'omp', 'anchor', or 'prob'.
        score_thresh (float, optional): only include spots with score >= score_thresh. Default: 0.
        intensity_thresh (float, optional): only include spots with colour intensity >= intensity_thresh. Default: 0

    Returns:
        (str): csv_file_path. The file path to the saved csv file.
    """
    if type(nb) is not Notebook:
        raise TypeError(f"nb must be type Notebook, got {type(nb)}")
    if type(method) is not str:
        raise TypeError(f"method must be type str, got {type(method)}")
    if method not in ("prob_init", "prob", "anchor", "omp"):
        raise ValueError(f"Unknown method: {method}")
    if score_thresh is None:
        score_thresh = 0
        print("Using no score threshold")
    if intensity_thresh is None:
        intensity_thresh = 0
        print("Using no intensity threshold")
    if not isinstance(intensity_thresh, Number) or not isinstance(score_thresh, Number):
        raise TypeError("Thresholds must be numbers")
    intensity_thresh = float(intensity_thresh)
    score_thresh = float(score_thresh)

    nbp_omp = None
    if nb.has_page("omp"):
        nbp_omp = nb.omp

    spot_data = MethodData(method, nb.basic_info, nb.stitch, nb.ref_spots, nb.call_spots, nbp_omp)
    spot_data.remove_data_at(np.logical_or(spot_data.score < score_thresh, spot_data.intensity < intensity_thresh))

    file_path = os.path.join(os.path.dirname(nb.directory), f"pciseq_{method}.csv")
    if os.path.isfile(file_path):
        print(f"WARNING: file {file_path} already exists, it will be overwritten")
        os.remove(file_path)

    # Save the global yxz positions with their corresponding gene indices.
    spot_data.save_csv(file_path, nb.call_spots.gene_names)
    print(f"pciSeq file saved for method {method} at " + file_path)

    return file_path
