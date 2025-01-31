import os
from numbers import Number

import numpy as np
import pandas as pd
import zarr

from ..plot.results_viewer import background
from ..results.base import MethodData
from ..setup import file_names
from ..setup.notebook import Notebook


def export_pciseq_unfiltered_dapi_image(nb: Notebook, config_file_path: str) -> None:
    """
    Save a global, unfiltered anchor DAPI image based on stitch results as uint16.

    The dapi_image.npz file is saved into the directory where the notebook is located. The image data is stored inside
    key `"arr_0"`.

    Args:
        nb (Notebook): the experiment's coppafisher outputted Notebook, must have completed at least up to stitch.
        config_file_path (str): the experiment's config file location.
    """
    nbp_file = file_names.get_file_names(nb.basic_info, config_file_path)

    dapi_image_paths = [
        nbp_file.tile_unfiltered[t][nb.basic_info.anchor_round][nb.basic_info.dapi_channel]
        for t in nb.basic_info.use_tiles
    ]
    dapi_images = [zarr.open_array(path, "r")[:] for path in dapi_image_paths]
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
    assert type(nb) is Notebook
    assert type(method) is str
    if method not in ("prob", "anchor", "omp"):
        raise ValueError(f"Unknown method: {method}")
    if score_thresh is None:
        score_thresh = 0
        print("Using no score threshold")
    if intensity_thresh is None:
        intensity_thresh = 0
        print("Using no intensity threshold")
    assert isinstance(intensity_thresh, Number), "Thresholds must be numbers"
    assert isinstance(score_thresh, Number), "Thresholds must be numbers"
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
    # Save the global yxz positions with their corresponding gene indices.
    df_to_export = pd.DataFrame()
    df_to_export["Gene"] = nb.call_spots.gene_names[spot_data.gene_no]
    df_to_export["y"] = spot_data.yxz[:, 0]
    df_to_export["x"] = spot_data.yxz[:, 1]
    df_to_export["z_stack"] = spot_data.yxz[:, 2]
    df_to_export["score"] = spot_data.score
    df_to_export.to_csv(file_path, mode="w", index=False)

    print(f"pciSeq file saved for method {method} at " + file_path)

    return file_path
